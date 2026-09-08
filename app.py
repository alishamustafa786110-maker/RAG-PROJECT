import os
import re
from typing import Dict, List

import faiss
import fitz
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"

# Smaller, paragraph-aware chunks generally work better for RAG.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# Retrieve more candidates, then rank them using both semantic
# similarity and simple lexical overlap.
VECTOR_TOP_K = 12
FINAL_TOP_K = 7

# Very low vector scores are usually poor matches.
MIN_VECTOR_SCORE = 0.15


# ============================================================
# MODEL
# ============================================================
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


# ============================================================
# API KEY
# ============================================================
def get_groq_api_key() -> str:
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

    return os.getenv("GROQ_API_KEY", "")


# ============================================================
# PDF EXTRACTION
# ============================================================
def extract_pdf_text(pdf_bytes: bytes) -> List[Dict]:
    """
    Extract text page-by-page while preserving page numbers.
    """
    pages = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append(
                    {
                        "page": page_number,
                        "text": text,
                    }
                )

    return pages


def clean_text(text: str) -> str:
    """
    Clean common PDF extraction artifacts while preserving
    paragraph boundaries.
    """
    text = text.replace("\x00", " ")

    # Fix words broken by a PDF line-wrap/hyphen.
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Convert line breaks inside paragraphs into spaces.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Preserve paragraph boundaries.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# CHUNKING
# ============================================================
def split_long_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split long text on natural boundaries where possible.
    """
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    # Prefer sentence boundaries.
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        candidate = f"{current} {sentence}".strip()

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)

            # If one sentence itself is too long, use a sliding window.
            if len(sentence) > chunk_size:
                start = 0

                while start < len(sentence):
                    end = min(start + chunk_size, len(sentence))
                    piece = sentence[start:end].strip()

                    if piece:
                        chunks.append(piece)

                    if end >= len(sentence):
                        break

                    start = max(end - overlap, start + 1)

                current = ""
            else:
                # Keep a small overlap from the previous chunk.
                overlap_text = (
                    current[-overlap:].strip() if overlap > 0 else ""
                )
                current = f"{overlap_text} {sentence}".strip()

    if current:
        chunks.append(current)

    return chunks


def chunk_text(pages: List[Dict]) -> List[Dict]:
    """
    Create page-aware, paragraph/sentence-oriented chunks.
    """
    chunks = []

    for page in pages:
        page_text = clean_text(page["text"])

        # Split on paragraph boundaries first.
        paragraphs = re.split(r"\n\s*\n", page_text)

        page_chunks = []
        current = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()

            if not paragraph:
                continue

            # Add the paragraph to the current chunk when possible.
            candidate = f"{current}\n\n{paragraph}".strip()

            if len(candidate) <= CHUNK_SIZE:
                current = candidate
            else:
                if current:
                    page_chunks.append(current)

                if len(paragraph) <= CHUNK_SIZE:
                    current = paragraph
                else:
                    page_chunks.extend(
                        split_long_text(
                            paragraph,
                            CHUNK_SIZE,
                            CHUNK_OVERLAP,
                        )
                    )
                    current = ""

        if current:
            page_chunks.append(current)

        for chunk_number, chunk in enumerate(page_chunks, start=1):
            chunks.append(
                {
                    "text": chunk,
                    "page": page["page"],
                    "chunk_id": f"page-{page['page']}-chunk-{chunk_number}",
                }
            )

    return chunks


# ============================================================
# TOKENIZATION
# ============================================================
def count_tokens(model, texts: List[str]) -> List[int]:
    tokenizer = model.tokenizer

    return [
        len(tokenizer.encode(text, add_special_tokens=True))
        for text in texts
    ]


# ============================================================
# EMBEDDINGS + FAISS
# ============================================================
def build_vector_index(model, chunks: List[Dict]):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    dimension = embeddings.shape[1]

    # Inner product on normalized vectors ~= cosine similarity.
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index, embeddings


# ============================================================
# SIMPLE LEXICAL RANKING
# ============================================================
def tokenize_for_matching(text: str) -> set:
    """
    Lightweight keyword tokenizer used as a second retrieval signal.
    """
    words = re.findall(r"\b[a-zA-Z0-9]{3,}\b", text.lower())

    stop_words = {
        "the", "and", "for", "that", "this", "with", "from",
        "what", "when", "where", "which", "who", "how", "why",
        "are", "was", "were", "is", "its", "into", "about",
        "have", "has", "had", "their", "there", "then", "than",
        "they", "them", "you", "your", "our", "can", "could",
        "would", "should", "does", "did", "not", "but", "also",
    }

    return {word for word in words if word not in stop_words}


def lexical_score(query: str, text: str) -> float:
    query_terms = tokenize_for_matching(query)
    text_terms = tokenize_for_matching(text)

    if not query_terms or not text_terms:
        return 0.0

    overlap = query_terms.intersection(text_terms)

    return len(overlap) / len(query_terms)


# ============================================================
# HYBRID RETRIEVAL
# ============================================================
def retrieve(
    query: str,
    model,
    index,
    chunks: List[Dict],
    vector_top_k: int = VECTOR_TOP_K,
    final_top_k: int = FINAL_TOP_K,
) -> List[Dict]:
    """
    Hybrid retrieval:
      1. semantic/vector similarity
      2. lexical keyword overlap
      3. combined ranking
    """
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    k = min(vector_top_k, len(chunks))

    vector_scores, indices = index.search(query_embedding, k)

    candidates = []

    for score, idx in zip(vector_scores[0], indices[0]):
        if idx == -1:
            continue

        vector_score = float(score)

        if vector_score < MIN_VECTOR_SCORE:
            continue

        chunk = dict(chunks[idx])

        lex_score = lexical_score(query, chunk["text"])

        # Semantic retrieval gets more weight, while lexical overlap
        # helps questions containing exact names, numbers and terms.
        combined_score = (0.75 * vector_score) + (0.25 * lex_score)

        chunk["vector_score"] = vector_score
        chunk["lexical_score"] = lex_score
        chunk["combined_score"] = combined_score

        candidates.append(chunk)

    candidates.sort(
        key=lambda item: item["combined_score"],
        reverse=True,
    )

    return candidates[:final_top_k]


# ============================================================
# GROQ
# ============================================================
def generate_answer(
    question: str,
    retrieved_chunks: List[Dict],
    groq_api_key: str,
) -> str:
    client = Groq(api_key=groq_api_key)

    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"""
[Source {i}]
PDF page: {chunk["page"]}
Chunk ID: {chunk["chunk_id"]}
Content:
{chunk["text"]}
""".strip()
        )

    context = "\n\n".join(context_parts)

    system_prompt = """
You are a precise Retrieval-Augmented Generation (RAG) assistant.

Your job is to answer questions about an uploaded PDF.

RULES:
1. Use the supplied document context as your primary and only factual source.
2. Do not invent information that is not supported by the context.
3. If the answer is supported by the context, answer directly.
4. Include the PDF page number when possible, using wording such as
   "(Page 5)" or "(Pages 5-6)".
5. If the retrieved context does not contain enough information to answer,
   clearly say that the information could not be found in the uploaded document.
6. If the question asks for a calculation, calculate only from values present
   in the supplied context.
7. If multiple sources/pages support the answer, mention all relevant pages.
8. Do not say that you searched the internet or use outside knowledge.
""".strip()

    user_prompt = f"""
DOCUMENT CONTEXT
================
{context}

USER QUESTION
=============
{question}

ANSWER:
""".strip()

    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.1,
        max_tokens=1500,
    )

    return completion.choices[0].message.content


# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 PDF RAG Assistant")

st.caption(
    "PDF → extraction → chunking → tokenization → embeddings → "
    "hybrid retrieval → Groq GPT-OSS"
)

with st.sidebar:
    st.header("RAG Settings")

    final_top_k = st.slider(
        "Final context chunks",
        min_value=3,
        max_value=10,
        value=FINAL_TOP_K,
    )

    vector_top_k = st.slider(
        "Vector candidates",
        min_value=5,
        max_value=20,
        value=VECTOR_TOP_K,
    )

    st.markdown(
        f"""
**LLM:** `{GROQ_MODEL}`

**Embedding model:** `{EMBEDDING_MODEL}`

**Vector store:** FAISS (in-memory)

**Chunk size:** `{CHUNK_SIZE}` characters

**Chunk overlap:** `{CHUNK_OVERLAP}` characters
"""
    )

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    help="The PDF is processed in memory for the current Streamlit session.",
)

if uploaded_file is None:
    st.info("Upload a PDF to build the RAG index.")
    st.stop()

api_key = get_groq_api_key()

if not api_key:
    st.error(
        "GROQ_API_KEY is not configured. Add it to Streamlit Secrets "
        "or set it as an environment variable."
    )
    st.stop()


# ============================================================
# BUILD INDEX
# ============================================================
file_signature = f"{uploaded_file.name}-{uploaded_file.size}"

if st.session_state.get("file_signature") != file_signature:
    # Only remove our previous RAG data.
    for key in [
        "pages",
        "chunks",
        "index",
        "file_signature",
    ]:
        st.session_state.pop(key, None)

    st.session_state["file_signature"] = file_signature

    with st.spinner("Extracting PDF text..."):
        pdf_bytes = uploaded_file.getvalue()
        pages = extract_pdf_text(pdf_bytes)

    if not pages:
        st.error(
            "No extractable text was found. This may be a scanned/image-only "
            "PDF. OCR is required for scanned PDFs."
        )
        st.stop()

    with st.spinner("Creating intelligent chunks..."):
        chunks = chunk_text(pages)

    if not chunks:
        st.error("No usable text chunks were created.")
        st.stop()

    embedding_model = load_embedding_model()

    with st.spinner("Tokenizing and creating embeddings..."):
        token_counts = count_tokens(
            embedding_model,
            [chunk["text"] for chunk in chunks],
        )

        for chunk, token_count in zip(chunks, token_counts):
            chunk["tokens"] = token_count

        index, _ = build_vector_index(
            embedding_model,
            chunks,
        )

    st.session_state["pages"] = pages
    st.session_state["chunks"] = chunks
    st.session_state["index"] = index


pages = st.session_state["pages"]
chunks = st.session_state["chunks"]
index = st.session_state["index"]

embedding_model = load_embedding_model()


# ============================================================
# DOCUMENT STATS
# ============================================================
col1, col2, col3, col4 = st.columns(4)

col1.metric("PDF pages", len(pages))
col2.metric("Chunks", len(chunks))
col3.metric(
    "Indexed tokens",
    f"{sum(chunk.get('tokens', 0) for chunk in chunks):,}",
)
col4.metric(
    "Vector dimension",
    "384",
)

st.divider()


# ============================================================
# QUESTION ANSWERING
# ============================================================
question = st.chat_input(
    "Ask a question about your PDF..."
)

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.spinner("Searching the document..."):
        retrieved_chunks = retrieve(
            question,
            embedding_model,
            index,
            chunks,
            vector_top_k=vector_top_k,
            final_top_k=final_top_k,
        )

    if not retrieved_chunks:
        st.warning(
            "I couldn't find sufficiently relevant passages in the "
            "uploaded document for this question."
        )
        st.stop()

    with st.chat_message("assistant"):
        try:
            answer = generate_answer(
                question,
                retrieved_chunks,
                api_key,
            )

            st.write(answer)

            with st.expander(
                f"🔎 Retrieved context ({len(retrieved_chunks)} chunks)"
            ):
                for i, chunk in enumerate(
                    retrieved_chunks,
                    start=1,
                ):
                    st.markdown(
                        f"""
**Source {i} — Page {chunk["page"]}**

- Vector similarity: `{chunk["vector_score"]:.3f}`
- Keyword overlap: `{chunk["lexical_score"]:.3f}`
- Combined score: `{chunk["combined_score"]:.3f}`
- Tokens: `{chunk.get("tokens", "N/A")}`
"""
                    )

                    st.write(chunk["text"])

                    if i < len(retrieved_chunks):
                        st.divider()

        except Exception as exc:
            st.error(f"Groq request failed: {exc}")

