import os
import re
from typing import List, Dict

import faiss
import fitz  # PyMuPDF
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer


# -----------------------------
# Configuration
# -----------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"

CHUNK_SIZE = 1200       # characters
CHUNK_OVERLAP = 200     # characters
TOP_K = 5


# -----------------------------
# Cached resources
# -----------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def get_groq_api_key() -> str:
    """Read the Groq API key from Streamlit secrets or environment variables."""
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

    return os.getenv("GROQ_API_KEY", "")


# -----------------------------
# PDF processing
# -----------------------------
def extract_pdf_text(pdf_bytes: bytes) -> List[Dict]:
    """Extract text page-by-page from a PDF."""
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
    """Normalize whitespace without destroying paragraph boundaries."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(
    pages: List[Dict],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Dict]:
    """
    Split page text into overlapping character chunks.

    The embedding model tokenizes each chunk internally during encode().
    We also calculate token counts for transparency in the UI.
    """
    chunks = []

    for page in pages:
        text = clean_text(page["text"])

        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(
                    {
                        "text": chunk,
                        "page": page["page"],
                    }
                )

            if end >= len(text):
                break

            start = max(end - overlap, start + 1)

    return chunks


def count_tokens(model, texts: List[str]) -> List[int]:
    """Count tokenizer tokens used by the embedding model."""
    tokenizer = model.tokenizer
    return [
        len(tokenizer.encode(text, add_special_tokens=True))
        for text in texts
    ]


# -----------------------------
# Vector index
# -----------------------------
def build_vector_index(model, chunks: List[Dict]):
    """Create normalized embeddings and a FAISS inner-product index."""
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index, embeddings


def retrieve(
    query: str,
    model,
    index,
    chunks: List[Dict],
    top_k: int = TOP_K,
) -> List[Dict]:
    """Retrieve the most semantically similar chunks."""
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    k = min(top_k, len(chunks))
    scores, indices = index.search(query_embedding, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        result = dict(chunks[idx])
        result["score"] = float(score)
        results.append(result)

    return results


# -----------------------------
# Groq generation
# -----------------------------
def generate_answer(
    question: str,
    retrieved_chunks: List[Dict],
    groq_api_key: str,
) -> str:
    """Generate a grounded answer using Groq's open-weight GPT-OSS model."""
    client = Groq(api_key=groq_api_key)

    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[Source {i} | PDF page {chunk['page']}]\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """You are a helpful document question-answering assistant.

Answer the user's question using ONLY the provided document context.
If the answer cannot be found in the context, say:
"I couldn't find that information in the uploaded document."

Do not invent facts or citations.
When possible, mention the relevant PDF page number(s).
Keep the answer clear and concise.
"""

    user_prompt = f"""DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}
"""

    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1200,
    )

    return completion.choices[0].message.content


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 PDF RAG Assistant")
st.caption(
    "Upload a PDF → extract text → chunk → tokenize → embed → FAISS retrieval → "
    "Groq GPT-OSS answer"
)

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Retrieved chunks", min_value=2, max_value=10, value=TOP_K)
    st.markdown(
        f"""
**LLM:** `{GROQ_MODEL}`  
**Embeddings:** `{EMBEDDING_MODEL}`  
**Vector store:** FAISS (in-memory)  
**Chunk size:** {CHUNK_SIZE} characters  
**Chunk overlap:** {CHUNK_OVERLAP} characters
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
        "GROQ_API_KEY is not configured. Add it to Streamlit Secrets or "
        "set it as an environment variable."
    )
    st.stop()

# Process only when the uploaded file changes.
file_signature = f"{uploaded_file.name}-{uploaded_file.size}"

if st.session_state.get("file_signature") != file_signature:
    st.session_state.clear()
    st.session_state["file_signature"] = file_signature

    with st.spinner("Extracting PDF text..."):
        pdf_bytes = uploaded_file.getvalue()
        pages = extract_pdf_text(pdf_bytes)

    if not pages:
        st.error(
            "No extractable text was found. This may be a scanned/image-only PDF. "
            "OCR would be needed for that type of document."
        )
        st.stop()

    with st.spinner("Creating chunks..."):
        chunks = chunk_text(pages)

    if not chunks:
        st.error("No usable text chunks were created from the PDF.")
        st.stop()

    embedding_model = load_embedding_model()

    with st.spinner("Tokenizing and creating embeddings..."):
        token_counts = count_tokens(
            embedding_model,
            [chunk["text"] for chunk in chunks],
        )

        for chunk, token_count in zip(chunks, token_counts):
            chunk["tokens"] = token_count

        index, _ = build_vector_index(embedding_model, chunks)

    st.session_state["pages"] = pages
    st.session_state["chunks"] = chunks
    st.session_state["index"] = index

pages = st.session_state["pages"]
chunks = st.session_state["chunks"]
index = st.session_state["index"]
embedding_model = load_embedding_model()

col1, col2, col3 = st.columns(3)
col1.metric("PDF pages", len(pages))
col2.metric("Chunks", len(chunks))
col3.metric(
    "Indexed tokens",
    f"{sum(chunk.get('tokens', 0) for chunk in chunks):,}",
)

st.divider()

question = st.chat_input("Ask a question about your PDF...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.spinner("Retrieving relevant passages..."):
        retrieved_chunks = retrieve(
            question,
            embedding_model,
            index,
            chunks,
            top_k=top_k,
        )

    with st.chat_message("assistant"):
        try:
            answer = generate_answer(
                question,
                retrieved_chunks,
                api_key,
            )
            st.write(answer)

            with st.expander("Retrieved context"):
                for i, chunk in enumerate(retrieved_chunks, start=1):
                    st.markdown(
                        f"**Source {i} — Page {chunk['page']} — "
                        f"Similarity {chunk['score']:.3f}**"
                    )
                    st.write(chunk["text"])

        except Exception as exc:
            st.error(f"Groq request failed: {exc}")
