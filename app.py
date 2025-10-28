import os
import tempfile
import hashlib
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv, find_dotenv

# === Env setup ===
load_dotenv(find_dotenv(), override=False)

# === Streamlit config ===
st.set_page_config(page_title="Ask My PDFs", page_icon="📄", layout="wide")
st.title("📄 Ask My PDFs")
st.write("Upload one or more PDFs and ask questions. I'll retrieve relevant parts and answer with citations.")

# === Guard: OpenAI key ===
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_KEY:
    st.warning("No OPENAI_API_KEY found. Set it in your .env (OPENAI_API_KEY=sk-...) before using the app.")

# === Imports that depend on libs ===
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# ---------- Helpers ----------
def file_md5(file_bytes: bytes) -> str:
    h = hashlib.md5()
    h.update(file_bytes)
    return h.hexdigest()

@st.cache_resource(show_spinner=False)
def get_embeddings():
    # Consider "text-embedding-3-large" if you want higher quality.
    return OpenAIEmbeddings(model="text-embedding-3-small")

@st.cache_resource(show_spinner=False)
def get_llm(model_name: str, temperature: float):
    return ChatOpenAI(model=model_name, temperature=temperature)

def ensure_session_state():
    if "index" not in st.session_state:
        # Persist inside a temp directory unique to this Streamlit session
        st.session_state["persist_dir"] = tempfile.mkdtemp(prefix="pdf_chroma_")
        st.session_state["index"] = None
        st.session_state["doc_hashes"] = set()
        st.session_state["docs_loaded"] = 0
    return st.session_state

def load_pdf_to_docs(tmp_path: str, filename: str):
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    # Stamp filename into metadata for clearer citations
    for p in pages:
        p.metadata = p.metadata or {}
        p.metadata["source"] = filename
    return pages

def chunk_docs(docs, chunk_size=900, chunk_overlap=120):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

def build_or_update_index(chunks, persist_dir):
    embeddings = get_embeddings()
    if st.session_state["index"] is None:
        st.session_state["index"] = Chroma.from_documents(
            chunks, embedding=embeddings, collection_name="pdf_kb", persist_directory=persist_dir
        )
    else:
        st.session_state["index"].add_documents(chunks)
    st.session_state["index"].persist()

# ---------- UI: Upload ----------
ss = ensure_session_state()
uploaded_files = st.file_uploader("Upload PDF(s)", accept_multiple_files=True, type=["pdf"])

col1, col2, col3 = st.columns([1,1,1], gap="small")
with col1:
    k = st.slider("Top-k passages", 2, 10, 4)
with col2:
    model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0)
with col3:
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0)

# Clear index button
if st.button("🧹 Clear index"):
    ss["index"] = None
    ss["doc_hashes"] = set()
    ss["docs_loaded"] = 0
    st.success("Index cleared for this session.")

new_docs = []
if uploaded_files:
    for f in uploaded_files:
        raw = f.read()
        file_hash = file_md5(raw)
        if file_hash in ss["doc_hashes"]:
            st.info(f"Already loaded: {f.name}")
            continue

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name

        try:
            docs = load_pdf_to_docs(tmp_path, f.name)
            st.success(f"Loaded **{f.name}**: {len(docs)} pages")
            new_docs.extend(docs)
            ss["doc_hashes"].add(file_hash)
            ss["docs_loaded"] += len(docs)
        except Exception as e:
            st.error(f"Failed to parse {f.name}: {e}")

# Chunk and index
if new_docs:
    chunks = chunk_docs(new_docs)
    st.info(f"Created **{len(chunks)}** chunks from new uploads")
    build_or_update_index(chunks, ss["persist_dir"])

# ---------- QA ----------
st.subheader("Ask a question about your PDFs")
question = st.text_input("Your question", placeholder="Example: What are the main findings?")

index = ss["index"]
retriever = index.as_retriever(search_kwargs={"k": k}) if index else None
llm = get_llm(model_name, temperature)

qa_chain = None
if retriever:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

ask = st.button("Ask")

if ask:
    if not qa_chain:
        st.warning("Please upload at least one PDF first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                result = qa_chain({"query": question})
                st.markdown("### Answer")
                st.write(result["result"])

                st.markdown("### Sources")
                src_docs = result.get("source_documents", []) or []
                if not src_docs:
                    st.write("No sources returned.")
                else:
                    for i, d in enumerate(src_docs, 1):
                        src = d.metadata.get("source", "uploaded.pdf")
                        page = d.metadata.get("page", "unknown")
                        # page is zero-indexed in many loaders
                        page_str = str(page + 1) if isinstance(page, int) else str(page)
                        st.write(f"{i}. {Path(src).name}, page {page_str}")
            except Exception as e:
                st.error(f"Error generating answer: {e}")
