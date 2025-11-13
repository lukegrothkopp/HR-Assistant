# app.py
import os
import hashlib
from pathlib import Path
from typing import List

import streamlit as st
from dotenv import load_dotenv

# LangChain / OpenAI wrappers
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# -----------------------
# Config & API key loading
# -----------------------
ASSETS = Path(__file__).parent / "assets"
FAVICON = ASSETS / "favicon.png"       # tab icon
HEADER_LOGO = ASSETS / "header_logo.png"  # page header logo

st.set_page_config(
    page_title="Grothko Consulting HR Assistant",
    page_icon=str(FAVICON) if FAVICON.exists() else "🤖",
    layout="wide"
)

# Header
left, right = st.columns([0.12, 0.88])
with left:
    if HEADER_LOGO.exists():
        st.image(str(HEADER_LOGO), width=64)
with right:
    st.title("Grothko Consulting HR Assistant")
    st.caption(
        "Self-hosted HR policy chatbot. Ingest PDFs, create embeddings, and query in natural language. "
        "Streamlit UI, LangChain retriever, Chroma vector store, OpenAI models."
    )

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.warning("Set OPENAI_API_KEY in your environment or Streamlit Cloud Secrets before asking questions.")
else:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# -----------------------
# App settings
# -----------------------
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5

SYSTEM_PROMPT = """You are a helpful HR assistant for employees.
Answer the user's question ONLY using the provided context from uploaded HR policy documents.
If the answer is not in the context, say you don't know.
Cite page numbers when helpful. Keep answers concise and professional."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:")
    ]
)

# -----------------------
# File upload UI
# -----------------------
uploaded_files = st.file_uploader(
    "📄 Upload one or more HR policy PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Upload PDF(s) to get started.")
    st.stop()

# Save uploads to a temp folder
TMP_DIR = Path("uploads")
TMP_DIR.mkdir(exist_ok=True)

pdf_paths: List[str] = []
for f in uploaded_files:
    dest = TMP_DIR / f.name
    with open(dest, "wb") as out:
        out.write(f.getbuffer())
    pdf_paths.append(str(dest))

def files_fingerprint(paths: List[str]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.encode("utf-8"))
        try:
            h.update(str(Path(p).stat().st_size).encode("utf-8"))
        except FileNotFoundError:
            pass
    return h.hexdigest()[:16]

fingerprint = files_fingerprint(pdf_paths)

@st.cache_resource(show_spinner="Indexing documents…")
def build_retriever(paths: List[str], cache_key: str):
    # Fresh imports here keep the cached object insulated from top-level changes
    from pathlib import Path as _Path
    from langchain_community.document_loaders import PyPDFLoader as _PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter as _Splitter
    from langchain_openai import OpenAIEmbeddings as _Emb
    from langchain_community.vectorstores import Chroma as _Chroma

    docs = []
    for p in paths:
        if not p.lower().endswith(".pdf") or not _Path(p).exists():
            continue
        loader = _PyPDFLoader(p)
        docs.extend(loader.load())

    if not docs:
        raise ValueError("No valid PDF pages were loaded.")

    splitter = _Splitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)

    embeddings = _Emb(model=EMBED_MODEL)
    vectordb = _Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vectordb.as_retriever(search_kwargs={"k": TOP_K})

retriever = build_retriever(pdf_paths, fingerprint)

if any(not Path(p).exists() for p in pdf_paths):
    st.error("Upload failed; please try again.")
    st.stop()

# -----------------------
# Minimal “manual” QA (no helper imports)
# -----------------------
llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

def answer_with_sources(question: str):
    # 1) retrieve – handle both "new" and "old" retriever APIs
    if hasattr(retriever, "invoke"):
        try:
            docs = retriever.invoke(question)
        except TypeError:
            # some Runnable retrievers expect a dict input
            docs = retriever.invoke({"input": question})
    elif hasattr(retriever, "get_relevant_documents"):
        docs = retriever.get_relevant_documents(question)
    else:
        raise RuntimeError(
            "Retriever does not support 'invoke' or 'get_relevant_documents'. "
            "Check your LangChain versions."
        )

    if not docs:
        return "I couldn't find anything in the uploaded HR documents that answers that question."

    # 2) build context
    context = "\n\n---\n\n".join(d.page_content for d in docs)

    # 3) call LLM with the prompt
    messages = prompt.format_messages(question=question, context=context)
    ai_msg = llm.invoke(messages)
    answer = ai_msg.content if hasattr(ai_msg, "content") else str(ai_msg)

    # 4) collect page numbers (1-indexed)
    pages = sorted(
        set(int(d.metadata.get("page", -1)) + 1 for d in docs if d.metadata.get("page", -1) >= 0)
    )
    if pages:
        answer += "\n\nSources: " + ", ".join(f"p.{p}" for p in pages)
    return answer

# -----------------------
# Ask a question
# -----------------------
query = st.text_input("💬 Ask your HR question:")
if st.button("Get Answer") and query:
    with st.spinner("Thinking…"):
        st.success(answer_with_sources(query))
