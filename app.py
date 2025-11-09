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
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate

# -----------------------
# Config & API key loading
# -----------------------
from pathlib import Path
import streamlit as st

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
LLM_MODEL = "gpt-4o-mini"        # fast & affordable; swap if you prefer
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

# Save uploads to a temp folder (Streamlit Cloud allows ephemeral disk during the session)
TMP_DIR = Path("uploads")
TMP_DIR.mkdir(exist_ok=True)

pdf_paths: List[str] = []
for f in uploaded_files:
    dest = TMP_DIR / f.name
    with open(dest, "wb") as out:
        out.write(f.getbuffer())
    pdf_paths.append(str(dest))

# Create a fingerprint to cache the vector store per unique set of files
def files_fingerprint(paths: List[str]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.encode("utf-8"))
        h.update(str(Path(p).stat().st_size).encode("utf-8"))
    return h.hexdigest()[:16]

fingerprint = files_fingerprint(pdf_paths)

@st.cache_resource(show_spinner="Indexing documents…")
def build_retriever(paths: list[str], cache_key: str):
    # cache_key is only used so Streamlit’s cache invalidates when files change
    from pathlib import Path
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma

    docs = []
    for p in paths:
        # defensively skip any non-pdf or missing path
        if not p.lower().endswith(".pdf") or not Path(p).exists():
            continue
        loader = PyPDFLoader(p)
        docs.extend(loader.load())

    if not docs:
        raise ValueError("No valid PDF pages were loaded.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings)  # note: embedding=
    return vectordb.as_retriever(search_kwargs={"k": TOP_K})

retriever = build_retriever(pdf_paths, fingerprint)

if any(not Path(p).exists() for p in pdf_paths):
    st.error("Upload failed; please try again.")
    st.stop()

# -----------------------
# QA Chain
# -----------------------
llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# -----------------------
# Ask a question
# -----------------------
query = st.text_input("💬 Ask your HR question:")
if st.button("Get Answer") and query:
    with st.spinner("Thinking…"):
        result = qa({"query": query})
        answer = result["result"]
        sources = result.get("source_documents", [])

        # Format source page numbers (1-indexed)
        pages = sorted(
            set(int(s.metadata.get("page", -1)) + 1 for s in sources if s.metadata.get("page", -1) >= 0)
        )
        if pages:
            answer += "\n\nSources: " + ", ".join(f"p.{p}" for p in pages)

        st.success(answer)
