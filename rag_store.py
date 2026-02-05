import os
from secret_key import OPENAI_API_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader


# ---------------- LOAD RESUME PDF ----------------
def load_pdf(path: str):
    loader = PyPDFLoader(path)
    return loader.load()


# ---------------- BUILD VECTOR STORE ----------------
def build_vector_store(docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


# ---------------- RAG QUERY ----------------
def rag_query(vectorstore, query: str):
    docs = vectorstore.similarity_search(query, k=5)
    return "\n".join([d.page_content for d in docs])