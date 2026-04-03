from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import shutil
import os
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def load_source(source:str):

    if source.startswith("http"):
        loader = WebBaseLoader(source)
    elif source.endswith(".docx"):
        loader = Docx2txtLoader(source)
    elif source.endswith(".pdf"):
        loader = PyMuPDFLoader(source)
    else:
        raise ValueError(f"Unsupported source: {source}")
    
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from {source}")
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 100,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks

def store_embeddings(chunks):

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma.from_documents(
        documents = chunks,
        embedding = embedding_model,
        persist_directory=CHROMA_PATH

    )
    print(f"Stored {len(chunks)} chunks in ChromaDB at {CHROMA_PATH}")
    return vectorstore

def ingest(source:str):
    documents = load_source(source)
    chunks = split_documents(documents)
    vectorstore = store_embeddings(chunks)
    
    return vectorstore

