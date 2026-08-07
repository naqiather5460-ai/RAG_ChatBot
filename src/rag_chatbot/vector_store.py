"""
Vector Store module.
Splits loaded documents into chunks, embeds them, and stores them in ChromaDB.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.rag_chatbot.config import (
    CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME, CHROMA_DB_DIR
)


def split_documents(documents):
    """
    Splits loaded documents into smaller overlapping chunks.
    RecursiveCharacterTextSplitter tries to split on paragraph breaks first,
    then sentences, then words -- keeping text as coherent as possible.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"[VectorStore] Split into {len(chunks)} chunks.")
    return chunks


def get_embedding_model():
    """
    Loads the free, local HuggingFace embedding model.
    Downloads automatically the first time this runs (~80MB).
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def build_vector_store(chunks):
    """
    Embeds the given chunks and stores them in ChromaDB on disk.
    Returns the Chroma vector store object.
    """
    embeddings = get_embedding_model()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DB_DIR),
    )
    print(f"[VectorStore] Stored {len(chunks)} chunks in ChromaDB at {CHROMA_DB_DIR}")
    return vector_store