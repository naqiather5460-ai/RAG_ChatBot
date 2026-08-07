"""
Retriever module.
Given a question, searches ChromaDB for the most relevant document chunks.
"""

from langchain_chroma import Chroma

from src.rag_chatbot.vector_store import get_embedding_model
from src.rag_chatbot.config import CHROMA_DB_DIR, RETRIEVAL_TOP_K


def load_vector_store():
    """
    Loads the existing ChromaDB vector store from disk (doesn't rebuild it --
    just reconnects to what's already there).
    """
    embeddings = get_embedding_model()
    vector_store = Chroma(
        persist_directory=str(CHROMA_DB_DIR),
        embedding_function=embeddings,
    )
    return vector_store


def retrieve_relevant_chunks(query: str, vector_store=None):
    """
    Given a question, returns the top K most relevant chunks from ChromaDB.
    """
    if vector_store is None:
        vector_store = load_vector_store()

    results = vector_store.similarity_search(query, k=RETRIEVAL_TOP_K)

    print(f"[Retriever] Found {len(results)} relevant chunks for query: '{query}'")
    return results