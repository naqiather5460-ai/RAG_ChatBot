"""
Retriever module.
Given a question, searches the vector store for the most relevant document chunks.
"""

from langchain_core.documents import Document
from langchain_chroma import Chroma


class Retriever:
    """Performs semantic similarity search against a Chroma vector store."""

    def __init__(self, vector_store: Chroma, top_k: int):
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve_relevant_chunks(self, query: str) -> list[Document]:
        """Given a question, returns the top-k most relevant chunks from the vector store."""
        results = self.vector_store.similarity_search(query, k=self.top_k)
        print(f"[Retriever] Found {len(results)} relevant chunks for query: '{query}'")
        return results