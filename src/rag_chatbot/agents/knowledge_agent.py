"""
Knowledge Agent.
Wraps the existing RAGChatbot to answer policy/benefits/security questions
using retrieval-augmented generation over the indexed HR documents.
"""

from src.rag_chatbot.chatbot import RAGChatbot
from src.rag_chatbot.vector_store import VectorStoreManager


class KnowledgeAgent:
    """Handles POLICY-category questions using the existing RAG pipeline."""

    def __init__(self, vector_store_manager: VectorStoreManager, top_k: int):
        self.chatbot = RAGChatbot(vector_store_manager, top_k=top_k)

    def handle(self, question: str) -> str:
        return self.chatbot.ask(question)

    def reload(self) -> None:
        """Reconnects to the vector store after a rebuild."""
        self.chatbot.reload_vector_store()