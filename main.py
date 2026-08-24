"""
RAG Chatbot - Command Line Interface
Company HR & Policy Assistant

Case Study: An internal assistant that answers employee questions about
company policies, benefits, and security procedures by retrieving answers
directly from official HR documents (PDF, DOCX, TXT, and Excel formats),
rather than relying on unverified general knowledge.

Run with: python main.py
"""

from src.rag_chatbot.document_loader import DocumentLoader
from src.rag_chatbot.vector_store import VectorStoreManager
from src.rag_chatbot.chatbot import RAGChatbot
from src.rag_chatbot.config import (
    DOCUMENTS_DIR, CHROMA_DB_DIR, EMBEDDING_MODEL_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVAL_TOP_K,
)


class ChatbotApplication:
    """Orchestrates the RAG chatbot application: indexing, rebuilding, and the chat loop."""

    def __init__(self):
        self.loader = DocumentLoader(DOCUMENTS_DIR)
        self.vector_store_manager = VectorStoreManager(
            chroma_db_dir=CHROMA_DB_DIR,
            embedding_model_name=EMBEDDING_MODEL_NAME,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        self.chatbot = RAGChatbot(self.vector_store_manager, top_k=RETRIEVAL_TOP_K)

    def rebuild_index(self) -> None:
        """Re-ingests all documents and rebuilds the ChromaDB vector store from scratch."""
        print("\n[App] Rebuilding vector store from documents in data/documents/...")
        docs = self.loader.load_all_documents()
        chunks = self.vector_store_manager.split_documents(docs)
        self.vector_store_manager.build_vector_store(chunks)
        self.chatbot.reload_vector_store()
        print("[App] Vector store rebuilt successfully.\n")

    def run(self) -> None:
        """Starts the interactive command-line chat loop."""
        print("=" * 60)
        print("Company HR & Policy Assistant")
        print("Type 'rebuild' to re-index documents, 'quit' to exit")
        print("=" * 60)

        while True:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                print("Goodbye!")
                break
            if user_input.lower() == "rebuild":
                self.rebuild_index()
                continue

            answer = self.chatbot.ask(user_input)
            print(f"\nBot: {answer}")


if __name__ == "__main__":
    app = ChatbotApplication()
    app.run()