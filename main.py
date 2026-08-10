"""
RAG Chatbot - Command Line Interface.
Run with: python main.py
"""

from src.rag_chatbot.document_loader import load_all_documents
from src.rag_chatbot.vector_store import split_documents, build_vector_store
from src.rag_chatbot.chatbot import RAGChatbot
from src.rag_chatbot.config import DOCUMENTS_DIR


def rebuild_index():
    """Re-ingests all documents and rebuilds the ChromaDB vector store from scratch."""
    print("\n[Main] Rebuilding vector store from documents in data/documents/...")
    docs = load_all_documents(DOCUMENTS_DIR)
    chunks = split_documents(docs)
    build_vector_store(chunks)
    print("[Main] Vector store rebuilt successfully.\n")


def main():
    print("=" * 60)
    print("RAG Chatbot - Ask questions about your documents")
    print("Type 'rebuild' to re-index documents, 'quit' to exit")
    print("=" * 60)

    bot = RAGChatbot()

    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if user_input.lower() == "rebuild":
            rebuild_index()
            bot = RAGChatbot()  # reload with the freshly rebuilt store
            continue

        answer = bot.ask(user_input)
        print(f"\nBot: {answer}")


if __name__ == "__main__":
    main()