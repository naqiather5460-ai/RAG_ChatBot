"""
Company HR & Policy Assistant - Multi-Agent System

Case Study: An internal assistant that answers employee questions about
company policies, benefits, employee lookups, and sensitive HR matters.
A Router Agent classifies each question and directs it to one of three
specialists:
  - Knowledge Agent   -> policy/benefits/security questions (RAG pipeline)
  - Directory Agent    -> employee/department lookups (structured query)
  - Escalation Agent    -> complaints/sensitive reports (flagged for a human,
                            never auto-answered by the LLM)

Run with: python main.py
"""

from src.rag_chatbot.document_loader import DocumentLoader
from src.rag_chatbot.vector_store import VectorStoreManager
from src.rag_chatbot.agents.router_agent import RouterAgent
from src.rag_chatbot.agents.knowledge_agent import KnowledgeAgent
from src.rag_chatbot.agents.directory_agent import DirectoryAgent
from src.rag_chatbot.agents.escalation_agent import EscalationAgent
from src.rag_chatbot.graph import build_graph
from src.rag_chatbot.config import (
    DOCUMENTS_DIR, CHROMA_DB_DIR, EMBEDDING_MODEL_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVAL_TOP_K,
)


class ChatbotApplication:
    """Orchestrates the multi-agent HR assistant: indexing, routing, and the chat loop."""

    def __init__(self):
        self.loader = DocumentLoader(DOCUMENTS_DIR)
        self.vector_store_manager = VectorStoreManager(
            chroma_db_dir=CHROMA_DB_DIR,
            embedding_model_name=EMBEDDING_MODEL_NAME,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        # --- Specialist agents ---
        self.router_agent = RouterAgent()
        self.knowledge_agent = KnowledgeAgent(self.vector_store_manager, top_k=RETRIEVAL_TOP_K)
        self.directory_agent = DirectoryAgent(DOCUMENTS_DIR / "employee_directory.xlsx")
        self.escalation_agent = EscalationAgent()

        # --- LangGraph orchestration wiring the agents together ---
        self.graph = build_graph(
            self.knowledge_agent, self.directory_agent,
            self.escalation_agent, self.router_agent
        )

    def rebuild_index(self) -> None:
        """Re-ingests all documents and rebuilds the ChromaDB vector store from scratch."""
        print("\n[App] Rebuilding vector store from documents in data/documents/...")
        docs = self.loader.load_all_documents()
        chunks = self.vector_store_manager.split_documents(docs)
        self.vector_store_manager.build_vector_store(chunks)
        self.knowledge_agent.reload()
        print("[App] Vector store rebuilt successfully.\n")

    def run(self) -> None:
        """Starts the interactive command-line chat loop, routed through the multi-agent graph."""
        print("=" * 60)
        print("Company HR & Policy Assistant (Multi-Agent)")
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

            result = self.graph.invoke({"question": user_input, "category": "", "answer": ""})
            print(f"\nBot [{result['category']}]: {result['answer']}")


if __name__ == "__main__":
    app = ChatbotApplication()
    app.run()