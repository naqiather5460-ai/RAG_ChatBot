"""
RAGChatbot module.
Ties together retrieval, prompt engineering, and the LLM for conversational Q&A.
"""

from langchain_groq import ChatGroq

from src.rag_chatbot.retriever import Retriever
from src.rag_chatbot.vector_store import VectorStoreManager
from src.rag_chatbot.config import GROQ_API_KEY, LLM_MODEL_NAME

PROMPT_TEMPLATE = """You are a helpful assistant answering questions based on the provided context.

Use ONLY the information in the context below to answer the question. If the answer isn't in the context, say "I don't have enough information to answer that" instead of guessing.

Context:
{context}

Conversation History:
{history}

Question: {question}

Answer:"""


class RAGChatbot:
    """Ties together retrieval, prompt engineering, and the LLM for conversational Q&A."""

    def __init__(self, vector_store_manager: VectorStoreManager, top_k: int):
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=LLM_MODEL_NAME,
            temperature=0.2,
        )
        self.vector_store_manager = vector_store_manager
        self.retriever = Retriever(
            vector_store=vector_store_manager.load_vector_store(),
            top_k=top_k,
        )
        self.chat_history: list[tuple[str, str]] = []

    def _format_history(self) -> str:
        """Formats the conversation so far into readable text for the prompt."""
        if not self.chat_history:
            return "No previous conversation."
        formatted = [f"User: {q}\nAssistant: {a}" for q, a in self.chat_history]
        return "\n".join(formatted)

    def reload_vector_store(self) -> None:
        """Reconnects the retriever to the vector store after a rebuild (new documents indexed)."""
        self.retriever = Retriever(
            vector_store=self.vector_store_manager.load_vector_store(),
            top_k=self.retriever.top_k,
        )

    def ask(self, question: str) -> str:
        """
        Answers a question using RAG: retrieves relevant chunks, builds a prompt,
        and generates an answer via the LLM.
        """
        relevant_chunks = self.retriever.retrieve_relevant_chunks(question)
        context = "\n\n".join(chunk.page_content for chunk in relevant_chunks)

        prompt = PROMPT_TEMPLATE.format(
            context=context,
            history=self._format_history(),
            question=question,
        )

        response = self.llm.invoke(prompt)
        answer = response.content

        self.chat_history.append((question, answer))
        return answer