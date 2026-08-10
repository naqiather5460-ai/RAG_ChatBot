from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from src.rag_chatbot.retriever import load_vector_store, retrieve_relevant_chunks
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

    def __init__(self):
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=LLM_MODEL_NAME,
            temperature=0.2,
        )
        self.vector_store = load_vector_store()
        self.chat_history = []  # stores (question, answer) pairs

    def _format_history(self):
        """Formats the conversation so far into readable text for the prompt."""
        if not self.chat_history:
            return "No previous conversation."

        formatted = []
        for question, answer in self.chat_history:
            formatted.append(f"User: {question}\nAssistant: {answer}")
        return "\n".join(formatted)

    def ask(self, question: str) -> str:
        """
        Answers a question using RAG: retrieves relevant chunks, builds a prompt,
        and generates an answer via the LLM.
        """
        # --- Retrieval ---
        relevant_chunks = retrieve_relevant_chunks(question, self.vector_store)
        context = "\n\n".join(chunk.page_content for chunk in relevant_chunks)

        # --- Prompt construction ---
        prompt = PROMPT_TEMPLATE.format(
            context=context,
            history=self._format_history(),
            question=question,
        )

        # --- Generation ---
        response = self.llm.invoke(prompt)
        answer = response.content

        # --- Update conversation memory ---
        self.chat_history.append((question, answer))

        return answer