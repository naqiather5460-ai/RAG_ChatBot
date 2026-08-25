"""
Router Agent.
Classifies an incoming question into one of three categories, deciding
which specialist agent should handle it. Does not answer questions itself --
its only job is routing.
"""

from langchain_groq import ChatGroq
from src.rag_chatbot.config import GROQ_API_KEY, LLM_MODEL_NAME

ROUTER_PROMPT = """You are a routing classifier for an HR assistant system.
Classify the user's question into EXACTLY ONE of these three categories:

POLICY - questions about company policies, benefits, rules, procedures
  (e.g. "how many vacation days", "what is the security policy")

DIRECTORY - questions asking to look up a specific employee or department
  (e.g. "who works in engineering", "what department is Sara in")

ESCALATION - complaints, harassment reports, disputes, or anything that
  should be handled by a human, not an AI
  (e.g. "I want to report an issue with my manager", "I have a complaint")

Respond with ONLY one word: POLICY, DIRECTORY, or ESCALATION. No explanation.

Question: {question}
Category:"""


class RouterAgent:
    """Classifies questions to determine which specialist agent should respond."""

    VALID_CATEGORIES = {"POLICY", "DIRECTORY", "ESCALATION"}

    def __init__(self):
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=LLM_MODEL_NAME,
            temperature=0,  # we want consistent, deterministic routing, not creativity
        )

    def route(self, question: str) -> str:
        """
        Classifies a question and returns one of: 'POLICY', 'DIRECTORY', 'ESCALATION'.
        Falls back to 'POLICY' if the model returns something unexpected.
        """
        prompt = ROUTER_PROMPT.format(question=question)
        response = self.llm.invoke(prompt)
        category = response.content.strip().upper()

        if category not in self.VALID_CATEGORIES:
            print(f"[RouterAgent] Unexpected response '{category}', defaulting to POLICY")
            return "POLICY"

        print(f"[RouterAgent] '{question}' -> {category}")
        return category