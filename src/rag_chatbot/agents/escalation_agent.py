"""
Escalation Agent.
Handles ESCALATION-category questions -- complaints, disputes, or sensitive
reports. Deliberately does NOT attempt to generate an answer; flags the
request for human HR staff instead. This is a responsible-AI design choice:
some requests should never be auto-answered by an LLM.
"""


class EscalationAgent:
    """Flags sensitive requests for human review instead of auto-answering."""

    def handle(self, question: str) -> str:
        return (
            "This looks like it may need personal attention from our HR team "
            "rather than an automated response. I've flagged this for a human "
            "HR representative to follow up with you directly. "
            f"(Logged request: \"{question}\")"
        )