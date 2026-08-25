"""
Directory Agent.
Handles DIRECTORY-category questions by querying the employee directory
directly as structured data, instead of using semantic retrieval. This is
a deliberate design choice: structured, tabular data is better served by
precise, direct queries than by RAG-style semantic search.
"""

import pandas as pd
from pathlib import Path


class DirectoryAgent:
    """Answers employee/department lookup questions using structured data."""

    def __init__(self, employee_file: Path):
        self.df = pd.read_excel(employee_file)

    def handle(self, question: str) -> str:
        question_lower = question.lower()

        # Simple, direct matching against the structured employee data --
        # no LLM needed for this kind of precise lookup.
        for _, row in self.df.iterrows():
            if row["Name"].lower() in question_lower:
                return (f"{row['Name']} works in the {row['Department']} department "
                        f"with {row['Years of Service']} years of service.")

        for _, row in self.df.iterrows():
            if row["Department"].lower() in question_lower:
                matches = self.df[self.df["Department"] == row["Department"]]["Name"].tolist()
                return f"Employees in {row['Department']}: {', '.join(matches)}."

        return "I couldn't find a matching employee or department for that question."