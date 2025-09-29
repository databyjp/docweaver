from pydantic import BaseModel


class Task(BaseModel):
    """A task for the document weaver to perform."""

    objective: str
    context: str
    focus: str

    def get_description(self) -> str:
        """Returns formatted task description for agents."""
        return f"""
Objective: {self.objective}
Context:
{self.context}
Focus: {self.focus}
"""
