from pydantic_ai import Agent, RunContext
from pydantic import BaseModel
from docweaver.db import search_chunks


docs_search_agent = Agent(
    model="anthropic:claude-3-5-haiku-latest",
    output_type=list[str],
    system_prompt="""
    You are a research assistant.
    You are given a user query, and you are to search the available Weaviate documentation
    and return the documents that sound most relevant to the query.
    """
)


class Query(BaseModel):
    query: str


@docs_search_agent.tool
def search_docs(ctx: RunContext[Query]) -> list[str]:
    return search_chunks(ctx.deps.query)
