from ast import mod
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, ConfigDict
from docweaver.db import search_chunks
from weaviate import WeaviateClient


class DocSearchDeps(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    client: WeaviateClient


class DocSearchReturn(BaseModel):
    path: str
    reason: str


docs_search_agent = Agent(
    model="anthropic:claude-3-5-haiku-latest",
    output_type=list[DocSearchReturn],
    system_prompt="""
    You are a research assistant.

    You are given a user query, and you are to search the available Weaviate documentation.
    Perform multiple searches with different query strings, if necessary.

    Review the returned data, and return the documents that sound most relevant to the query.

    Return the file path, and the reason why the file is relevant.
    """
)


@docs_search_agent.tool
def search_docs(ctx: RunContext[DocSearchDeps], query=str) -> list[dict[str, str]]:
    return search_chunks(ctx.deps.client, query)
