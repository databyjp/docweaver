from docweaver.agents import docs_search_agent, Query
import asyncio


async def main():
    query = "What is the replication architecture of Weaviate?"
    docs = await docs_search_agent.run(
        "Find the documentation that is most relevant to the query",
        deps=Query(query=query)
    )
    print(docs)


if __name__ == "__main__":
    asyncio.run(main())
