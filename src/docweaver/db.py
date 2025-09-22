from weaviate.classes.config import Property, DataType, Configure
import weaviate
from weaviate import WeaviateClient
import os
from docweaver.config import COLLECTION_NAME


def connect() -> WeaviateClient:
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=os.getenv("WEAVIATE_API_KEY"),
    )


def create_collection():
    with connect() as client:
        client.collections.create(
            COLLECTION_NAME,
            properties=[
                Property(name="path", data_type=DataType.TEXT),
                Property(name="chunk", data_type=DataType.TEXT),
                Property(name="chunk_no", data_type=DataType.INT),
            ],
            vector_config=[
                Configure.Vectors.text2vec_weaviate(
                    name="chunk",
                    source_properties=["title", "chunk"],
                ),
                Configure.Vectors.text2vec_weaviate(
                    name="topics",
                    source_properties=["topics"],
                ),
            ],
            generative_config=Configure.Generative.anthropic(
                model="claude-3-5-haiku-latest",
            ),
        )
        print(f"Collection {COLLECTION_NAME} created successfully")


def search_chunks(query: str) -> list[str]:
    with connect() as client:
        chunks = client.collections.use(COLLECTION_NAME)
        response = chunks.query.hybrid(
            query=query,
            limit=20,
            alpha=0.5
        )
        docpaths = [o.properties["path"] for o in response.objects]
        return docpaths
