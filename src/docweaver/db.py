from docweaver.config import COLLECTION_NAME, CATALOG_COLLECTION_NAME
from weaviate.classes.config import Property, DataType, Configure
import weaviate
from weaviate import WeaviateClient
from weaviate.util import generate_uuid5
import os


def connect() -> WeaviateClient:
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=os.getenv("WEAVIATE_API_KEY"),
    )


def delete_collection():
    user_input = f"You are about to delete {COLLECTION_NAME} on {os.getenv('WEAVIATE_URL')}! Are you sure? (Y to continue): "
    if user_input == "y":
        print("Deleting the collection!")
        with connect() as client:
            client.collections.delete(COLLECTION_NAME)
    else:
        print("Not deleting the collection")
        return False


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
                    source_properties=["path", "chunk"],
                ),
                Configure.Vectors.text2vec_weaviate(
                    name="path",
                    source_properties=["path"],
                ),
            ],
            generative_config=Configure.Generative.anthropic(
                model="claude-3-5-haiku-latest",
            ),
        )


def add_chunks(src_chunks: list[dict]):
    with connect() as client:
        chunks = client.collections.use(COLLECTION_NAME)
        with chunks.batch.fixed_size(batch_size=100) as batch:
            for i, src_chunk in enumerate(src_chunks):
                batch.add_object(
                    properties={
                        "path": src_chunk["path"],
                        "chunk": src_chunk["chunk"],
                        "chunk_no": i + 1,
                    },
                    uuid=generate_uuid5(src_chunk["path"] + str(i + 1)),
                )


def search_chunks(client: WeaviateClient, queries: list[str]) -> list[dict[str, str]]:
    with client:
        chunks = client.collections.use(COLLECTION_NAME)
        chunk_objs = list()
        for query in queries:
            for target_vector, limit in [("path", 5), ("chunk", 20)]:
                response = chunks.query.near_text(
                    query=query, target_vector=target_vector, limit=limit
                )
                chunk_objs.extend([o.properties for o in response.objects])

        return list(
            {(chunk["path"], chunk["chunk_no"]): chunk for chunk in chunk_objs}.values()
        )


def create_catalog_collection():
    """Create Weaviate collection for document catalog (metadata only)."""
    with connect() as client:
        client.collections.create(
            CATALOG_COLLECTION_NAME,
            properties=[
                Property(name="path", data_type=DataType.TEXT),
                Property(name="title", data_type=DataType.TEXT),
                Property(name="topics", data_type=DataType.TEXT_ARRAY),
                Property(name="doctype", data_type=DataType.TEXT),
                Property(name="summary", data_type=DataType.TEXT),
                Property(name="hash", data_type=DataType.TEXT),
            ],
            vectorizer_config=Configure.Vectorizer.text2vec_weaviate(
                vectorize_collection_name=False
            ),
        )


def add_catalog_entries(entries: list[dict]):
    """Add or update document catalog entries in Weaviate."""
    with connect() as client:
        catalog = client.collections.use(CATALOG_COLLECTION_NAME)
        with catalog.batch.fixed_size(batch_size=50) as batch:
            for entry in entries:
                batch.add_object(
                    properties={
                        "path": entry["path"],
                        "title": entry.get("title"),
                        "topics": entry.get("topics", []),
                        "doctype": entry.get("doctype"),
                        "summary": entry.get("summary"),
                        "hash": entry.get("hash"),
                    },
                    uuid=generate_uuid5(entry["path"]),
                )


def search_catalog(client: WeaviateClient, query: str, limit: int = 10) -> list[dict]:
    """Search document catalog by semantic similarity."""
    with client:
        catalog = client.collections.use(CATALOG_COLLECTION_NAME)
        response = catalog.query.near_text(query=query, limit=limit)
        return [o.properties for o in response.objects]
