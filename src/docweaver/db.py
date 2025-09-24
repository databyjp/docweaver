from docweaver.config import COLLECTION_NAME
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
            response = chunks.query.hybrid(
                query=query, target_vector=["chunk"], limit=20, alpha=0.5
            )
            chunk_objs.extend([o.properties for o in response.objects])

        return list({
            (chunk["path"], chunk["chunk_no"]): chunk
            for chunk in chunk_objs
        }.values())
