from docweaver.db import delete_collection, connect
from docweaver.config import COLLECTION_NAME, CATALOG_COLLECTION_NAME


def main():
    with connect() as client:
        print(f"Connected to Weaviate at {client._connection.url}")
        collections_to_delete = [
            COLLECTION_NAME,
            CATALOG_COLLECTION_NAME
        ]
        for cname in collections_to_delete:
            try:
                client.collections.delete(cname)
                print(f"Deleted collection: {cname}")
            except Exception as e:
                print(f"Failed to delete collection {cname}: {e}")


if __name__ == "__main__":
    main()
