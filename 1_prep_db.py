import docweaver.db
from docweaver.utils import chunk_text
from pathlib import Path


def main():
    docweaver.db.delete_collection()
    docweaver.db.create_collection()
    md_files = Path("docs/docs/weaviate/concepts/replication-architecture").rglob("*.md*")
    for file in md_files:
        with open(file, "r") as f:
            print(f"Importing {file}")
            text = f.read()
            chunks = chunk_text(text)
            chunk_texts = [{"path": file.as_posix(), "chunk": chunk.text} for chunk in chunks]
            docweaver.db.add_chunks(chunk_texts)


if __name__ == "__main__":
    main()
