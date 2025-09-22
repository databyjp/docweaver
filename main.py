import docweaver.db
from docweaver.utils import chunk_text
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def main():
    docweaver.db.delete_collection()
    docweaver.db.create_collection()
    for file in Path("docs/docs/weaviate/concepts/replication-architecture").glob("*.md*"):
        with open(file, "r") as f:
            text = f.read()
            chunks = chunk_text(text)
            chunk_texts = [{"path": file.as_posix(), "chunk": chunk.text} for chunk in chunks]
            docweaver.db.add_chunks(chunk_texts)


if __name__ == "__main__":
    main()
