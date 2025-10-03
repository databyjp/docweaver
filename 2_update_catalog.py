from docweaver.pipeline import update_catalog
import asyncio
from docweaver.helpers import setup_logging


async def main():
    setup_logging(__file__)

    # Update catalog for all changed documents
    # Paths are configured in src/docweaver/config.py
    result = await update_catalog(
        limit=None,  # Set to a number like 5 for testing
    )

    print(f"Catalog update complete.")
    print(f"Total documents in catalog: {result['total_docs']}")
    print(f"Documents updated: {result['updated_docs']}")
    print(f"Documents removed: {result['removed_docs']}")
    print(f"Catalog saved to: {result['catalog_path']}")


if __name__ == "__main__":
    asyncio.run(main())
