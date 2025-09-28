from docweaver.pipeline import coordinate_changes
import asyncio
from helpers import TECH_DESCRIPTION_RESHARDING, setup_logging


async def main():
    setup_logging(__file__)
    result = await coordinate_changes(TECH_DESCRIPTION_RESHARDING)

    print(f"Change coordination complete.")
    print(f"Generated {len(result['instructions'])} editing instructions")
    print(f"Processed {result['documents_processed']} documents")
    print(f"Instructions saved to: {result['output_path']}")
    print(f"Token usage: {result['token_usage']}")


if __name__ == "__main__":
    asyncio.run(main())
