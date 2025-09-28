from docweaver.pipeline import make_changes
import asyncio
from helpers import TECH_DESCRIPTION_RESHARDING, setup_logging


async def main():
    setup_logging(__file__)
    result = await make_changes(TECH_DESCRIPTION_RESHARDING)

    print(f"Document changes complete.")
    print(f"Files changed: {result['files_changed']}")
    print(f"Processing time: {result['total_processing_time']:.2f} seconds")
    print(f"Revised documents saved to: {result['output_path']}")
    print(f"Raw edits saved to: {result['edits_path']}")


if __name__ == "__main__":
    asyncio.run(main())
