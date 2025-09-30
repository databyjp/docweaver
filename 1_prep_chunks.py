from docweaver.pipeline import prep_database
from helpers import setup_logging


def main():
    setup_logging(__file__)
    result = prep_database(reset_collection=True)
    print(
        f"Database preparation complete. Processed {result['files_processed']} files."
    )


if __name__ == "__main__":
    main()
