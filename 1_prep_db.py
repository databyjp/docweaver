from docweaver.pipeline import prep_database


def main():
    result = prep_database()
    print(
        f"Database preparation complete. Processed {result['files_processed']} files."
    )


if __name__ == "__main__":
    main()
