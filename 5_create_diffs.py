from docweaver.pipeline import create_diffs


def main():
    result = create_diffs()

    if result["has_changes"]:
        print(f"Created {result['diffs_created']} diffs")
        print(f"Diffs saved to: {result['output_path']}")
    else:
        print("No changes found to create diffs for")


if __name__ == "__main__":
    main()
