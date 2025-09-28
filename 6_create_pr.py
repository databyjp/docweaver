from docweaver.pipeline import create_pr
from rich.console import Console
from dotenv import load_dotenv

load_dotenv()


def main():
    console = Console()

    try:
        console.print("📝 Applying diffs and creating PR...")
        result = create_pr()

        if result["success"]:
            console.print(f"✅ {result['message']}")
            console.print(f"Branch: {result['branch_name']}")
        else:
            console.print(f"ℹ️ {result['message']}")

    except Exception as e:
        console.print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
