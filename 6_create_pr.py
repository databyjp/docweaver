import os
import json
from pathlib import Path
from rich.console import Console
from github import Github, Auth
import git
from dotenv import load_dotenv

load_dotenv()


def apply_diffs():
    """Apply the generated diffs to the actual files using GitPython."""
    diff_log_path = Path("logs/diffs.log")

    if not diff_log_path.exists():
        raise FileNotFoundError("No diffs.log found. Run 5_create_diffs.py first.")
    else:
        with open(diff_log_path, 'r') as f:
            diff_content = f.read()
    console = Console()

    # Check if there are any diffs to apply
    if diff_log_path.stat().st_size == 0:
        console.print("✅ No diffs to apply, skipping.")
        return

    # Work in the docs directory where the actual Weaviate docs are
    docs_path = Path("docs")
    if not docs_path.exists():
        raise FileNotFoundError("docs/ directory not found")

    repo = git.Repo(docs_path)

    # Apply diffs using GitPython by passing the file path directly
    try:
        # Replace 'docs/' prefix in file paths since we're now working from docs directory
        # Handle Git's a/ and b/ prefixes properly
        modified_diff = diff_content.replace('--- a/docs/', '--- a/').replace('+++ b/docs/', '+++ b/')

        # Write modified diff to a temporary file in docs directory
        temp_diff_path = docs_path / "temp_diffs.patch"
        with open(temp_diff_path, 'w') as f:
            f.write(modified_diff)

        # Apply the modified diff (use just the filename since we're in docs directory)
        repo.git.apply("--verbose", "temp_diffs.patch")

        # Clean up temporary file
        temp_diff_path.unlink()

        console.print("✅ Diffs applied successfully")

    except git.exc.GitCommandError as e:
        raise RuntimeError(f"Failed to apply diffs: {e}")
    except Exception as e:
        # Clean up temp file if it exists
        temp_diff_path = docs_path / "temp_diffs.patch"
        if temp_diff_path.exists():
            temp_diff_path.unlink()
        raise RuntimeError(f"Failed to process diffs: {e}")


def create_pr(title: str, body: str, branch_name: str = "docweaver-updates"):
    """Create a PR with the applied changes using PyGithub."""
    console = Console()

    # Get GitHub token from environment
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        raise ValueError("GITHUB_TOKEN environment variable not set")

    # Work in the docs directory where the actual Weaviate docs are
    docs_path = Path("docs")
    if not docs_path.exists():
        raise FileNotFoundError("docs/ directory not found")

    # Initialize Git and GitHub clients for the docs repo
    repo = git.Repo(docs_path)
    auth = Auth.Token(github_token)
    g = Github(auth=auth)

    # Get remote URLs to determine fork and upstream
    origin_url = repo.remotes.origin.url
    # Handle both SSH and HTTPS URLs
    if origin_url.startswith("git@"):
        # SSH format: git@github.com:owner/repo.git
        repo_part = origin_url.split(":")[-1]
        fork_owner = repo_part.split("/")[0]
        repo_name = repo_part.split("/")[1].replace(".git", "")
    else:
        # HTTPS format: https://github.com/owner/repo
        url_parts = origin_url.rstrip("/").split("/")
        fork_owner = url_parts[-2]
        repo_name = url_parts[-1].replace(".git", "")

    # Ensure we're on main branch before creating new branch
    try:
        repo.heads.main.checkout()
    except:
        # Try master if main doesn't exist
        repo.heads.master.checkout()

    # Create and checkout new branch
    if branch_name in repo.heads:
        console.print(f"Branch '{branch_name}' already exists. Recreating it.")
        repo.delete_head(branch_name, force=True)

    new_branch = repo.create_head(branch_name)
    new_branch.checkout()

    # Stage and commit changes
    repo.git.add(".")

    # Check if there are any changes to commit
    if not repo.is_dirty() and not repo.untracked_files:
        console.print("⚠️  No changes to commit")
        return None

    repo.index.commit(title)

    # Push to fork
    repo.remotes.origin.push(new_branch)

    # Create PR using PyGithub - this creates a PR within the same repo
    # since databyjp/docs is your fork of the weaviate docs
    github_repo = g.get_repo(f"{fork_owner}/{repo_name}")

    # Create PR against the main branch of your fork
    # If you want to create a PR to the upstream weaviate repo,
    # you'd need to specify the upstream repo here instead
    pr = github_repo.create_pull(
        title=title,
        body=body,
        head=branch_name,
        base="main"
    )

    console.print(f"✅ PR created: {pr.html_url}")
    return pr


def main():
    console = Console()

    try:
        console.print("📝 Applying diffs...")
        apply_diffs()

        console.print("🚀 Creating PR...")
        create_pr(
            title="📚 Automated documentation updates via DocWeaver",
            body="This PR contains automated documentation improvements generated by DocWeaver.\n\n"
            "Changes include content updates, formatting improvements, and clarity enhancements.",
        )

    except Exception as e:
        console.print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
