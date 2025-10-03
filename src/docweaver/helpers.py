import json
import importlib
from pathlib import Path

import logging

from .models import Task

DOCUMENTATION_META_INFO = """
The Weaviate documentation generally follows the Diataxis framework.

Accordingly, each document aims to be primarily one of
[concepts, reference, how-to, or tutorial] formats; although, this isn't always possible.

When searching, reviewing, or editing the documentation file, keep this in mind.
Each document should stick to one of these purposes closely if possible.

Generally, you can tell from the document path, and the first few lines what type of document it is.

It is important to follow this framework to ensure clarity and ease of use for our readers.

You can replicate some information across multiple documents;
however, it is preferable to separate the information into distinct documents to achieve separation of concerns.
"""

NEW_CODE_EXAMPLE_MARKER = (
    "# [!NOTE] This code block is a placeholder and is not yet implemented."
)


def setup_logging(script_name: str):
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"{Path(script_name).stem}.log"

    # Clear previous handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
        ],
    )
    # Add a handler for INFO level logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)


def load_task(task_name: str) -> Task:
    """Load task from Python file in tasks/ directory."""
    try:
        module = importlib.import_module(f"tasks.{task_name}")
        return module.task
    except (ModuleNotFoundError, AttributeError) as e:
        raise FileNotFoundError(
            f"Task file not found or is invalid: tasks/{task_name}.py"
        ) from e


def list_available_tasks() -> list[str]:
    """List all available task files in the tasks/ directory."""
    tasks_dir = Path("tasks")
    if not tasks_dir.exists():
        return []
    return [f.stem for f in tasks_dir.glob("*.py") if f.stem != "__init__"]
