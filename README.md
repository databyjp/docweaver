# DocWeaver

Agentic documentation editing tool that uses AI to search, coordinate, and apply changes to documentation.

## Setup

1. Install dependencies:
```bash
uv sync
```

2. Create `.env` file:
```
WEAVIATE_URL=....gcp.weaviate.cloud
WEAVIATE_API_KEY=b0llbXFYUEY...
ANTHROPIC_API_KEY=your_key_here
GITHUB_TOKEN=your_github_token
```

3. Clone the `docs` repo locally
    - It should end up in `docs/docs`

4. (Only if needed) Prepare the DB:
    - Add chunks to the database
    - Create a catalog of documents with metadata & summary
    ```bash
    python 1_prep_chunks.py
    python 2_update_catalog.py
    ```
    - The catalog will also be saved locally to `catalog.json`

## Usage

### Make Changes
1. Create a task file in `tasks/` (see `tasks/resharding_feature.py`)
2. Add it to `TASKS_TO_RUN` in `3_make_changes.py`
e.g.:
```python
TASKS_TO_RUN = [
    "training_schema_design",
    "training_backup",
    "training_monitoring",
    "training_deployment"
]
```
3. Run:
```bash
python 3_make_changes.py
```

The pipeline will search documents, generate edit instructions, apply changes, create diffs, and create a PR.

## Output

Results saved to `outputs/<task_name>`:
- `doc_search_agent.log` - Documents found
- `doc_instructor_agent.log` - Edit instructions
- `doc_writer_agent_raw_output_<name>.log` - Individual edits
- `doc_writer_agent_edits.log` - Collated edits
- `doc_writer_agent.log` - Revised documents
