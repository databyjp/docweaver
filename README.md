# DocWeaver

Agentic documentation editing tool that uses AI to search, coordinate, and apply changes to documentation.

## Setup

1. Install dependencies:
```bash
pip install -e .
```

2. Create `.env` file:
```
WEAVIATE_URL=....gcp.weaviate.cloud
WEAVIATE_API_KEY=b0llbXFYUEY...
ANTHROPIC_API_KEY=your_key_here
GITHUB_TOKEN=your_github_token
```

3. Clone the `docs` repo locally

4. Prepare the database:
```bash
python 0_prep_db.py
```

## Usage

### Update Catalog (Optional)
Generate metadata for documentation files:
```bash
python 2_update_catalog.py
```

This creates a searchable catalog with document metadata (topics, type, summary).

### Make Changes
1. Create a task file in `tasks/` (see `tasks/resharding_feature.py`)
2. Update `CURRENT_TASK_NAME` in `1_make_changes.py`
3. Run:
```bash
python 1_make_changes.py
```

The pipeline will search documents, generate edit instructions, apply changes, create diffs, and create a PR.

## Output

Results saved to `outputs/`:
- `catalog.json` - Document metadata catalog
- `doc_search_agent.log` - Documents found
- `doc_instructor_agent.log` - Edit instructions
- `doc_writer_agent.log` - Revised documents
- `diffs.log` - Unified diffs