# DocWeaver

An AI-powered documentation automation tool updates technical documentation based on instructions.

## How It Works

A multi-agent AI system automatically updates documentation. 

### Workflow

#### 1. **Smart Document Discovery** (`doc_search_agent`)
- Uses vector search (Weaviate) to find all relevant documentation files
- Searches both document content chunks and catalog metadata (topics, summaries)
- Intentionally "overfetches" to ensure no relevant documents are missed

#### 2. **Coordinated Edit Planning** (`doc_instructor_agent`)
- Reviews all found documents and plans necessary edits
- Creates specific, line-by-line instructions for each file
- Handles both markdown documentation and referenced code examples
- Preserves existing quality content—adds new information rather than rewriting

#### 3. **Precise Document Editing** (`doc_writer_agent`)
- Executes edits with line-number precision
- Classifies each edit (add new, update outdated, enhance, delete)
- Maintains documentation style and structure
- Edits code examples in source files, not in markdown

#### 4. **Automated PR Creation**
- Applies all changes to a new git branch
- Generates comprehensive PR description
- Creates pull request ready for review

#### Key Capabilities
- **Vector-powered search**: Finds semantically relevant docs across large codebases
- **Context-aware editing**: Understands document structure, cross-references, and code examples
- **Conservative by design**: Adds information rather than replacing expert-written content
- **Evidence-based changes**: Only modifies existing text when there's specific justification
- **Handles complexity**: Coordinates multi-file edits and code-documentation consistency

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
