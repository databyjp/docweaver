# DocWeaver

Agentic documentation editing tool that uses AI to search, coordinate, and apply changes to documentation.

## Architecture

DocWeaver uses the [weaviate-docs-mcp](https://github.com/yourusername/weaviate-docs-mcp) server for document search functionality. The MCP (Model Context Protocol) server provides semantic search over documentation using Weaviate.

## Setup

### 1. Set up weaviate-docs-mcp server

First, set up the MCP server that provides document search:

```bash
# Clone and set up weaviate-docs-mcp
cd ~/code
git clone https://github.com/yourusername/weaviate-docs-mcp.git
cd weaviate-docs-mcp

# Install dependencies
uv sync

# Create .env file
cp .env.example .env
# Edit .env with your Weaviate and Cohere credentials

# Clone the docs repository
git clone https://github.com/weaviate/docs.git

# Update the catalog (generates metadata and populates Weaviate)
uv run python update_catalog.py
```

See the [weaviate-docs-mcp README](https://github.com/yourusername/weaviate-docs-mcp/blob/main/README.md) for detailed setup instructions.

### 2. Set up DocWeaver

```bash
# Install dependencies
uv sync

# Create .env file
cp .env.example .env
```

Edit `.env` with:
```
ANTHROPIC_API_KEY=your_anthropic_key_here
GITHUB_TOKEN=your_github_token
```

### 3. Clone the docs repo locally

The `docs` repository should be cloned to `./docs`:
```bash
git clone https://github.com/weaviate/docs.git
```

The documentation files should be in `docs/docs/`.

## Usage

### Making Documentation Changes

1. **Create a task file** in `tasks/` directory:
   - See `tasks/resharding_feature.py` for an example
   - Define the feature description and any specific requirements

2. **Add task to the run list** in `3_make_changes.py`:
   ```python
   TASKS_TO_RUN = [
       "training_schema_design",
       "training_backup",
       "training_monitoring",
       "training_deployment"
   ]
   ```

3. **Run the pipeline**:
   ```bash
   uv run python 3_make_changes.py
   ```

   The pipeline will:
   - Search documents via the MCP server
   - Generate coordinated edit instructions
   - Apply changes to documentation
   - Create a branch and pull request

### Cleaning Task Outputs

To remove cached results and start fresh for a specific task:
```bash
uv run python 3_make_changes.py --clean
```

## How It Works

1. **Document Search** (`search_documents`): Uses the weaviate-docs-mcp server to find relevant documents via semantic search
2. **Change Coordination** (`coordinate_changes`): Analyzes documents and generates structured editing instructions
3. **Apply Changes** (`make_changes`): Executes edits on documentation files
4. **Create PR** (`create_pr`): Commits changes and creates a draft pull request

## Output

Results saved to `outputs/<task_name>`:
- `doc_search_agent.log` - Documents found
- `doc_instructor_agent.log` - Edit instructions
- `doc_writer_agent_raw_output_<name>.log` - Individual edits
- `doc_writer_agent_edits.log` - Collated edits
- `doc_writer_agent.log` - Revised documents
