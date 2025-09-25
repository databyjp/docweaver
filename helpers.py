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
    import logging
    from pathlib import Path

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


# Sample task configurations
SAMPLE_RESHARDING_TASK = {
    "objective": "Document the new resharding feature for multi-node Weaviate clusters",
    "context": """
A multi-node Weaviate cluster can now be re-sharded to redistribute data across nodes for improved performance and scalability.

This feature allows administrators to dynamically adjust the number of shards in an existing cluster without downtime. The resharding process works by creating new shard mappings, migrating vector embeddings and metadata in batches, and updating the distributed hash ring to reflect the new topology.

Key capabilities include:
- Automatic load balancing during migration to prevent node overload
- Configurable batch sizes and migration speed throttling
- Real-time consistency checks to ensure data integrity
- Rollback support in case of migration failures
- Monitoring endpoints to track resharding progress

The resharding operation is triggered via the `/v1/cluster/resharding` API endpoint with parameters for target shard count, migration speed, and validation settings.

Each Weaviate client library (Python, JS/TS, Go, Java) will get its own native functions to do this. The exact syntax is not yet known.

During resharding, read operations continue normally while writes are temporarily queued and replayed after shard migration completes.

Typical use cases include scaling up clusters under heavy load, rebalancing after node additions/removals, and optimizing shard distribution for query performance patterns.
""",
    "focus": "Emphasize operational aspects, migration safety, and provide clear usage examples for administrators"
}

SAMPLE_SPFRESH_TASK = {
    "objective": "Create comprehensive documentation for the new SPFresh vector index type",
    "context": """
## SPFresh Index Type - Technical Description

**Overview:**
SPFresh is a hybrid disk-based vector index that supports incremental in-place updates for billion-scale datasets. It maintains partition centroids in an in-memory HNSW index for fast candidate selection, while storing vector partitions on disk. Unlike traditional approaches that require periodic global rebuilds, SPFresh uses LIRE (Lightweight Incremental REbalancing) to maintain index quality through local partition splits and vector reassignments.

**Architecture:**
- **In-memory HNSW index**: Stores partition centroids for fast nearest-partition lookup
- **Disk-based partitions**: Store the actual vectors, organized by proximity
- **Background rebalancing**: Automatically maintains partition quality as data changes

**How Search Works:**
1. Query vector searches the in-memory HNSW index to find nearest partition centroids
2. Identified partitions are loaded from disk in parallel
3. Full distance calculations performed on candidate vectors
4. Top-K results returned

**How Updates Work:**
1. Insert: Vector assigned to nearest partition (via HNSW centroid search), appended to disk
2. When partition exceeds `maxPostingSize`, it splits into two partitions
3. New centroids are computed and updated in the HNSW index
4. Nearby partitions checked for vectors needing reassignment (NPA compliance)
5. Only boundary vectors are reassigned, minimizing overhead

**Key Features:**
- **Fast partition selection**: HNSW index on centroids provides O(log n) partition lookup
- **In-place updates**: Append vectors directly without requiring global index rebuilds
- **Incremental rebalancing**: Maintains index quality through local splits and reassignments
- **Low resource overhead**: Requires ~1% of DRAM and <10% of CPU cores compared to global rebuild
- **Stable performance**: Consistent search latency and accuracy during continuous updates

**Memory Requirements:**
- HNSW centroid index: ~40 bytes per partition (for 1B vectors with 100K partitions = ~4GB)
- Block mapping metadata: ~40 bytes per partition (~4GB for billion-scale)
- Version tracking: 1 byte per vector for reassignment tracking (~1GB for billion-scale)
- Total: ~10-20GB for billion-scale datasets

**When to Use:**
- Collections with frequent updates (>1% daily update rate)
- Billion-scale datasets where global rebuilds are too expensive
- Use cases requiring stable tail latency during updates
- Applications needing fresh data without performance degradation

**Configuration Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `maxPostingSize` | integer | 10000 | Maximum vectors per partition before split triggers |
| `minPostingSize` | integer | 1000 | Minimum partition size before merge considered |
| `reassignRange` | integer | 64 | Number of nearby partitions to check for reassignment |
| `vectorCacheMaxObjects` | integer | 1e12 | Maximum objects in memory cache |
| `ef` | integer | -1 | Search list size for HNSW centroid index (uses dynamic ef by default) |
| `efConstruction` | integer | 128 | Build parameter for HNSW centroid index |

**Limitations:**
- Centroid HNSW index resides in memory (though small compared to full HNSW)
- Requires sufficient disk IOPS for optimal performance
- Initial index building still requires balanced clustering

---

## Draft Client API

### Python Client v4

```python
from weaviate.classes.config import Configure, VectorDistances

client.collections.create(
    "Articles",
    vector_config=Configure.Vectors.text2vec_openai(
        name="default",
        # Configure SPFresh index
        vector_index_config=Configure.VectorIndex.spfresh(
            distance_metric=VectorDistances.COSINE,
            # Partition management
            max_posting_size=10000,
            min_posting_size=1000,
            reassign_range=64,
            # Centroid HNSW configuration
            ef=-1,  # Dynamic ef for centroid search
            ef_construction=128,  # Build quality for centroid index
            vector_cache_max_objects=1000000000000
        )
    )
)
```

### Migration from HNSW with Updates

```python
# Before: HNSW requiring periodic rebuilds
client.collections.create(
    "Articles",
    vector_config=Configure.Vectors.text2vec_openai(
        vector_index_config=Configure.VectorIndex.hnsw()
    )
)

# After: SPFresh for continuous updates
client.collections.create(
    "Articles",
    vector_config=Configure.Vectors.text2vec_openai(
        vector_index_config=Configure.VectorIndex.spfresh(
            max_posting_size=10000,
            reassign_range=64
        )
    )
)
```
""",
    "focus": "Emphasize practical implementation, configuration examples, and when to choose SPFresh over HNSW for production use cases"
}

# Current task configuration - change this to switch between tasks
CURRENT_TASK = SAMPLE_SPFRESH_TASK

def get_current_task_description() -> str:
    """Returns formatted task description for agents."""
    if isinstance(CURRENT_TASK, dict):
        return f"""
Objective: {CURRENT_TASK['objective']}

Context:
{CURRENT_TASK['context']}

Focus: {CURRENT_TASK['focus']}
"""
    else:
        # Fallback for simple string tasks
        return CURRENT_TASK
