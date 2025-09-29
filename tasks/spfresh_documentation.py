from src.docweaver.models import Task

task = Task(
    objective="Create comprehensive documentation for the new SPFresh vector index type",
    context="""## SPFresh Index Type - Technical Description

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
```""",
    focus="Emphasize practical implementation, configuration examples, and when to choose SPFresh over HNSW for production use cases",
)
