from src.docweaver.models import Task

task = Task(
    objective="Document the new resharding feature for multi-node Weaviate clusters",
    context="""A multi-node Weaviate cluster can now be re-sharded to redistribute data across nodes for improved performance and scalability.

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

Typical use cases include scaling up clusters under heavy load, rebalancing after node additions/removals, and optimizing shard distribution for query performance patterns.""",
    focus="Emphasize operational aspects, migration safety, and provide clear usage examples for administrators",
)
