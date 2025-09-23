TECH_DESCRIPTION_COLLECTION_ALIASES = """
Collection aliases allow you to create alternative names for your collections. This is useful for changing collection definitions without downtime, A/B testing, or providing more convenient names for collections. An alias acts as a reference to a collection - when you query using an alias name, Weaviate automatically routes the request to the target collection.

Aliases provide indirection (an intermediate layer) between your application and collections, enabling operational flexibility without downtime.

**If** you deploy schema changes → **Use blue-green deployment with aliases**

```json
Products (alias) → ProductsV1 (collection)
(deploy v2) → switch Products (alias) → ProductsV2 (collection)

```

**If** you need version management → **Use aliases for rollback capability**

- Deploy to new collection, switch alias, keep old version for rollback
"""

TECH_DESCRIPTION_RESHARDING = """
A multi-node Weaviate cluster can now be re-sharded to redistribute data across nodes for improved performance and scalability.

This feature allows administrators to dynamically adjust the number of shards in an existing cluster without downtime. The resharding process works by creating new shard mappings, migrating vector embeddings and metadata in batches, and updating the distributed hash ring to reflect the new topology.

Key capabilities include:
- Automatic load balancing during migration to prevent node overload
- Configurable batch sizes and migration speed throttling
- Real-time consistency checks to ensure data integrity
- Rollback support in case of migration failures
- Monitoring endpoints to track resharding progress

The resharding operation is triggered via the `/v1/cluster/resharding` API endpoint with parameters for target shard count, migration speed, and validation settings. During resharding, read operations continue normally while writes are temporarily queued and replayed after shard migration completes.

Typical use cases include scaling up clusters under heavy load, rebalancing after node additions/removals, and optimizing shard distribution for query performance patterns.
"""
