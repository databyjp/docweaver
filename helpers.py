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
