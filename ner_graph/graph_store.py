import neo4j
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore


class SafeNeo4jPropertyGraphStore(Neo4jPropertyGraphStore):
    def get_schema(self, refresh: bool = False) -> str:
        try:
            return super().get_schema(refresh=refresh)
        except neo4j.exceptions.CypherTypeError as exc:
            print(f"[debug] Skip schema refresh due to CypherTypeError: {exc}")
            return str(self.structured_schema)


def create_graph_store(
    neo4j_uri: str, neo4j_username: str, neo4j_password: str
) -> SafeNeo4jPropertyGraphStore:
    print(f"[db] Connecting Neo4j at {neo4j_uri}...")
    return SafeNeo4jPropertyGraphStore(
        username=neo4j_username,
        password=neo4j_password,
        url=neo4j_uri,
        refresh_schema=False,
        enhanced_schema=False,
    )


def sanitize_entity_names(graph_store: Neo4jPropertyGraphStore) -> int:
    rows = graph_store.structured_query(
        """
        MATCH (n:`__Entity__`)
        WHERE n.name IS NOT NULL
        RETURN id(n) AS node_id, n.name AS raw_name
        """
    )
    fixed_count = 0
    for row in rows:
        raw_name = row.get("raw_name")
        if isinstance(raw_name, list) and len(raw_name) > 0:
            canonical_name = str(raw_name[0])
            graph_store.structured_query(
                """
                MATCH (n:`__Entity__`) WHERE id(n) = $node_id
                SET n.name = $canonical_name
                """,
                {"node_id": int(row["node_id"]), "canonical_name": canonical_name},
            )
            fixed_count += 1
    print(f"[db] Converted LIST names to string: {fixed_count}")
    return fixed_count
