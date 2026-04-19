import neo4j
import re
import unicodedata
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore


class SafeNeo4jPropertyGraphStore(Neo4jPropertyGraphStore):
    def get_schema(self, refresh: bool = False) -> str:
        # refresh: if True, re-read apoc.meta; we usually avoid refresh on bad legacy types.
        try:
            return super().get_schema(refresh=refresh)
        except neo4j.exceptions.CypherTypeError as exc:
            print(f"[debug] Skip schema refresh due to CypherTypeError: {exc}")
            return str(self.structured_schema)


def create_graph_store(
    neo4j_uri: str, neo4j_username: str, neo4j_password: str
) -> SafeNeo4jPropertyGraphStore:
    # neo4j_uri: Bolt/Neo4j URI; neo4j_username/password: DB credentials.
    print(f"[db] Connecting Neo4j at {neo4j_uri}...")
    return SafeNeo4jPropertyGraphStore(
        username=neo4j_username,
        password=neo4j_password,
        url=neo4j_uri,
        refresh_schema=False,  # skip apoc.meta on connect (faster; fewer type errors)
        enhanced_schema=False,  # lighter schema for prompts
    )


def sanitize_entity_names(graph_store: Neo4jPropertyGraphStore) -> int:
    # graph_store: connected store used to run Cypher fixes in-place.
    rows = graph_store.structured_query(
        """
        MATCH (n:`__Entity__`)
        WHERE n.name IS NOT NULL OR n.id IS NOT NULL
        RETURN id(n) AS node_id, n.name AS raw_name, n.id AS raw_id
        """
    )
    fixed_count = 0
    for row in rows:
        node_id = int(row["node_id"])
        raw_name = row.get("raw_name")
        if isinstance(raw_name, list) and len(raw_name) > 0:
            canonical_name = str(raw_name[0])
            graph_store.structured_query(
                """
                MATCH (n:`__Entity__`) WHERE id(n) = $node_id
                SET n.name = $canonical_name
                """,
                {"node_id": node_id, "canonical_name": canonical_name},
            )
            fixed_count += 1

        raw_entity_id = row.get("raw_id")
        if isinstance(raw_entity_id, list) and len(raw_entity_id) > 0:
            canonical_id = str(raw_entity_id[0])
            graph_store.structured_query(
                """
                MATCH (n:`__Entity__`) WHERE id(n) = $node_id
                SET n.id = $canonical_id
                """,
                {"node_id": node_id, "canonical_id": canonical_id},
            )
            fixed_count += 1

    print(f"[db] Converted LIST name/id fields to string: {fixed_count}")
    return fixed_count


def _normalize_lookup_id(raw_value: str) -> str:
    lowered = raw_value.lower().strip()
    folded = unicodedata.normalize("NFKD", lowered)
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    folded = re.sub(r"[^a-z0-9 ]+", " ", folded)
    tokens = [token for token in folded.split() if token not in {"ong", "ba"}]
    return " ".join(tokens)


def upsert_entity_lookup_ids(graph_store: Neo4jPropertyGraphStore) -> int:
    # Build a stable lookup key for fuzzy/exact matching across honorific/diacritic variants.
    rows = graph_store.structured_query(
        """
        MATCH (n:`__Entity__`)
        WHERE n.id IS NOT NULL OR n.name IS NOT NULL
        RETURN id(n) AS node_id, n.id AS raw_id, n.name AS raw_name
        """
    )
    updated_count = 0
    for row in rows:
        raw_id = row.get("raw_id")
        raw_name = row.get("raw_name")
        fallback_value = str(raw_name) if raw_name is not None else ""
        source_value = str(raw_id) if raw_id is not None else fallback_value
        lookup_id = _normalize_lookup_id(source_value)
        if lookup_id == "":
            continue
        graph_store.structured_query(
            """
            MATCH (n:`__Entity__`) WHERE id(n) = $node_id
            SET n.lookup_id = $lookup_id
            """,
            {"node_id": int(row["node_id"]), "lookup_id": lookup_id},
        )
        updated_count += 1
    print(f"[db] Upserted lookup_id for entities: {updated_count}")
    return updated_count
