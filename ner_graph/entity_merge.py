import re
import unicodedata
from itertools import groupby

from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.openai import OpenAI

STOP_WORDS: frozenset[str] = frozenset(
    {
        "tap",
        "doan",
        "cong",
        "ty",
        "viet",
        "nam",
        "ong",
        "ba",
        "anh",
        "chi",
        "group",
        "company",
        "co",
        "ltd",
        "corp",
        "inc",
        "the",
        "a",
        "an",
        "of",
        "and",
        "for",
        "in",
        "at",
        "by",
        "to",
    }
)


def normalize_name(name: str) -> str:
    lowered = name.lower().strip()
    folded = unicodedata.normalize("NFKD", lowered)
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    folded = re.sub(r"[^a-z0-9 ]+", " ", folded)
    return re.sub(r"\s+", " ", folded).strip()


def _meaningful_tokens(name: str) -> set[str]:
    return {token for token in normalize_name(name).split() if token not in STOP_WORDS}


def _candidate_similarity(left_name: str, right_name: str) -> float:
    left_tokens = _meaningful_tokens(left_name)
    right_tokens = _meaningful_tokens(right_name)
    union = left_tokens | right_tokens
    if len(union) == 0:
        return 0.0
    jaccard = len(left_tokens & right_tokens) / len(union)
    smaller = left_tokens if len(left_tokens) <= len(right_tokens) else right_tokens
    larger = right_tokens if len(left_tokens) <= len(right_tokens) else left_tokens
    containment = (len(smaller & larger) / len(smaller)) if len(smaller) > 0 else 0.0
    return max(jaccard, 0.7 * containment)


def _blocking_key(name: str) -> str:
    tokens = sorted(_meaningful_tokens(name))
    if len(tokens) > 0:
        return tokens[0]
    return normalize_name(name)[:3]


def _get_entity_rows(graph_store: Neo4jPropertyGraphStore) -> list[dict[str, str | int]]:
    raw_rows = graph_store.structured_query(
        """
        MATCH (n:`__Entity__`)
        WHERE n.name IS NOT NULL
        RETURN id(n) AS node_id, n.name AS name
        """
    )
    rows: list[dict[str, str | int]] = []
    for row in raw_rows:
        raw_name = row.get("name")
        if isinstance(raw_name, list):
            if len(raw_name) == 0:
                continue
            name_value = str(raw_name[0])
        else:
            name_value = str(raw_name)
        rows.append({"node_id": int(row["node_id"]), "name": name_value})
    return rows


def merge_similar_entities(
    graph_store: Neo4jPropertyGraphStore,
    llm: OpenAI,
    similarity_threshold: float,
    max_llm_checks: int,
) -> int:
    if similarity_threshold <= 0.0 or similarity_threshold > 1.0:
        raise RuntimeError(
            f"similarity_threshold must be in (0, 1], got {similarity_threshold}"
        )

    rows = _get_entity_rows(graph_store)
    print(f"[merge] Found {len(rows)} entity nodes")
    if len(rows) < 2:
        return 0

    name_by_id: dict[int, str] = {
        int(row["node_id"]): str(row["name"]) for row in rows
    }

    exact_groups: dict[str, list[int]] = {}
    for row in rows:
        key = normalize_name(str(row["name"]))
        exact_groups.setdefault(key, []).append(int(row["node_id"]))

    exact_pairs = [
        (ids[0], ids[index])
        for ids in exact_groups.values()
        if len(ids) > 1
        for index in range(1, len(ids))
    ]

    rows_sorted = sorted(rows, key=lambda row: _blocking_key(str(row["name"])))
    candidates: list[tuple[float, int, int, str, str]] = []
    for _, group_iter in groupby(
        rows_sorted, key=lambda row: _blocking_key(str(row["name"]))
    ):
        group = list(group_iter)
        for left_index in range(len(group)):
            for right_index in range(left_index + 1, len(group)):
                left_name = str(group[left_index]["name"])
                right_name = str(group[right_index]["name"])
                if normalize_name(left_name) == normalize_name(right_name):
                    continue
                score = _candidate_similarity(left_name, right_name)
                if score >= similarity_threshold:
                    candidates.append(
                        (
                            score,
                            int(group[left_index]["node_id"]),
                            int(group[right_index]["node_id"]),
                            left_name,
                            right_name,
                        )
                    )
    candidates.sort(key=lambda item: -item[0])

    llm_approved: list[tuple[int, int, str]] = []
    llm_checks = 0
    for score, left_id, right_id, left_name, right_name in candidates:
        if llm_checks >= max_llm_checks:
            break
        llm_checks += 1
        print(
            f"[merge] LLM check {llm_checks}/{max_llm_checks}: "
            f"{left_name} vs {right_name} ({score:.3f})"
        )
        verdict = llm.complete(
            "Are these two entity names the same real-world entity? "
            "Answer only YES or NO.\n"
            f"Name A: {left_name}\n"
            f"Name B: {right_name}"
        ).text.strip().upper()
        if verdict.startswith("YES"):
            llm_approved.append((left_id, right_id, left_name))

    merge_instructions: list[tuple[int, int, str]] = []
    for left_id, right_id in exact_pairs:
        canonical_name = name_by_id.get(left_id, str(left_id))
        merge_instructions.append((left_id, right_id, canonical_name))
    merge_instructions.extend(llm_approved)

    deduped: list[tuple[int, int, str]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for left_id, right_id, canonical_name in merge_instructions:
        pair = (left_id, right_id)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        deduped.append((left_id, right_id, canonical_name))

    merged_count = 0
    for left_id, right_id, canonical_name in deduped:
        print(f"[merge] Merge node {right_id} -> {left_id} ({canonical_name})")
        graph_store.structured_query(
            """
            MATCH (left:`__Entity__`) WHERE id(left) = $left_id
            MATCH (right:`__Entity__`) WHERE id(right) = $right_id
            WITH left, right
            CALL apoc.refactor.mergeNodes(
                [left, right],
                {properties: "discard", mergeRels: true}
            ) YIELD node
            SET node.name = $canonical_name
            RETURN id(node) AS merged_id
            """,
            {
                "left_id": left_id,
                "right_id": right_id,
                "canonical_name": canonical_name,
            },
        )
        merged_count += 1

    print(f"[merge] Completed, merged_count={merged_count}")
    return merged_count
