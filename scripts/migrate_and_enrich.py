"""
scripts/migrate_and_enrich.py
Script di chuyển nhãn cũ trong Neo4j sang 9 nhãn chuẩn UPPERCASE,
và tự động làm giàu SMILES từ NIH Cactus / PubChem cho tất cả các node DRUG.
"""

import os
import sys
from dotenv import load_dotenv
from neo4j import GraphDatabase

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
load_dotenv(os.path.join(project_root, ".env"), override=True)

from ner_graph.schema_definition import fetch_drug_smiles

def run_migration():
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    pw = os.getenv("NEO4J_PASSWORD")
    db = os.getenv("NEO4J_DATABASE")

    print(f"[migration] Connecting to Neo4j at {uri} (db: {db})...")
    driver = GraphDatabase.driver(uri, auth=(user, pw))

    label_mappings = [
        ("drug", "DRUG"),
        ("disease", "DISEASE"),
        ("symptom", "SIDE_EFFECT"),
        ("enzyme", "ENZYME"),
        ("protein", "TARGET"),
        ("mechanism", "TARGET"),
        ("condition", "CONDITION"),
    ]

    rel_mappings = [
        ("interacts_with", "INTERACTS_WITH"),
        ("treats", "TREATS"),
        ("causes_side_effect", "CAUSES_SIDE_EFFECT"),
        ("metabolized_by", "METABOLIZED_BY"),
    ]

    with driver.session(database=db) as session:
        print("[migration] Migrating node labels to standard schema...")
        for old_l, new_l in label_mappings:
            q = f"MATCH (n:`{old_l}`) SET n:`{new_l}` REMOVE n:`{old_l}` RETURN count(n) as cnt"
            res = session.run(q).single()
            if res and res["cnt"] > 0:
                print(f"  Migrated {res['cnt']} nodes from :{old_l} -> :{new_l}")

        print("\n[migration] Migrating relationships to standard schema...")
        for old_r, new_r in rel_mappings:
            q = f"""
            MATCH (a)-[r:`{old_r}`]->(b)
            MERGE (a)-[r2:`{new_r}`]->(b)
            SET r2 += properties(r)
            DELETE r
            RETURN count(r2) as cnt
            """
            res = session.run(q).single()
            if res and res["cnt"] > 0:
                print(f"  Migrated {res['cnt']} rels from [:{old_r}] -> [:{new_r}]")

        print("\n[migration] Enriching DRUG nodes with SMILES and name_lower...")
        drugs = session.run("MATCH (d:DRUG) WHERE d.name IS NOT NULL AND (d.smiles IS NULL OR d.smiles = '') RETURN d.name as name").data()
        print(f"  Found {len(drugs)} DRUG nodes needing SMILES lookup.")
        
        enriched_count = 0
        for item in drugs[:30]:  # batch of 30
            name = item["name"]
            smiles = fetch_drug_smiles(name)
            if smiles:
                session.run(
                    "MATCH (d:DRUG {name: $name}) SET d.smiles = $smiles, d.name_lower = toLower($name)",
                    name=name, smiles=smiles
                )
                print(f"  ✅ {name} -> SMILES: {smiles[:35]}...")
                enriched_count += 1
            else:
                session.run("MATCH (d:DRUG {name: $name}) SET d.name_lower = toLower($name)", name=name)

        print(f"\n[migration] Successfully enriched {enriched_count} drugs with SMILES.")

    driver.close()
    print("[migration] Migration and enrichment completed!")

if __name__ == "__main__":
    run_migration()
