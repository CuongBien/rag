"""
scripts/setup_constraints.py
Khởi tạo Unique Constraints & Indexes trên Neo4j AuraDB cho 9 thực thể chuẩn:
DRUG, DRUG_CLASS, DISEASE, CONDITION, ENZYME, TRANSPORTER, TARGET, SIDE_EFFECT, SUBSTANCE.
Giúp lệnh MERGE chạy với độ phức tạp O(1) và ngăn chặn trùng lặp node.
"""

import os
import sys
from dotenv import load_dotenv
from neo4j import GraphDatabase

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def setup_constraints():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(project_root, ".env"), override=True)

    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    neo4j_database = os.getenv("NEO4J_DATABASE")

    print(f"[constraints] Connecting to Neo4j at {neo4j_uri} (db: {neo4j_database})...")
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))

    labels = [
        "DRUG",
        "DRUG_CLASS",
        "DISEASE",
        "CONDITION",
        "ENZYME",
        "TRANSPORTER",
        "TARGET",
        "SIDE_EFFECT",
        "SUBSTANCE"
    ]

    with driver.session(database=neo4j_database) as session:
        print("[constraints] Creating Unique Constraints for 9 schema entities...")
        for label in labels:
            constraint_name = f"constraint_unique_{label.lower()}_name"
            query = f"""
            CREATE CONSTRAINT {constraint_name} IF NOT EXISTS
            FOR (n:{label})
            REQUIRE n.name IS UNIQUE
            """
            try:
                session.run(query)
                print(f"  ✅ Constraint created/verified for :{label}(name)")
            except Exception as e:
                print(f"  ⚠️ Warning creating constraint for {label}: {e}")

        # Tạo thêm index tìm kiếm nhanh cho thuộc tính name_lower
        for label in labels:
            index_name = f"index_{label.lower()}_namelower"
            idx_query = f"""
            CREATE INDEX {index_name} IF NOT EXISTS
            FOR (n:{label})
            ON (n.name_lower)
            """
            try:
                session.run(idx_query)
            except Exception:
                pass

    driver.close()
    print("[constraints] All constraints & indexes established successfully!")

if __name__ == "__main__":
    setup_constraints()
