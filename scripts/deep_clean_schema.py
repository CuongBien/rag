"""
scripts/deep_clean_schema.py
Script tổng vệ sinh và chuẩn hóa triệt để dữ liệu Neo4j:
1. Quy đổi toàn bộ các nhãn y tế phụ/tương đương về đúng 9 nhãn chuẩn:
   DRUG, DRUG_CLASS, DISEASE, CONDITION, ENZYME, TRANSPORTER, TARGET, SIDE_EFFECT, SUBSTANCE.
2. Xóa bỏ các nhãn phụ/nhãn rác còn dính trên node.
3. Detach delete toàn bộ các node rác phi y tế (năm, phim, công ty, quốc gia, phần trăm, boolean...).
4. Dọn sạch các quan hệ rác nối vào node phi y tế.
"""

import os
import sys
from dotenv import load_dotenv
from neo4j import GraphDatabase

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv(r"D:\rag\.env", override=True)
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

STANDARD_LABELS = {
    "DRUG",
    "DRUG_CLASS",
    "DISEASE",
    "CONDITION",
    "ENZYME",
    "TRANSPORTER",
    "TARGET",
    "SIDE_EFFECT",
    "SUBSTANCE"
}

# Bản đồ quy đổi các nhãn tương đương về 9 nhãn chuẩn
LABEL_SYNONYMS = {
    # DRUG
    "drug": "DRUG",
    "medication": "DRUG",
    "pharmaceutical": "DRUG",
    "drug_brand": "DRUG",
    "brand_name": "DRUG",
    "brand": "DRUG",
    "tradename": "DRUG",
    "TRADE_NAME": "DRUG",
    "TRADENAME": "DRUG",
    "BRAND": "DRUG",

    # DRUG_CLASS
    "drug_class": "DRUG_CLASS",
    "classification": "DRUG_CLASS",
    "CLASSIFICATION": "DRUG_CLASS",
    "drug_classification": "DRUG_CLASS",
    "class": "DRUG_CLASS",
    "family": "DRUG_CLASS",

    # DISEASE
    "disease": "DISEASE",
    "disorder": "DISEASE",
    "DISORDER": "DISEASE",
    "pathology": "DISEASE",
    "PATHOLOGY": "DISEASE",
    "medical_condition": "DISEASE",
    "infection": "DISEASE",

    # CONDITION
    "condition": "CONDITION",
    "patient_group": "CONDITION",
    "age_group": "CONDITION",
    "demographic": "CONDITION",

    # ENZYME
    "enzyme": "ENZYME",
    "enzyme_class": "ENZYME",

    # TRANSPORTER
    "transporter": "TRANSPORTER",

    # TARGET
    "target": "TARGET",
    "protein": "TARGET",
    "PROTEIN": "TARGET",
    "receptor": "TARGET",
    "RECEPTOR": "TARGET",
    "ion_channel": "TARGET",
    "mechanism": "TARGET",
    "MECHANISM": "TARGET",
    "subunit": "TARGET",

    # SIDE_EFFECT
    "symptom": "SIDE_EFFECT",
    "SYMPTOM": "SIDE_EFFECT",
    "SYMPTOM_CATEGORY": "SIDE_EFFECT",
    "side_effect": "SIDE_EFFECT",
    "adverse_effect": "SIDE_EFFECT",
    "ADVERSE_EFFECT": "SIDE_EFFECT",
    "effect": "SIDE_EFFECT",
    "EFFECT": "SIDE_EFFECT",
    "toxicity": "SIDE_EFFECT",

    # SUBSTANCE
    "substance": "SUBSTANCE",
    "chemical": "SUBSTANCE",
    "CHEMICAL": "SUBSTANCE",
    "compound": "SUBSTANCE",
    "plant": "SUBSTANCE",
    "PLANT": "SUBSTANCE",
    "dietary_component": "SUBSTANCE",
    "solvent": "SUBSTANCE",
    "biological_compound": "SUBSTANCE",
    "biomolecule": "SUBSTANCE",
    "BIOMOLECULE": "SUBSTANCE",
    "molecule": "SUBSTANCE",
}

def clean_database():
    with driver.session(database=os.getenv("NEO4J_DATABASE")) as session:
        print("=== BƯỚC 1: QUY ĐỔI CÁC NHÃN TƯƠNG ĐƯƠNG VỀ 9 NHÃN CHUẨN ===")
        for syn, target in LABEL_SYNONYMS.items():
            query = f"MATCH (n:`{syn}`) SET n:`{target}` RETURN count(n) as cnt"
            res = session.run(query).single()
            if res and res["cnt"] > 0:
                print(f"  Gán nhãn chuẩn :{target} cho {res['cnt']} node có nhãn :{syn}")

        print("\n=== BƯỚC 2: XÓA CÁC NODE HOÀN TOÀN PHI Y TẾ (KHÔNG THUỘC 9 NHÃN) ===")
        # Lấy tất cả các node trong __Entity__ nhưng không có bất kỳ nhãn nào trong 9 nhãn chuẩn
        query_junk = """
        MATCH (n:__Entity__)
        WHERE NOT any(lbl IN labels(n) WHERE lbl IN [
            'DRUG', 'DRUG_CLASS', 'DISEASE', 'CONDITION', 'ENZYME', 
            'TRANSPORTER', 'TARGET', 'SIDE_EFFECT', 'SUBSTANCE'
        ])
        DETACH DELETE n
        RETURN count(n) as deleted_count
        """
        res_del = session.run(query_junk).single()
        print(f"  Đã xóa sạch {res_del['deleted_count']} node rác phi y tế (công ty, năm, tổ chức, phim, boolean...).")

        print("\n=== BƯỚC 3: GỠ BỎ TẤT CẢ CÁC NHÃN PHỤ/NHÃN RÁC KHỎI CÁC NODE Y TẾ ===")
        # Với mỗi node y tế, chỉ giữ lại: __Entity__, __Node__, và DUY NHẤT 1 nhãn thuộc 9 nhãn chuẩn
        # Lấy danh sách tất cả các nhãn không chuẩn đang còn tồn tại
        res_labels = session.run("""
            MATCH (n:__Entity__)
            UNWIND labels(n) AS lbl
            WITH DISTINCT lbl
            WHERE NOT lbl IN [
                '__Entity__', '__Node__',
                'DRUG', 'DRUG_CLASS', 'DISEASE', 'CONDITION', 'ENZYME', 
                'TRANSPORTER', 'TARGET', 'SIDE_EFFECT', 'SUBSTANCE'
            ]
            RETURN lbl
        """)
        junk_labels = [r["lbl"] for r in res_labels]
        print(f"  Tìm thấy {len(junk_labels)} nhãn phụ/rác cần gỡ bỏ khỏi các node y tế: {junk_labels[:15]}...")
        for j_lbl in junk_labels:
            session.run(f"MATCH (n:`{j_lbl}`) REMOVE n:`{j_lbl}`")
        print("  Đã gỡ bỏ thành công tất cả các nhãn rác!")

        print("\n=== BƯỚC 4: XÓA CÁC QUAN HỆ RÁC (KHÔNG THUỘC 9 QUAN HỆ CHUẨN HOẶC LLAMAINDEX CHUNK) ===")
        res_rels = session.run("""
            MATCH ()-[r]->()
            WHERE NOT type(r) IN [
                'BELONGS_TO', 'TREATS', 'CONTRAINDICATED_IN', 'INTERACTS_WITH',
                'METABOLIZED_BY', 'TRANSPORTED_BY', 'TARGETS', 'CAUSES_SIDE_EFFECT', 'AFFECTS',
                'MENTIONS', 'SOURCE', 'NEXT', 'PREVIOUS'
            ]
            DELETE r
            RETURN count(r) as deleted_rels
        """).single()
        print(f"  Đã xóa {res_rels['deleted_rels']} quan hệ rác (sold_under, developed_by, is_a, v.v.).")

        print("\n=== BƯỚC 5: KIỂM TRA LẠI THỐNG KÊ SAU KHI DỌN DẸP ===")
        res_final = session.run("""
            MATCH (n:__Entity__) 
            RETURN labels(n) AS nhan_node, count(n) AS so_luong 
            ORDER BY so_luong DESC
        """)
        for r in res_final:
            print(f"  {r['so_luong']}: {r['nhan_node']}")

    driver.close()
    print("\n[clean] Hoàn tất tổng vệ sinh Neo4j!")

if __name__ == "__main__":
    clean_database()
