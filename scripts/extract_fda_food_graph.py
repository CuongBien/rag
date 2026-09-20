"""
scripts/extract_fda_food_graph.py
Bóc tách tri thức Tương tác Thức ăn - Đồ uống (Food-Drug Interactions) từ văn bản gốc openFDA
bằng Cloud LLM (Gemini) và nạp vào đồ thị Neo4j AuraDB:
1. Đọc văn bản thô từ data/openfda_food/{drug}.txt
2. Trích xuất khách quan 100% bằng LLM:
   - Substance: Grapefruit juice, Alcohol, Dairy products, High-fat meal...
   - Target/Enzyme: CYP3A4, CYP2E1, OATP1B1, Chelation...
   - Action: inhibitor, inducer, delayed_absorption, additive_toxicity...
   - Severity: Major, Moderate, Minor
   - Clinical_Effect: Tác hại lâm sàng (Tăng nồng độ máu, tiêu cơ vân...)
   - Management: Khuyến cáo xử trí từ FDA (Tránh dùng, uống cách 2 giờ...)
   - Evidence_Quote: Câu văn gốc chính thức từ FDA làm bằng chứng hiển thị trên đồ thị.
3. Nạp vào Neo4j:
   - (:DRUG) -[:INTERACTS_WITH {severity, clinical_effect, management, evidence_quote}]-> (:SUBSTANCE)
   - (:SUBSTANCE) -[:AFFECTS {action, evidence_quote}]-> (:ENZYME | :TARGET)
4. Checkpoint an toàn tại openfda_food_graph_checkpoint.json
"""

import os
import sys
import json
import time
import re
import argparse
from dotenv import load_dotenv
from neo4j import GraphDatabase

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

load_dotenv(os.path.join(project_root, ".env"), override=True)

from ner_graph.config import load_config
from ner_graph.llm_client import create_llm

DATA_DIR = os.path.join(project_root, "data", "openfda_food")
CHECKPOINT_FILE = os.path.join(project_root, "openfda_food_graph_checkpoint.json")

ENZYME_REGEX = re.compile(r"^(CYP\w+|UGT\w+|MAO\w*|[a-z0-9_]+ase)$", re.IGNORECASE)

EXTRACTION_PROMPT_TEMPLATE = """You are an expert Clinical Pharmacologist and Knowledge Graph Engineer.
Analyze the official FDA Drug Label text below for the active pharmaceutical ingredient "{drug_name}" and objectively extract all Food, Beverage, Fruit, Dairy, Mineral, and Dietary interactions.

Rules:
1. Extract every distinct dietary interaction documented in the text (e.g. Grapefruit juice, Alcohol, Meals/Food, Dairy products, Vitamin K foods, Calcium, Caffeine, etc.).
2. If the text or pharmacokinetics indicates the underlying enzyme or target (e.g. CYP3A4, CYP2E1, OATP1B1, Chelation, GABA_A receptor), capture it in "target_or_enzyme". If not specified or general, set to null.
3. For each interaction:
   - "substance": Standardized English entity name (e.g. "Grapefruit juice", "Alcohol", "Dairy products", "High-fat meal", "Vitamin K rich foods", "Calcium-fortified juice")
   - "target_or_enzyme": Affected enzyme, transporter, or target receptor if mentioned (e.g. "CYP3A4", "CYP2E1", "OATP1B1", null)
   - "action": Pharmacological mechanism/action (e.g. "inhibitor", "inducer", "delayed_absorption", "additive_toxicity", "chelation", "potentiates_effect")
   - "severity": Clinical severity level ("Major", "Moderate", "Minor")
   - "clinical_effect": Concise clinical consequence summary (e.g. "Increases plasma concentration and risk of myopathy and rhabdomyolysis")
   - "management": Actionable recommendation from FDA (e.g. "Avoid grapefruit juice when taking simvastatin")
   - "evidence_quote": Exact, verbatim sentence directly from the FDA text as proof.

Output MUST be a valid JSON array of objects.
Do not include markdown code block ticks (no ```json). Output ONLY valid raw JSON array.
If no dietary/food interactions are documented, output [].

OFFICIAL FDA TEXT:
{text}
"""

def clean_json_response(raw_text: str) -> list:
    """Loại bỏ markdown codeblock và parse JSON mảng an toàn"""
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "interactions" in data:
            return data["interactions"]
        return []
    except Exception as e:
        # Thử trích xuất mảng JSON bằng regex
        match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        print(f"    ⚠️ Lỗi parse JSON LLM: {e}")
        return []

def ingest_food_interactions(session, drug_name: str, interactions: list):
    """Nạp thực thể và quan hệ tương tác thức ăn vào Neo4j AuraDB"""
    cap_drug = drug_name.capitalize()
    
    for item in interactions:
        substance = item.get("substance", "").strip()
        if not substance or len(substance) < 2:
            continue
            
        cap_sub = substance.title()
        target = item.get("target_or_enzyme")
        if target:
            target = target.strip()
            
        action = item.get("action", "interacts_with").strip()
        severity = item.get("severity", "Moderate").strip().capitalize()
        clinical_effect = item.get("clinical_effect", "").strip()
        management = item.get("management", "").strip()
        evidence_quote = item.get("evidence_quote", "").strip()

        # 1. Tạo node SUBSTANCE và quan hệ INTERACTS_WITH với DRUG
        session.run("""
            MERGE (d:DRUG {name: $drug_name})
            SET d.name_lower = toLower($drug_name),
                d:`__Entity__`, d:`__Node__`
            MERGE (sub:SUBSTANCE {name: $sub_name})
            SET sub.name_lower = toLower($sub_name),
                sub.category = 'food_and_beverage',
                sub:`__Entity__`, sub:`__Node__`
            MERGE (d)-[r:INTERACTS_WITH]->(sub)
            SET r.severity = $severity,
                r.clinical_effect = $clinical_effect,
                r.management = $management,
                r.evidence_quote = $evidence_quote,
                r.source = 'openFDA',
                r.interaction_type = 'Food-Drug'
        """,
            drug_name=cap_drug,
            sub_name=cap_sub,
            severity=severity,
            clinical_effect=clinical_effect,
            management=management,
            evidence_quote=evidence_quote
        )

        # 2. Nếu có target hoặc enzyme (e.g. CYP3A4)
        if target and target.lower() not in ["null", "none", "n/a", ""]:
            cap_target = target.upper() if len(target) <= 7 else target.title()
            
            is_enzyme = bool(ENZYME_REGEX.match(target))
            target_label = "ENZYME" if is_enzyme else "TARGET"
            
            query = f"""
                MERGE (sub:SUBSTANCE {{name: $sub_name}})
                MERGE (t:{target_label} {{name: $target_name}})
                SET t.name_lower = toLower($target_name),
                    t:`__Entity__`, t:`__Node__`
                MERGE (sub)-[r:AFFECTS]->(t)
                SET r.action = $action,
                    r.evidence_quote = $evidence_quote,
                    r.source = 'openFDA'
            """
            session.run(query, sub_name=cap_sub, target_name=cap_target, action=action, evidence_quote=evidence_quote)

def main():
    parser = argparse.ArgumentParser(description="Trích xuất Food-Drug Interactions từ file raw text và nạp vào Neo4j.")
    parser.add_argument("--drug", type=str, help="Chỉ xử lý 1 thuốc cụ thể")
    parser.add_argument("--force", action="store_true", help="Chạy lại cả những thuốc đã có checkpoint")
    parser.add_argument("--limit", type=int, default=0, help="Giới hạn số thuốc xử lý")
    args = parser.parse_args()

    config = load_config(project_root)
    llm = create_llm(config)

    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USERNAME")
    neo4j_pw = os.getenv("NEO4J_PASSWORD")
    neo4j_db = os.getenv("NEO4J_DATABASE")

    print(f"[openFDA Food Graph] Kết nối Neo4j AuraDB: {neo4j_uri}...")
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pw))

    # Tải checkpoint
    processed_set = set()
    if not args.force and os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                processed_set.update(json.load(f))
        except Exception:
            pass

    # Lấy danh sách file raw text
    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]
    if args.drug:
        target_name = f"{args.drug.lower().strip()}.txt"
        all_files = [target_name] if target_name in all_files else [f"{args.drug.lower().strip()}.txt"]

    pending_files = [f for f in all_files if f.replace(".txt", "").lower() not in processed_set] if not args.force else all_files
    if args.limit > 0:
        pending_files = pending_files[:args.limit]

    print(f"[openFDA Food Graph] 🎯 Cần xử lý: {len(pending_files)} thuốc (Đã có checkpoint: {len(processed_set)} thuốc).\n")

    with driver.session(database=neo4j_db) as session:
        for idx, file_name in enumerate(pending_files, 1):
            drug_name = file_name.replace(".txt", "").strip()
            cap_drug = drug_name.capitalize()
            file_path = os.path.join(DATA_DIR, file_name)

            if not os.path.exists(file_path):
                print(f"[{idx}/{len(pending_files)}] ⚠️ Không có file raw text: {file_path}")
                continue

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()

            if not text or "No explicit food" in text or "Không tìm thấy nhãn FDA" in text:
                print(f"[{idx}/{len(pending_files)}] ⏭️ {cap_drug}: Không có tương tác thức ăn/đồ uống ghi nhận.")
                processed_set.add(drug_name.lower())
                with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                    json.dump(sorted(list(processed_set)), f, indent=2)
                continue

            print(f"[{idx}/{len(pending_files)}] 🤖 Đang trích xuất LLM: {cap_drug}...")
            # Giới hạn độ dài văn bản an toàn (Gemini xử lý được 1M token, cắt ở 25000 ký tự là quá đủ cho các mục food)
            prompt = EXTRACTION_PROMPT_TEMPLATE.format(drug_name=cap_drug, text=text[:25000])
            
            try:
                res = llm.complete(prompt)
                interactions = clean_json_response(res.text)
                
                if interactions:
                    ingest_food_interactions(session, cap_drug, interactions)
                    print(f"    ✅ Đã nạp {len(interactions)} tương tác thức ăn/đồ uống kèm bằng chứng FDA.")
                    for item in interactions:
                        sub = item.get("substance", "")
                        sev = item.get("severity", "Moderate")
                        tgt = f" -> {item.get('target_or_enzyme')}" if item.get('target_or_enzyme') else ""
                        print(f"       • [{sev}] {cap_drug} ↔ {sub}{tgt}")
                else:
                    print(f"    ℹ️ LLM không tìm thấy tương tác lâm sàng cụ thể.")

                processed_set.add(drug_name.lower())
                with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                    json.dump(sorted(list(processed_set)), f, indent=2)

            except Exception as e:
                print(f"    ❌ Lỗi khi xử lý {cap_drug}: {e}")

            time.sleep(0.5)

    print("\n" + "="*60)
    print(f"🏁 HOÀN TẤT TRÍCH XUẤT VÀ NẠP ĐỒ THỊ FDI!")
    print(f"   - Tổng số thuốc trong Checkpoint: {len(processed_set)}")
    print("="*60)

if __name__ == "__main__":
    main()
