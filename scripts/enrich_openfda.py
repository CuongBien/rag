"""
scripts/enrich_openfda.py
Tự động thu thập dữ liệu y dược chuẩn quốc tế từ openFDA (Hoa Kỳ) và NIH Chemical Resolver:
1. openFDA: Boxed Warning (Cảnh báo hộp đen), Chống chỉ định (Contraindications), Tương tác thuốc (Drug Interactions).
2. NIH Cactus: Chuỗi SMILES, Công thức hóa học (Formula), Khối lượng phân tử (MW), Danh pháp IUPAC.
3. Trích xuất khách quan 100% qua SchemaLLMPathExtractor (KHÔNG hardcode bất kỳ từ khóa nào):
   Tự động phát hiện mọi Bệnh lý (Myasthenia gravis, Glaucoma...), Cơ địa (Trẻ em dưới 6 tháng, Suy gan...),
   Thực phẩm/Chất kích thích, Đích tác dụng (GABA_A receptor...), và Tác dụng phụ.
4. Nạp trực tiếp vào đồ thị tri thức Neo4j AuraDB.
"""

import os
import sys
import json
import time
import ssl
import urllib.request
import urllib.parse
from dotenv import load_dotenv
from neo4j import GraphDatabase

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv(os.path.join(project_root, ".env"), override=True)

from llama_index.core import Document
from ner_graph.config import load_config
from ner_graph.llm_client import create_llm
from ner_graph.schema_definition import (
    EntityTypes,
    RelationTypes,
    ENTITY_PROPERTIES,
    RELATION_PROPERTIES,
    VALIDATION_SCHEMA,
    MEDICAL_EXTRACTION_PROMPT,
)
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor

SSL_CTX = ssl._create_unverified_context()

DRUGS_54 = [
    "aciclovir", "alprazolam", "amitriptyline", "amlodipine", "amoxicillin", 
    "aspirin", "atorvastatin", "azithromycin", "caffeine", "cefalexin", 
    "celecoxib", "cetirizine", "ciprofloxacin", "clopidogrel", "diazepam", 
    "diclofenac", "digoxin", "doxycycline", "escitalopram", "esomeprazole", 
    "fentanyl", "fluconazole", "fluoxetine", "fluticasone", "gabapentin", 
    "glipizide", "haloperidol", "ibuprofen", "insulin glargine", "levothyroxine", 
    "lisinopril", "lithium", "loperamide", "loratadine", "losartan", 
    "metformin", "methotrexate", "metoprolol", "metronidazole", "montelukast", 
    "morphine", "naproxen", "omeprazole", "pantoprazole", "paracetamol", 
    "ranitidine", "rivaroxaban", "salbutamol", "sertraline", "sildenafil", 
    "simvastatin", "tramadol", "warfarin", "zolpidem"
]

def fetch_nih_chemical_props(drug_name: str) -> dict:
    """Lấy SMILES, IUPAC Name, Formula, MW từ NIH Cactus"""
    props = {"smiles": "", "formula": "", "mw": "", "iupac_name": ""}
    clean = drug_name.strip()
    if clean.lower() == "paracetamol":
        clean = "acetaminophen"  # NIH/FDA dùng acetaminophen
        
    encoded = urllib.parse.quote(clean)
    for key in ["smiles", "formula", "mw", "iupac_name"]:
        url = f"https://cactus.nci.nih.gov/chemical/structure/{encoded}/{key}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, context=SSL_CTX, timeout=6) as resp:
                if resp.status == 200:
                    val = resp.read().decode("utf-8").strip()
                    if val and not val.startswith("<html") and len(val) < 800:
                        props[key] = val
        except Exception:
            pass
    return props

USAN_ALIASES = {
    "paracetamol": "acetaminophen",
    "salbutamol": "albuterol",
    "aciclovir": "acyclovir",
    "cefalexin": "cephalexin",
}

def fetch_openfda_label(drug_name: str) -> dict:
    """
    Lấy Cảnh báo hộp đen, Chống chỉ định, Tương tác thuốc từ openFDA.
    Hỗ trợ cả thuốc kê đơn (Rx) lẫn thuốc không kê đơn (OTC) và tên chuẩn USAN.
    """
    data = {
        "boxed_warning": "",
        "contraindications": "",
        "drug_interactions": ""
    }
    search_name = drug_name.strip().lower()
    search_name = USAN_ALIASES.get(search_name, search_name)

    queries = [
        f'openfda.generic_name:"{search_name}"',
        f'openfda.substance_name:"{search_name}"',
        f'openfda.brand_name:"{search_name}"'
    ]

    for q in queries:
        encoded_q = urllib.parse.quote(q)
        url = f"https://api.fda.gov/drug/label.json?search={encoded_q}&limit=1"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, context=SSL_CTX, timeout=8) as resp:
                if resp.status == 200:
                    res_json = json.loads(resp.read().decode("utf-8"))
                    result = res_json.get("results", [{}])[0]
                    
                    # 1. Cảnh báo Hộp Đen / Cảnh báo nguy hiểm (Rx hoặc OTC)
                    if "boxed_warning" in result:
                        data["boxed_warning"] = " ".join(result["boxed_warning"]).strip()[:1200]
                    elif "warnings_and_cautions" in result:
                        data["boxed_warning"] = " ".join(result["warnings_and_cautions"]).strip()[:1200]
                    elif "warnings" in result:
                        data["boxed_warning"] = " ".join(result["warnings"]).strip()[:1200]

                    # 2. Chống chỉ định (Rx: contraindications, OTC: do_not_use)
                    if "contraindications" in result:
                        data["contraindications"] = " ".join(result["contraindications"]).strip()[:1200]
                    elif "do_not_use" in result:
                        data["contraindications"] = " ".join(result["do_not_use"]).strip()[:1200]

                    # 3. Tương tác thuốc & Cơ địa (Rx: drug_interactions, OTC: ask_doctor)
                    if "drug_interactions" in result:
                        data["drug_interactions"] = " ".join(result["drug_interactions"]).strip()[:1200]
                    elif "ask_doctor_or_pharmacist" in result:
                        data["drug_interactions"] = " ".join(result["ask_doctor_or_pharmacist"]).strip()[:1200]
                    elif "ask_doctor" in result:
                        data["drug_interactions"] = " ".join(result["ask_doctor"]).strip()[:1200]

                    if data["boxed_warning"] or data["contraindications"] or data["drug_interactions"]:
                        break
        except Exception:
            continue
    return data

def build_combined_fda_text(drug_cap: str, fda_data: dict) -> str:
    """Tập hợp văn bản chính thức của FDA để LLM đọc và bóc tách"""
    chunks = [f"OFFICIAL FDA LABEL FOR {drug_cap}:"]
    if fda_data["boxed_warning"]:
        chunks.append(f"FDA WARNINGS / BOXED WARNING:\n{fda_data['boxed_warning']}")
    if fda_data["contraindications"]:
        chunks.append(f"FDA CONTRAINDICATIONS / DO NOT USE:\n{fda_data['contraindications']}")
    if fda_data["drug_interactions"]:
        chunks.append(f"FDA DRUG INTERACTIONS / PRECAUTIONS:\n{fda_data['drug_interactions']}")
    
    if len(chunks) == 1:
        return ""
    return "\n\n".join(chunks)

def enrich_all_drugs():
    config = load_config(project_root)
    llm = create_llm(config)

    extractor = SchemaLLMPathExtractor(
        llm=llm,
        possible_entities=EntityTypes,
        possible_entity_props=ENTITY_PROPERTIES,
        possible_relations=RelationTypes,
        possible_relation_props=RELATION_PROPERTIES,
        kg_validation_schema=VALIDATION_SCHEMA,
        extract_prompt=MEDICAL_EXTRACTION_PROMPT,
        strict=True,
        max_triplets_per_chunk=12
    )

    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USERNAME")
    neo4j_pw = os.getenv("NEO4J_PASSWORD")
    neo4j_db = os.getenv("NEO4J_DATABASE")

    print(f"[openFDA] Connecting to Neo4j at {neo4j_uri}...")
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pw))

    checkpoint_file = os.path.join(project_root, "openfda_checkpoint.json")
    processed_set = set()
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                processed_set.update(json.load(f))
        except Exception as e:
            print(f"[openFDA] Warning reading checkpoint: {e}")

    with driver.session(database=neo4j_db) as session:
        # Đồng bộ các thuốc đã nạp thực sự có nội dung FDA từ Neo4j
        try:
            db_drugs = session.run("""
                MATCH (d:DRUG) 
                WHERE (d.boxed_warning IS NOT NULL AND d.boxed_warning <> '') 
                   OR (d.contraindications IS NOT NULL AND d.contraindications <> '') 
                RETURN coalesce(d.name, '') AS name
            """).data()
            for r in db_drugs:
                if r.get("name"):
                    processed_set.add(r["name"].lower().strip())
        except Exception:
            pass

        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(sorted(list(processed_set)), f, indent=2)

        print(f"[openFDA] 🎯 {len(DRUGS_54) - len(processed_set)} thuốc cần xử lý (Đã có checkpoint: {len(processed_set)} thuốc).\n")

        for idx, drug in enumerate(DRUGS_54, 1):
            cap_drug = drug.capitalize()
            drug_key = drug.lower().strip()

            if drug_key in processed_set:
                print(f"[{idx}/{len(DRUGS_54)}] ⏭️ Đã có dữ liệu openFDA: Bỏ qua {cap_drug}")
                continue

            print(f"[{idx}/{len(DRUGS_54)}] Đang xử lý: {cap_drug}...")

            # 1. Kéo hóa dược từ NIH Cactus
            chem = fetch_nih_chemical_props(drug)

            # 2. Kéo nhãn lâm sàng từ openFDA
            fda = fetch_openfda_label(drug)

            # 3. Cập nhật lý lịch hóa dược & văn bản FDA vào node DRUG
            # Trước tiên tìm node DRUG hiện có theo tên
            session.run("""
                MERGE (d:DRUG {name: $cap_drug})
                SET d.name_lower = toLower($cap_drug),
                    d.smiles = CASE WHEN $smiles <> '' THEN $smiles ELSE coalesce(d.smiles, '') END,
                    d.formula = CASE WHEN $formula <> '' THEN $formula ELSE coalesce(d.formula, '') END,
                    d.mw = CASE WHEN $mw <> '' THEN $mw ELSE coalesce(d.mw, '') END,
                    d.iupac_name = CASE WHEN $iupac_name <> '' THEN $iupac_name ELSE coalesce(d.iupac_name, '') END,
                    d.boxed_warning = CASE WHEN $boxed_warning <> '' THEN $boxed_warning ELSE coalesce(d.boxed_warning, '') END,
                    d.contraindications = CASE WHEN $contraindications <> '' THEN $contraindications ELSE coalesce(d.contraindications, '') END,
                    d.fda_source = 'openFDA & NIH Cactus',
                    d:`__Entity__`, d:`__Node__`
            """, 
                cap_drug=cap_drug,
                smiles=chem["smiles"],
                formula=chem["formula"],
                mw=chem["mw"],
                iupac_name=chem["iupac_name"],
                boxed_warning=fda["boxed_warning"],
                contraindications=fda["contraindications"]
            )

            # 4. Trích xuất Khách quan 100% qua LLM Schema Extractor
            fda_text = build_combined_fda_text(cap_drug, fda)
            if fda_text:
                try:
                    doc = Document(text=fda_text)
                    extracted_doc = extractor([doc])[0]
                    nodes = extracted_doc.metadata.get("nodes", [])
                    relations = extracted_doc.metadata.get("relations", [])

                    # Nạp các node trích xuất khách quan vào Neo4j an toàn
                    for n in nodes:
                        lbl = n.label
                        clean_name = n.name.strip()
                        session.run(f"""
                            MERGE (node:`{lbl}` {{name: $name}})
                            SET node.name_lower = toLower($name),
                                node:`__Entity__`, node:`__Node__`
                        """, name=clean_name)

                    # Nạp các cạnh trích xuất kèm properties phong phú
                    for r in relations:
                        r_lbl = r.label
                        s_name = r.source_id.strip()
                        t_name = r.target_id.strip()
                        session.run(f"""
                            MATCH (s) WHERE toLower(s.name) = toLower($s_name)
                            MATCH (t) WHERE toLower(t.name) = toLower($t_name)
                            MERGE (s)-[rel:`{r_lbl}`]->(t)
                            SET rel += $props
                        """, 
                            s_name=s_name, 
                            t_name=t_name,
                            props=r.properties
                        )

                    print(f"    ✅ Extracted: {len(nodes)} nodes, {len(relations)} clinical relations.")
                except Exception as e:
                    print(f"    ⚠️ Warning extracting with LLM: {e}")
            else:
                print(f"    ℹ️ Không tìm thấy nhãn mở rộng trên openFDA cho {cap_drug} (vẫn lưu NIH data).")

            # Lưu checkpoint thành công cho thuốc này
            processed_set.add(drug_key)
            try:
                with open(checkpoint_file, "w", encoding="utf-8") as f:
                    json.dump(sorted(list(processed_set)), f, indent=2)
            except Exception:
                pass

            time.sleep(0.5)

    driver.close()
    print("\n🎉 [openFDA & NIH] Hoàn tất nạp dữ liệu khách quan 100% cho 54 thuốc!")

if __name__ == "__main__":
    enrich_all_drugs()
