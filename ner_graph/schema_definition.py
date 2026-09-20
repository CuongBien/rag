"""
ner_graph/schema_definition.py
Định nghĩa Schema chuẩn y dược cho Knowledge Graph (9 Thực Thể & 9 Mối Quan Hệ).
Nguồn chân lý duy nhất (Single Source of Truth) cho hệ thống Medical Graph RAG.
"""

from typing import Literal, List, Tuple
import urllib.request
import urllib.parse
import json

# 1. Định nghĩa 9 Thực thể (Entities)
EntityTypes = Literal[
    "DRUG",
    "DRUG_CLASS",
    "DISEASE",
    "CONDITION",
    "ENZYME",
    "TRANSPORTER",
    "TARGET",
    "SIDE_EFFECT",
    "SUBSTANCE",
]

# 2. Định nghĩa 9 Mối quan hệ (Relations)
RelationTypes = Literal[
    "BELONGS_TO",
    "TREATS",
    "CONTRAINDICATED_IN",
    "INTERACTS_WITH",
    "METABOLIZED_BY",
    "TRANSPORTED_BY",
    "TARGETS",
    "CAUSES_SIDE_EFFECT",
    "AFFECTS",
]

# 3. Thuộc tính của Thực thể (Entity Properties)
ENTITY_PROPERTIES: List[Tuple[str, str]] = [
    ("smiles", "SMILES chemical structure notation for pure active ingredients"),
    ("half_life", "Elimination half-life of the drug in human body"),
    ("atc_code", "Anatomical Therapeutic Chemical (ATC) classification code"),
    ("icd10", "International Classification of Diseases (ICD-10) code"),
    ("category", "Substance category: Fruit, Herb, Food, or Beverage"),
    ("active_compound", "Active chemical / bioactive compound contained in the substance"),
]

# 4. Thuộc tính của Mối quan hệ (Relation Properties)
RELATION_PROPERTIES: List[Tuple[str, str]] = [
    ("dosage", "Recommended dosage or dosage range (e.g., 500mg daily)"),
    ("indication_type", "Line of therapy: First-line or Second-line"),
    ("level", "Contraindication severity level: Absolute or BlackBox"),
    ("reason", "Physiological or clinical rationale for contraindication"),
    ("severity", "Interaction severity level: Major, Moderate, or Minor"),
    ("clinical_effect", "Specific clinical consequence or manifestation of interaction"),
    ("role", "Pharmacological role: substrate, inhibitor, inducer, or antagonist"),
    ("action", "Pharmacodynamic action on target: agonist, antagonist, or inhibitor"),
    ("frequency", "Incidence rate of adverse effect: Common, Rare, or Very Rare"),
]

# 5. Bộ ba hợp lệ để xác thực (Validation Triples)
VALIDATION_SCHEMA: List[Tuple[str, str, str]] = [
    ("DRUG", "BELONGS_TO", "DRUG_CLASS"),
    ("DRUG", "TREATS", "DISEASE"),
    ("DRUG", "TREATS", "CONDITION"),
    ("DRUG", "CONTRAINDICATED_IN", "CONDITION"),
    ("DRUG", "CONTRAINDICATED_IN", "DISEASE"),
    ("DRUG", "INTERACTS_WITH", "DRUG"),
    ("DRUG", "INTERACTS_WITH", "SUBSTANCE"),
    ("DRUG", "METABOLIZED_BY", "ENZYME"),
    ("DRUG", "TRANSPORTED_BY", "TRANSPORTER"),
    ("DRUG", "TARGETS", "TARGET"),
    ("DRUG", "CAUSES_SIDE_EFFECT", "SIDE_EFFECT"),
    ("SUBSTANCE", "AFFECTS", "ENZYME"),
    ("SUBSTANCE", "AFFECTS", "TRANSPORTER"),
    ("SUBSTANCE", "AFFECTS", "TARGET"),
]

# 6. Prompt hướng dẫn trích xuất chuyên sâu cho Dược lý
MEDICAL_EXTRACTION_PROMPT = """You are an expert Clinical Pharmacologist and Biomedical Knowledge Engineer.
Extract precise knowledge graph entities and relations from the medical text strictly adhering to the schema below.

Allowed Entity Types:
- DRUG: Pure active chemical ingredient (e.g., Paracetamol, Simvastatin, Warfarin, Diazepam).
- DRUG_CLASS: Pharmacological/therapeutic class (e.g., Statin, Benzodiazepine, ACE Inhibitor, Macrolide).
- DISEASE: Clinical disease, pathology, or medical condition being treated (e.g., Hypertension, Peptic Ulcer).
- CONDITION: Patient physiological status or terrain (e.g., Pregnancy, Renal Impairment, Hepatic Failure, Pediatric, Elderly, Breastfeeding).
- ENZYME: Metabolic enzymes, primarily Cytochrome P450 (e.g., CYP3A4, CYP2C9, CYP2D6, ALDH).
- TRANSPORTER: Membrane transport proteins / efflux pumps (e.g., P-gp, OATP1B1, OATP2B1).
- TARGET: Biological receptor, ion channel, or target enzyme (e.g., Mu-opioid receptor, HMG-CoA reductase, COX-2, ACE).
- SIDE_EFFECT: Adverse drug reaction or toxicity (e.g., Rhabdomyolysis, Bleeding, Hepatotoxicity, Drowsiness).
- SUBSTANCE: Food, dietary factor, herbal product, or beverage (e.g., Alcohol, Grapefruit Juice, St. John's Wort, Milk, Tobacco).

Allowed Relations and required properties:
- (:DRUG) -[:BELONGS_TO]-> (:DRUG_CLASS)
- (:DRUG) -[:TREATS {dosage: "...", indication_type: "First-line/Second-line"}]-> (:DISEASE | :CONDITION)
- (:DRUG) -[:CONTRAINDICATED_IN {level: "Absolute/BlackBox", reason: "..."}]-> (:CONDITION | :DISEASE)
- (:DRUG) -[:INTERACTS_WITH {severity: "Major/Moderate", clinical_effect: "..."}]-> (:DRUG | :SUBSTANCE)
- (:DRUG) -[:METABOLIZED_BY {role: "substrate/inhibitor/inducer"}]-> (:Enzyme)
- (:DRUG) -[:TRANSPORTED_BY {role: "substrate/inhibitor"}]-> (:Transporter)
- (:DRUG) -[:TARGETS {action: "agonist/antagonist/inhibitor"}]-> (:Target)
- (:DRUG) -[:CAUSES_SIDE_EFFECT {frequency: "Common/Rare"}]-> (:Side_Effect)
- (:SUBSTANCE) -[:AFFECTS {role: "inhibitor/inducer/antagonist"}]-> (:Enzyme | :Transporter | :Target)

Extract up to {max_triplets_per_chunk} of the highest clinical significance relationships.
Do NOT invent relations or entity types outside this ontology.

Medical text:
-------
{text}
-------
"""

def fetch_drug_smiles(drug_name: str) -> str:
    """
    Tra cứu chuỗi SMILES chuẩn từ NIH Chemical Identifier Resolver (Cactus) hoặc PubChem.
    Trả về chuỗi SMILES hoặc chuỗi rỗng nếu không tìm thấy.
    """
    import ssl
    clean_name = drug_name.strip()
    if not clean_name or len(clean_name) < 3:
        return ""
    
    # 1. Thử qua NIH Cactus (Rất nhanh và chuẩn)
    try:
        ctx = ssl._create_unverified_context()
        encoded = urllib.parse.quote(clean_name)
        url = f"https://cactus.nci.nih.gov/chemical/structure/{encoded}/smiles"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=8) as resp:
            if resp.status == 200:
                smiles = resp.read().decode("utf-8").strip()
                if smiles and not smiles.startswith("<html") and len(smiles) < 500:
                    return smiles
    except Exception:
        pass

    # 2. Thử qua PubChem API
    try:
        ctx = ssl._create_unverified_context()
        encoded = urllib.parse.quote(clean_name)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/property/CanonicalSMILES/JSON"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=8) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                props = data.get("PropertyTable", {}).get("Properties", [])
                if props and "CanonicalSMILES" in props[0]:
                    return props[0]["CanonicalSMILES"]
    except Exception:
        pass

    return ""

