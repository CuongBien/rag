"""
scripts/fetch_fda_food_text.py
Tự động thu thập văn bản tương tác Thức ăn - Đồ uống - Dược phẩm (Food-Drug Interactions)
từ nguồn Dược thư chính thức của Cục quản lý Thực phẩm và Dược phẩm Hoa Kỳ (openFDA):
1. Quét qua 54 hoạt chất mục tiêu trong cơ sở dữ liệu.
2. Trích xuất toàn bộ các đoạn văn gốc của FDA liên quan đến:
   - Hoa quả, nước ép (Grapefruit juice, Citrus, Cranberry...)
   - Đồ uống có cồn (Alcohol, Ethanol, Wine, Beer...)
   - Sữa, chế phẩm từ sữa & Cation đa hóa trị (Milk, Dairy, Calcium, Iron...)
   - Bữa ăn, chế độ ăn (High-fat meal, Empty stomach, Fasting, Tyramine, Vitamin K...)
   - Hướng dẫn dùng thuốc cùng thức ăn trong Dosage & Administration.
3. Lưu từng hoạt chất thành file văn bản thô (Raw text) tại: data/openfda_food/{drug}.txt
4. Đảm bảo hỗ trợ Checkpoint để có thể chạy tiếp mà không bị trùng lặp.
"""

import os
import sys
import json
import ssl
import time
import argparse
import urllib.request
import urllib.parse
import re

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

OUTPUT_DIR = os.path.join(project_root, "data", "openfda_food")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

USAN_ALIASES = {
    "paracetamol": "acetaminophen",
    "salbutamol": "albuterol",
    "aciclovir": "acyclovir",
    "cefalexin": "cephalexin",
}

# Các từ khóa chuyên sâu về Thực phẩm, Đồ uống, Hoa quả, Khoáng chất & Dinh dưỡng
FOOD_KEYWORDS = [
    r"\bfood\b", r"\bfoods\b", r"\bmeal\b", r"\bmeals\b", r"\bfasting\b", r"\bempty stomach\b",
    r"\bgrapefruit\b", r"\bcitrus\b", r"\borange juice\b", r"\bapple juice\b", r"\bcranberry\b",
    r"\balcohol\b", r"\balcoholic\b", r"\bethanol\b", r"\bwine\b", r"\bbeer\b", r"\bliquor\b", 
    r"\bbeverage\b", r"\bbeverages\b", r"\bdrinks\b", r"\bdrink\b",
    r"\bmilk\b", r"\bdairy\b", r"\byogurt\b", r"\bcheese\b", r"\bcalcium\b", r"\biron\b",
    r"\bvitamin k\b", r"\btyramine\b", r"\bcoffee\b", r"\bcaffeine\b", r"\btea\b",
    r"\bhigh-fat\b", r"\blow-fat\b", r"\bsalt substitute\b", r"\bpotassium\b", r"\bsodium intake\b",
    r"\bdietary\b", r"\bchelation\b", r"\bmultivalent cations\b", r"\beating\b", r"\beat\b"
]
FOOD_REGEX = re.compile("|".join(FOOD_KEYWORDS), re.IGNORECASE)

SECTIONS_TO_CHECK = [
    "food_and_drug_interactions",
    "drug_interactions",
    "dosage_and_administration",
    "warnings",
    "warnings_and_cautions",
    "boxed_warning",
    "precautions",
    "directions",
    "ask_doctor",
    "ask_doctor_or_pharmacist",
    "information_for_patients",
    "patient_counseling_information",
    "spl_patient_package_insert",
    "spl_medguide",
    "clinical_pharmacology",
    "pharmacokinetics"
]

def fetch_openfda_label(drug_name: str) -> dict:
    """Gọi API openFDA để lấy nhãn thuốc chính thức (ưu tiên nhãn kê đơn có drug_interactions, rồi đến nhãn OTC)"""
    search_name = drug_name.strip().lower()
    search_name = USAN_ALIASES.get(search_name, search_name)
    
    queries = [
        f'(openfda.generic_name:"{search_name}" OR openfda.substance_name:"{search_name}") AND _exists_:drug_interactions',
        f'openfda.brand_name:"{search_name}" AND _exists_:drug_interactions',
        f'(openfda.generic_name:"{search_name}" OR openfda.substance_name:"{search_name}") AND openfda.route:"oral"',
        f'openfda.generic_name:"{search_name}" OR openfda.brand_name:"{search_name}"',
        f'openfda.substance_name:"{search_name}"'
    ]
    
    for q in queries:
        encoded_q = urllib.parse.quote(q)
        url = f"https://api.fda.gov/drug/label.json?search={encoded_q}&limit=1"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        try:
            with urllib.request.urlopen(req, context=SSL_CTX, timeout=9) as resp:
                if resp.status == 200:
                    res_json = json.loads(resp.read().decode("utf-8"))
                    results = res_json.get("results", [])
                    if results:
                        return results[0]
        except Exception:
            continue
    return None

def extract_food_interaction_blocks(label: dict, drug_name: str) -> list:
    """Bóc tách các khối văn bản nguyên văn của FDA có chứa thông tin về thức ăn/đồ uống"""
    extracted = []

    # 1. Nếu có hẳn mục food_and_drug_interactions
    if "food_and_drug_interactions" in label:
        content = "\n".join(label["food_and_drug_interactions"]).strip()
        cleaned = re.sub(r'<[^>]+>', ' ', content)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        if cleaned:
            extracted.append({
                "section": "FOOD AND DRUG INTERACTIONS",
                "text": cleaned
            })

    # 2. Duyệt qua các mục lâm sàng quan trọng khác
    for sec in SECTIONS_TO_CHECK:
        if sec == "food_and_drug_interactions" or sec not in label:
            continue
        raw_items = label[sec]
        if isinstance(raw_items, str):
            raw_items = [raw_items]
            
        full_sec_text = " ".join(raw_items)
        # Làm sạch mã HTML/XML
        text_clean = re.sub(r'<[^>]+>', ' ', full_sec_text)
        text_clean = re.sub(r'\s+', ' ', text_clean).strip()
        
        # Tách thành các câu hoàn chỉnh
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text_clean) if s.strip()]
        
        matched_indices = set()
        for i, sent in enumerate(sentences):
            if FOOD_REGEX.search(sent):
                # Gom ngữ cảnh xung quanh câu chứa từ khóa (-1 câu trước, +2 câu sau)
                start = max(0, i - 1)
                end = min(len(sentences), i + 3)
                for j in range(start, end):
                    matched_indices.add(j)
                    
        if not matched_indices:
            continue
            
        # Ghép các chỉ số liền kề thành đoạn văn mạch lạc
        curr_block = []
        sec_blocks = []
        for idx in sorted(list(matched_indices)):
            if not curr_block or idx == curr_block[-1] + 1:
                curr_block.append(idx)
            else:
                block_str = ' '.join([sentences[k] for k in curr_block]).strip()
                if len(block_str) > 30:
                    sec_blocks.append(block_str)
                curr_block = [idx]
        if curr_block:
            block_str = ' '.join([sentences[k] for k in curr_block]).strip()
            if len(block_str) > 30:
                sec_blocks.append(block_str)
                
        # Giới hạn tối đa 4 khối quan trọng nhất cho mỗi section để giữ súc tích
        seen_snippets = set()
        for b in sec_blocks[:4]:
            snip = b[:120]
            if snip not in seen_snippets:
                seen_snippets.add(snip)
                sec_title = sec.replace('_', ' ').upper()
                extracted.append({
                    "section": sec_title,
                    "text": b
                })
                
    return extracted

def process_drug(drug_name: str, force: bool = False) -> str:
    """Tải và lưu raw text tương tác thức ăn cho 1 thuốc"""
    file_path = os.path.join(OUTPUT_DIR, f"{drug_name.lower().strip()}.txt")
    
    if not force and os.path.exists(file_path) and os.path.getsize(file_path) > 80:
        return "EXISTS"
        
    label = fetch_openfda_label(drug_name)
    if not label:
        content = f"OFFICIAL FDA LABEL FOR {drug_name.upper()}:\n[WARNING] Không tìm thấy nhãn FDA phù hợp trên openFDA API."
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return "NOT_FOUND"
        
    blocks = extract_food_interaction_blocks(label, drug_name)
    
    brand_names = label.get("openfda", {}).get("brand_name", [])
    brand_str = f" (Brand: {', '.join(brand_names[:3])})" if brand_names else ""
    
    header = [
        f"================================================================================",
        f"OFFICIAL FDA LABEL: FOOD, BEVERAGE & DIETARY INTERACTIONS FOR {drug_name.upper()}{brand_str}",
        f"Source: openFDA (U.S. Food and Drug Administration)",
        f"================================================================================",
        ""
    ]
    
    if not blocks:
        header.append("No explicit food, fruit juice, beverage, or dietary interaction warnings found in FDA label.")
    else:
        for b in blocks:
            header.append(f"### [FDA SECTION: {b['section']}]")
            header.append(b['text'])
            header.append("")
            
    content = "\n".join(header)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    return f"SAVED ({len(blocks)} blocks)"

def main():
    parser = argparse.ArgumentParser(description="Tải dữ liệu Food-Drug Interactions từ openFDA cho 54 thuốc.")
    parser.add_argument("--force", action="store_true", help="Ghi đè tất cả file đã có")
    parser.add_argument("--drug", type=str, help="Chỉ tải cho 1 thuốc cụ thể")
    args = parser.parse_args()

    targets = [args.drug] if args.drug else DRUGS_54
    print(f"🚀 Bắt đầu thu thập tương tác thức ăn/đồ uống openFDA cho {len(targets)} thuốc...")
    print(f"📁 Thư mục lưu trữ: {OUTPUT_DIR}\n")

    saved_count = 0
    skipped_count = 0
    not_found_count = 0

    for idx, drug in enumerate(targets, 1):
        cap_drug = drug.capitalize()
        status = process_drug(drug, force=args.force)
        
        if status == "EXISTS":
            print(f"[{idx}/{len(targets)}] ⏭️ Bỏ qua {cap_drug}: File raw text đã tồn tại.")
            skipped_count += 1
        elif status == "NOT_FOUND":
            print(f"[{idx}/{len(targets)}] ⚠️ {cap_drug}: Không tìm thấy nhãn trên openFDA.")
            not_found_count += 1
        else:
            print(f"[{idx}/{len(targets)}] ✅ {cap_drug}: {status}")
            saved_count += 1
            
        time.sleep(0.2)  # Tuân thủ giới hạn tần suất gọi API openFDA

    print("\n" + "="*60)
    print(f"🏁 HOÀN THÀNH:")
    print(f"   - Đã lưu mới / cập nhật: {saved_count} thuốc")
    print(f"   - Đã có sẵn (Checkpoint): {skipped_count} thuốc")
    print(f"   - Không tìm thấy nhãn: {not_found_count} thuốc")
    print(f"   - Tổng file trong thư mục: {len(os.listdir(OUTPUT_DIR))}")
    print("="*60)

if __name__ == "__main__":
    main()
