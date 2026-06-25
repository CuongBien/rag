import os
import json
import requests
import difflib
import streamlit as st
from PIL import Image
from io import BytesIO
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

CACHE_FILE = "chem_cache.json"

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_cache(cache_data):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache_data, f)

def get_from_cache(query, q_type):
    cache = load_cache()
    if q_type not in cache:
        cache[q_type] = {}
    return cache[q_type].get(query)

def update_cache(query, q_type, result):
    cache = load_cache()
    if q_type not in cache:
        cache[q_type] = {}
    cache[q_type][query] = result
    save_cache(cache)

@st.cache_resource
def load_molt5_model():
    # Tìm đường dẫn tuyệt đối đến MolT5 checkpoint
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "MolT5_model", "MolT5_Checkpoint_bigger")
    
    if not os.path.exists(model_path):
        print(f"Không tìm thấy model tại: {model_path}")
        return None, None
        
    print(f"Đang nạp mô hình MolT5 từ {model_path} vào CUDA...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = T5ForConditionalGeneration.from_pretrained(model_path, local_files_only=True).to(device)
        model.eval()
        return tokenizer, model
    except Exception as e:
        print(f"Lỗi nạp mô hình MolT5: {e}")
        return None, None

def translate_via_local_model(query, direction):
    tokenizer, model = load_molt5_model()
    if tokenizer is None or model is None:
        return None
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Chuẩn bị Prompt
    if direction == "smiles_to_iupac":
        prompt = f"SMILES to IUPAC: {query}"
    else:
        prompt = f"IUPAC to SMILES: {query}"
        
    try:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=512,
                num_beams=5,
                early_stopping=True
            )
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    except Exception as e:
        print(f"MolT5 Inference Error: {e}")
        return None
        
    # Cơ chế ZERO-HALLUCINATION (Chống ảo giác bằng RDKit)
    if direction == "iupac_to_smiles":
        if RDKIT_AVAILABLE:
            mol = Chem.MolFromSmiles(prediction)
            if mol is None:
                # Ảo giác: Hóa chất vô lý / Không có thật
                print(f"🚫 [ZERO-HALLUCINATION] AI sinh ra SMILES lỗi/vô lý: '{prediction}'. Tự động Fallback sang Tier 3 (PubChem API)!")
                return None
            else:
                # --- CYCLE CONSISTENCY (BACK-TRANSLATION) ---
                back_prompt = f"SMILES to IUPAC: {prediction}"
                try:
                    back_input_ids = tokenizer(back_prompt, return_tensors="pt").input_ids.to(device)
                    with torch.no_grad():
                        back_outputs = model.generate(back_input_ids, max_length=512, num_beams=5, early_stopping=True)
                    back_prediction = tokenizer.decode(back_outputs[0], skip_special_tokens=True).strip()
                    
                    similarity = difflib.SequenceMatcher(None, query.lower(), back_prediction.lower()).ratio()
                    is_substring = (len(back_prediction.strip()) > 3) and (query.lower() in back_prediction.lower() or back_prediction.lower() in query.lower())
                    
                    if similarity < 0.4 and not is_substring:
                        print(f"🚫 [CYCLE-CONSISTENCY] Ảo giác Ngữ nghĩa! Đầu vào '{query}' sinh ra SMILES '{prediction}', nhưng khi dịch ngược lại ra '{back_prediction}' (Độ giống: {similarity:.2f}). Bắt buộc ném sang Tier 3!")
                        return None
                    else:
                        print(f"✅ [CYCLE-CONSISTENCY] Xác thực thành công: Đầu vào '{query}' khớp với kết quả dịch ngược '{back_prediction}'.")
                except Exception as e:
                    print(f"Lỗi khi chạy Cycle Consistency: {e}. Buộc ném sang Tier 3 để an toàn!")
                    return None
        else:
            if not prediction:
                return None
    else:
        # SMILES to IUPAC
        if not prediction or len(prediction) < 2:
            return None
            
    return prediction

def get_from_pubchem(query, direction):
    try:
        if direction == "smiles_to_iupac":
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{requests.utils.quote(query)}/property/IUPACName/JSON"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                return resp.json()['PropertyTable']['Properties'][0]['IUPACName']
        else:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{requests.utils.quote(query)}/property/IsomericSMILES/JSON"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                props = resp.json()['PropertyTable']['Properties'][0]
                return props.get('IsomericSMILES', props.get('SMILES'))
    except Exception as e:
        print(f"PubChem API Error: {e}")
    return None

def translate_molecule(query, direction):
    if direction == "iupac_to_smiles":
        normalized_query = query.strip().lower()
    else:
        normalized_query = query.strip()
        
    result = get_from_cache(normalized_query, direction)
    if result:
        return result, "Tier 1 (Local Cache)"
    
    result = translate_via_local_model(normalized_query, direction)
    if result:
        update_cache(normalized_query, direction, result)
        return result, "Tier 2 (Local ML Model)"
        
    result = get_from_pubchem(normalized_query, direction)
    if result:
        update_cache(normalized_query, direction, result)
        return result, "Tier 3 (PubChem API)"
        
    return None, "Not Found"

def render_molecule_image(smiles):
    if RDKIT_AVAILABLE:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                img = Draw.MolToImage(mol, size=(300, 300))
                st.image(img, caption="Generated by RDKit")
                return
        except Exception:
            pass
            
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{requests.utils.quote(smiles)}/PNG"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            img = Image.open(BytesIO(resp.content))
            st.image(img, caption="Fetched from PubChem")
        else:
            st.warning("Could not render molecule image from PubChem.")
    except Exception as e:
        st.warning("Error rendering molecule image.")
