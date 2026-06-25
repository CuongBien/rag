import wikipedia
import os
import time

wikipedia.set_lang("en") 

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data_vector")

# Danh sách 54 Hero Compounds
compounds = [
    # Nhóm Giảm đau / Kháng viêm
    "Aspirin", "Paracetamol", "Ibuprofen", "Naproxen", "Diclofenac", 
    "Celecoxib", "Tramadol", "Morphine", "Fentanyl",
    
    # Nhóm Tim mạch / Đông máu
    "Warfarin", "Clopidogrel", "Atorvastatin", "Simvastatin", "Amlodipine", 
    "Lisinopril", "Losartan", "Metoprolol", "Digoxin", "Rivaroxaban",
    
    # Nhóm Kháng sinh / Kháng nấm / Kháng virus
    "Amoxicillin", "Ciprofloxacin", "Azithromycin", "Doxycycline", 
    "Cefalexin", "Metronidazole", "Fluconazole", "Aciclovir",
    
    # Nhóm Thần kinh / Tâm thần
    "Caffeine", "Diazepam", "Fluoxetine", "Sertraline", "Escitalopram", 
    "Alprazolam", "Zolpidem", "Haloperidol", "Lithium (medication)", 
    "Amitriptyline", "Gabapentin",
    
    # Nhóm Tiêu hóa
    "Omeprazole", "Pantoprazole", "Esomeprazole", "Ranitidine", "Loperamide",
    
    # Nhóm Nội tiết / Tiểu đường
    "Metformin", "Glipizide", "Insulin glargine", "Levothyroxine",
    
    # Nhóm Hô hấp / Dị ứng
    "Salbutamol", "Fluticasone", "Cetirizine", "Loratadine", "Montelukast",
    
    # Nhóm Khác
    "Sildenafil", "Methotrexate"
]

os.makedirs(DATA_DIR, exist_ok=True)

print(f"🚀 ĐANG TẢI DỮ LIỆU TỪ WIKIPEDIA CHO {len(compounds)} CHẤT...")
for compound in compounds:
    file_name = f"{compound.lower().replace(' (medication)', '')}.txt"
    file_path = os.path.join(DATA_DIR, file_name)
    
    # Tính năng "Resume": Bỏ qua nếu file đã tồn tại và có dữ liệu
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        print(f"⏭️ Bỏ qua {compound} (đã tải trước đó)")
        continue
        
    try:
        content = wikipedia.page(compound, auto_suggest=False).content
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ Đã tải xong: {compound}")
        
        # Thêm độ trễ 2 giây để tránh bị Wikipedia chặn (Rate Limit)
        time.sleep(2)
        
    except Exception as e:
        print(f"❌ Lỗi với {compound}: {e}")
        # Chờ lâu hơn một chút nếu gặp lỗi để xả rate limit
        time.sleep(5)

print("🎉 Hoàn thành! Toàn bộ file text đã nằm trong folder 'data_vector'.")
