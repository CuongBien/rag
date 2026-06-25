import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load env
load_dotenv(override=True)
os.environ["PHOENIX_ENABLED"] = "0"

# Import NER logic
from ner_graph.router_agent import get_router_query_engine
from services.translation_service import translate_molecule

app = FastAPI(title="Cheminformatics API")

# Allow Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent
agent = None

@app.on_event("startup")
def startup_event():
    global agent
    print("[server] Starting Agent (Loading Models & Graph)...")
    project_root = os.path.dirname(os.path.abspath(__file__))
    agent = get_router_query_engine(project_root)
    print("[server] Agent is ready!")

class ChatRequest(BaseModel):
    query: str

class TranslateRequest(BaseModel):
    query: str
    direction: str  # iupac_to_smiles or smiles_to_iupac

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent chưa sẵn sàng")
    
    try:
        from llama_index.core import QueryBundle
        bundle = QueryBundle(req.query)
        response = agent.query(bundle)
        
        graph_data = None
        if hasattr(response, "metadata") and response.metadata and "graph_data" in response.metadata:
            graph_data = response.metadata["graph_data"]
            
        return {"answer": str(response), "graph_data": graph_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/translate")
def translate_endpoint(req: TranslateRequest):
    try:
        prediction, source = translate_molecule(req.query, req.direction)
        if prediction is None:
            return {"prediction": "None", "source": "Not Found / Rejected by Zero-Hallucination"}
        return {"prediction": prediction, "source": source}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class SandboxRequest(BaseModel):
    drugs: list[str]

@app.get("/api/drugs")
def get_drugs():
    import glob
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_vector")
    if not os.path.exists(data_dir):
        return {"drugs": []}
    files = glob.glob(os.path.join(data_dir, "*.txt"))
    drugs = [os.path.basename(f).replace(".txt", "") for f in files]
    return {"drugs": sorted(drugs)}

@app.post("/api/sandbox/relations")
def get_sandbox_relations(req: SandboxRequest):
    try:
        from ner_graph.pg_query import get_sandbox_graph_data
        graph_data = get_sandbox_graph_data(req.drugs)
        return {"graph_data": graph_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class PolypharmacyRequest(BaseModel):
    prescription: str

@app.post("/api/polypharmacy")
def analyze_polypharmacy(req: PolypharmacyRequest):
    try:
        raw_list = req.prescription.split(",")
        # QUAN TRỌNG: Phải .lower() các phần tử vì query graphDB dùng toLower(a.id)
        drugs = [d.strip().lower() for d in raw_list if d.strip()]
        
        from ner_graph.pg_query import get_sandbox_graph_data
        graph_data = get_sandbox_graph_data(drugs)
        
        if not graph_data["links"]:
            return {
                "graph_data": graph_data,
                "analysis": "✅ **An toàn!** Không phát hiện tương tác nguy hiểm nào giữa các thuốc trong toa này theo cơ sở dữ liệu hiện tại."
            }
            
        interactions = []
        for link in graph_data["links"]:
            interactions.append(f"- {link['source'].capitalize()} và {link['target'].capitalize()} (Quan hệ: {link['label']})")
        
        interactions_text = "\n".join(interactions)
        
        prompt = f"""Bạn là một Dược sĩ Lâm sàng xuất sắc. Bệnh nhân được kê một toa thuốc Đa Bạo Bệnh (Polypharmacy).
Hệ thống GraphDB vừa phát hiện các tương tác thuốc sau trong toa:
{interactions_text}

Hãy viết một báo cáo ngắn gọn (bằng tiếng Việt) cảnh báo an toàn cho toa thuốc này.
1. Nêu rõ rủi ro y khoa của các tương tác trên.
2. Gợi ý đổi thuốc an toàn hơn nếu cần thiết.
Trình bày chuyên nghiệp, dùng markdown và icon cảnh báo (🚨, ⚠️) phù hợp để làm slide báo cáo."""

        # SỬ DỤNG HỆ THỐNG LLM ĐÃ CẤU HÌNH CỦA DỰ ÁN
        import os
        from ner_graph.config import load_config
        from ner_graph.llm_client import create_llm
        config = load_config(os.getcwd())
        llm = create_llm(config)
        
        response = llm.complete(prompt)
        
        return {
            "graph_data": graph_data,
            "analysis": str(response)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
