import os
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import chromadb

app = FastAPI()

# CORS Middleware (Google Sites / Render မှာ ချိတ်ဆက်နိုင်ရန်)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ChromaDB Persistent Storage
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="throne_chat_logs")

print("Loading AI Model...")
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Knowledge Base ဖိုင်ကို ဖတ်ယူခြင်း
with open("throne_data.json", "r", encoding="utf-8") as f:
    kb_raw = json.load(f)

class ChatRequest(BaseModel):
    email: str
    message: str
    language: str = "mm"

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    user_query = req.message.strip()
    lang = req.language if req.language in ["mm", "en", "th"] else "mm"
    
    # ChromaDB တွင် Log သိမ်းဆည်းခြင်း
    try:
        log_id = f"chat_{len(collection.get()['ids']) + 1}"
        collection.add(
            documents=[user_query],
            metadatas=[{"email": req.email, "language": lang}],
            ids=[log_id]
        )
    except Exception as e:
        print(f"ChromaDB Error: {e}")

    query_lower = user_query.lower()
    matched_item = None

    # 1. Exact Match စစ်ဆေးခြင်း
    for cat in kb_raw.get("throne_categories", []):
        name_mm = cat.get("name", {}).get("mm", "").lower()
        name_en = cat.get("name", {}).get("en", "").lower()
        name_th = cat.get("name", {}).get("th", "").lower()
        cat_id = cat.get("id", "").lower()
        category = cat.get("category", "").lower()
        
        if (query_lower in name_mm or query_lower in name_en or query_lower in name_th or 
            query_lower == cat_id or query_lower == category):
            matched_item = cat
            break

    if matched_item:
        summary = matched_item.get("summary", {}).get(lang, "")
        detailed = matched_item.get("detailed_info", {}).get(lang, "")
        detailed_str = "\n".join(detailed) if isinstance(detailed, list) else detailed
            
        return {
            "title": matched_item.get("name", {}).get(lang, ""),
            "summary": summary,
            "detailed_info": detailed_str,
            "type": "content"
        }

    # 2. Semantic Search စစ်ဆေးခြင်း (Threshold = 0.60)
    all_texts = []
    for cat in kb_raw.get("throne_categories", []):
        if lang in cat.get("name", {}): 
            all_texts.append({"cat": cat, "text": cat["name"][lang]})

    if all_texts:
        texts_only = [item["text"] for item in all_texts]
        embeddings = model.encode(texts_only, convert_to_tensor=True)
        user_embedding = model.encode(user_query, convert_to_tensor=True)
        cos_scores = util.cos_sim(user_embedding, embeddings)[0]
        best_score = float(cos_scores.max())
        
        if best_score > 0.60:
            best_idx = int(cos_scores.argmax())
            matched_item = all_texts[best_idx]["cat"]
            summary = matched_item.get("summary", {}).get(lang, "")
            detailed = matched_item.get("detailed_info", {}).get(lang, "")
            detailed_str = "\n".join(detailed) if isinstance(detailed, list) else detailed
                
            return {
                "title": matched_item.get("name", {}).get(lang, ""),
                "summary": summary,
                "detailed_info": detailed_str,
                "type": "content"
            }

    # 3. Out-of-domain Rejection
    if lang == "th":
        err_msg = "กรุณาถามเกี่ยวกับพระราชบัลลังก์เท่านั้นครับ/ค่ะ"
    elif lang == "mm":
        err_msg = "ကျေးဇူးပြု၍ သီဟာသနပလ္လင်တော်ဆိုင်ရာ အချက်အလက်များကိုသာ မေးမြန်းပေးပါရှင်။"
    else:
        err_msg = "Please ask about the Royal Lion Throne only."

    return {"reply": err_msg, "type": "error"}

@app.get("/")
def home():
    return {"message": "Royal Lion Throne Chatbot API is running on Render!"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
