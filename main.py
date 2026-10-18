from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import datetime
# import chronodb  # (ဥပမာ ChronoDB library သုံးရန် သို့မဟုတ် Custom DB connection)

app = FastAPI()

# Enable CORS for Google Sites embedding
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    email: str
    message: str

class LogRequest(BaseModel):
    email: str
    question: str
    answer: str
    source: str

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    # 1. Save user email and question to ChronoDB (Time-series / Database logic)
    # chronodb.insert({"email": req.email, "question": req.message, "timestamp": datetime.datetime.now()})
    
    # 2. Call AI API (OpenAI / Gemini / Local Model) since it's missing from JSON
    ai_response = f"AI Generated Answer for: '{req.message}'"
    
    return {"reply": ai_response}

@app.post("/log")
def log_endpoint(req: LogRequest):
    # Save predefined Q&A logs to ChronoDB
    # chronodb.insert({"email": req.email, "question": req.question, "answer": req.answer, "source": req.source, "timestamp": datetime.datetime.now()})
    return {"status": "logged successfully"}
