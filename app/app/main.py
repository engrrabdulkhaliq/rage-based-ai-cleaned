from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def home():
    return {"status": "Server running"}

@app.post("/api/chat")
def chat(request: ChatRequest):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="API key missing")
    return {"response": f"Got: {request.message}"}
