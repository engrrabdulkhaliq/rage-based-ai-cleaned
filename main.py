import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq

app = FastAPI()

# Railway se automatically key milegi
client = Groq(api_key=os.environ.get("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"))

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def home():
    return {"status": "RageBase API Running"}

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": request.message
                }
            ],
            model="llama-3.3-70b-versatile",
        )

        return {
            "response": response.choices[0].message.content
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
