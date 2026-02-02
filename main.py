import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from pydantic import BaseModel
import uvicorn

# ---------------- CONFIG ----------------
app = FastAPI()

# API Key from environment variable
api_key = os.getenv("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") 
if not api_key:
    # Fallback for testing if allowed, but normally we'd want this in env
    api_key = "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"

groq_client = Groq(api_key=api_key)

# ---------------- LOAD MODEL & DATA ----------------
print("Loading Model...")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

print("Loading Data...")
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), "embeddings.pkl")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Embeddings file not found at {file_path}")
    return pd.read_pickle(file_path)

df = load_data()

# ---------------- RAG LOGIC ----------------
def create_embedding(text):
    return model.encode(text).tolist()

def detect_language(text):
    hindi_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    total_chars = len([c for c in text if c.isalpha()])
    if total_chars == 0:
        return "english"
    return "hindi" if hindi_chars / total_chars > 0.3 else "english"

def get_rag_response(incoming_query):
    query_lang = detect_language(incoming_query)
    filtered_df = df[df["language"] == query_lang].reset_index(drop=True)

    if filtered_df.empty:
        return "No relevant content found for this language."

    query_embedding = create_embedding(incoming_query)
    similarities = cosine_similarity(
        np.vstack(filtered_df["embedding"].values),
        [query_embedding]
    ).flatten()

    top_k = 3
    top_idx = similarities.argsort()[::-1][:top_k]
    results = filtered_df.loc[top_idx]

    chunks_context = ""
    for i, (_, row) in enumerate(results.iterrows(), 1):
        chunks_context += f"""
Chunk {i}:
- Video Number: {row.get('number', 'N/A')}
- Title: {row.get('title', 'Unknown')}
- Timestamp: {row['timestamp']}
- Content: {row['text']}
"""

    prompt = f"""
I am teaching web development in my Sigma Web Development course.

Below are video subtitle chunks:
{chunks_context}

User question:
"{incoming_query}"

Instructions:
- Answer ONLY using the provided chunks
- Mention exact video number & timestamp (MM:SS)
- Be teacher-like
- If unrelated, politely refuse
"""

    completion = groq_client.chat.completions.create(
        model="moonshotai/kimi-k2-instruct-0905",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful teaching assistant. Always convert timestamps to MM:SS format."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        max_tokens=500
    )

    return completion.choices[0].message.content.strip()

# ---------------- MODELS ----------------
class ChatRequest(BaseModel):
    message: str

# ---------------- ROUTES ----------------
@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_path = os.path.join(os.path.dirname(__file__), "ragebase-ui", "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Empty message")
        
        reply = get_rag_response(request.message)
        return {"reply": reply}
    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(status_code=500, content={"reply": "I'm sorry, I'm having trouble connecting to the brain. Please try again later."})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
