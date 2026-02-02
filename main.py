import os
import sys
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from pydantic import BaseModel
import uvicorn
import logging
import warnings

# ============ SETUP ============
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Environment
IS_RAILWAY = os.getenv('RAILWAY_ENVIRONMENT') is not None
PORT = int(os.environ.get("PORT", 8080))

# ============ APP ============
app = FastAPI(title="Sigma RAG API")

# ============ GLOBALS ============
model = None
df = None
groq_client = None

# ============ STARTUP ============
@app.on_event("startup")
async def startup():
    global model, df, groq_client
    
    logger.info("🚀 Starting application...")
    
    # Load API key
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("gsk")
    if not api_key:
        logger.error("❌ GROQ_API_KEY not set")
        sys.exit(1)
    
    groq_client = Groq(api_key=api_key)
    logger.info("✅ Groq client initialized")
    
    # Load model
    try:
        logger.info("📦 Loading model...")
        model = SentenceTransformer("BAAI/bge-small-en-v1.5", device='cpu')
        logger.info("✅ Model loaded")
    except Exception as e:
        logger.error(f"❌ Model failed: {e}")
        sys.exit(1)
    
    # Load embeddings
    try:
        logger.info("📦 Loading embeddings...")
        paths = ["embeddings.pkl", "/app/embeddings.pkl"]
        
        for path in paths:
            if os.path.exists(path):
                df = pd.read_pickle(path)
                logger.info(f"✅ Loaded {len(df)} embeddings from {path}")
                break
        
        if df is None:
            logger.error("❌ embeddings.pkl not found")
    except Exception as e:
        logger.error(f"❌ Embeddings failed: {e}")

# ============ FUNCTIONS ============
def create_embedding(text):
    return model.encode(text, show_progress_bar=False, convert_to_numpy=True).tolist()

def detect_language(text):
    hindi_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    total_chars = len([c for c in text if c.isalpha()])
    return "hindi" if total_chars > 0 and hindi_chars / total_chars > 0.3 else "english"

def get_rag_response(query):
    if df is None or model is None:
        return "System not ready"
    
    try:
        lang = detect_language(query)
        filtered = df[df["language"] == lang].reset_index(drop=True)
        
        if filtered.empty:
            return "No content found for this language"
        
        query_emb = create_embedding(query)
        sims = cosine_similarity(np.vstack(filtered["embedding"].values), [query_emb]).flatten()
        
        top_idx = sims.argsort()[::-1][:3]
        results = filtered.loc[top_idx]
        
        context = ""
        for i, (_, row) in enumerate(results.iterrows(), 1):
            context += f"\nChunk {i}:\n- Video: {row.get('number', 'N/A')}\n- Title: {row.get('title', 'Unknown')}\n- Time: {row['timestamp']}\n- Text: {row['text']}\n"
        
        completion = groq_client.chat.completions.create(
            model="moonshotai/kimi-k2-instruct-0905",
            messages=[
                {"role": "system", "content": "You are a helpful teaching assistant for Sigma Web Dev course. Convert timestamps to MM:SS."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer using only the context above."}
            ],
            temperature=0.4,
            max_tokens=500
        )
        
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"RAG error: {e}")
        return "Error processing request"

# ============ ROUTES ============
class ChatRequest(BaseModel):
    message: str

@app.get("/")
async def root():
    return {
        "status": "running",
        "model": "loaded" if model else "not loaded",
        "embeddings": "loaded" if df is not None else "not loaded",
        "count": len(df) if df is not None else 0
    }

@app.get("/health")
async def health():
    if model is None or df is None:
        raise HTTPException(status_code=503, detail="Not ready")
    return {"status": "healthy"}

@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")
    if df is None:
        raise HTTPException(status_code=503, detail="Not ready")
    
    reply = get_rag_response(req.message)
    return {"reply": reply}

@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(status_code=204, content={})

# ============ RUN ============
if __name__ == "__main__":
    logger.info(f"Starting on 0.0.0.0:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")