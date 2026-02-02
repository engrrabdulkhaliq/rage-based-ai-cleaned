import os
import sys
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from pydantic import BaseModel
import uvicorn
import logging
import warnings

# ============ ENVIRONMENT DETECTION ============
IS_RAILWAY = os.getenv('RAILWAY_ENVIRONMENT') is not None
PORT = int(os.environ.get("PORT", 8080))  # Railway uses PORT env var

print(f"🌍 Environment: {'Railway' if IS_RAILWAY else 'Local'}")
print(f"🔌 Port: {PORT}")

# ============ LOGGING ============
logging.basicConfig(
    level=logging.INFO if not IS_RAILWAY else logging.ERROR,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress unnecessary logs
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ============ FASTAPI APP ============
app = FastAPI(title="Sigma RAG API", docs_url="/docs" if not IS_RAILWAY else None)

# ============ API KEY ============
api_key = os.getenv("GROQ_API_KEY") or os.getenv("gsk")
if not api_key:
    logger.error("❌ GROQ_API_KEY not set!")
    sys.exit(1)

groq_client = Groq(api_key=api_key)

# ============ LOAD MODEL ============
logger.info("📦 Loading model...")
try:
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device='cpu')
    logger.info("✅ Model loaded")
except Exception as e:
    logger.error(f"❌ Model load failed: {e}")
    sys.exit(1)

# ============ LOAD EMBEDDINGS ============
logger.info("📦 Loading embeddings...")
df = None

def load_data():
    paths = [
        "embeddings.pkl",
        "/app/embeddings.pkl",
        os.path.join(os.path.dirname(__file__), "embeddings.pkl")
    ]
    
    for path in paths:
        if os.path.exists(path):
            logger.info(f"✅ Found at: {path}")
            return pd.read_pickle(path)
    
    raise FileNotFoundError("❌ embeddings.pkl not found")

try:
    df = load_data()
    logger.info(f"✅ Loaded {len(df)} embeddings")
except Exception as e:
    logger.error(f"❌ Failed to load embeddings: {e}")
    # Don't exit - let health check fail instead

# ============ RAG FUNCTIONS ============
def create_embedding(text):
    return model.encode(text, show_progress_bar=False, convert_to_numpy=True).tolist()

def detect_language(text):
    hindi_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    total_chars = len([c for c in text if c.isalpha()])
    return "hindi" if total_chars > 0 and hindi_chars / total_chars > 0.3 else "english"

def get_rag_response(query):
    if df is None:
        return "System not ready"
    
    try:
        lang = detect_language(query)
        filtered = df[df["language"] == lang].reset_index(drop=True)
        
        if filtered.empty:
            return "No content found"
        
        query_emb = create_embedding(query)
        sims = cosine_similarity(
            np.vstack(filtered["embedding"].values),
            [query_emb]
        ).flatten()
        
        top_idx = sims.argsort()[::-1][:3]
        results = filtered.loc[top_idx]
        
        context = ""
        for i, (_, row) in enumerate(results.iterrows(), 1):
            context += f"\nChunk {i}:\n- Video: {row.get('number', 'N/A')}\n- Title: {row.get('title', 'Unknown')}\n- Timestamp: {row['timestamp']}\n- Content: {row['text']}\n"
        
        prompt = f"""I am teaching web development in my Sigma Web Development course.

Below are video subtitle chunks:
{context}

User question: "{query}"

Instructions:
- Answer ONLY using the provided chunks
- Mention exact video number & timestamp (MM:SS)
- Be teacher-like
- If unrelated, politely refuse"""
        
        completion = groq_client.chat.completions.create(
            model="moonshotai/kimi-k2-instruct-0905",
            messages=[
                {"role": "system", "content": "You are a helpful teaching assistant. Always convert timestamps to MM:SS format."},
                {"role": "user", "content": prompt}
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
    """Health check endpoint"""
    return {
        "status": "running",
        "model": "loaded" if model else "failed",
        "embeddings": "loaded" if df is not None else "failed",
        "count": len(df) if df is not None else 0
    }

@app.get("/health")
async def health():
    """Railway health check"""
    if df is None:
        raise HTTPException(status_code=503, detail="Embeddings not loaded")
    return {"status": "healthy"}

@app.post("/chat")
async def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")
    
    if df is None:
        raise HTTPException(status_code=503, detail="System not ready")
    
    try:
        reply = get_rag_response(request.message)
        return {"reply": reply}
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Internal error")

@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(status_code=204, content={})

# ============ STARTUP ============
@app.on_event("startup")
async def startup():
    logger.info("🚀 App started successfully")

# ============ RUN (Critical for Railway) ============
if __name__ == "__main__":
    logger.info(f"🚀 Starting server on 0.0.0.0:{PORT}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",  # MUST be 0.0.0.0 for Railway
        port=PORT,       # MUST use Railway's PORT
        log_level="error" if IS_RAILWAY else "info"
    )
