import os
import sys
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from pydantic import BaseModel
import uvicorn
import logging
import warnings

# ============ LOGGING CONFIGURATION ============
IS_RAILWAY = os.getenv('RAILWAY_ENVIRONMENT') is not None
IS_PRODUCTION = os.getenv('ENVIRONMENT') == 'production'

# Configure logging based on environment
if IS_PRODUCTION or IS_RAILWAY:
    # Production: Only show errors
    logging.basicConfig(
        level=logging.ERROR,
        format='%(levelname)s: %(message)s',
        stream=sys.stdout
    )
else:
    # Local: Show all info
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )

logger = logging.getLogger(__name__)

# Suppress third-party library logs
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('uvicorn').setLevel(logging.WARNING)
logging.getLogger('uvicorn.access').setLevel(logging.WARNING)

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ============ CONFIG ============
app = FastAPI(title="Sigma RAG API")

# Mount static files if folder exists
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# ============ API KEY ============
api_key = os.getenv("GROQ_API_KEY") or os.getenv("gsk")

if not api_key:
    api_key = "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"
    if not IS_PRODUCTION:
        logger.warning("Using hardcoded API key. Set GROQ_API_KEY environment variable.")

groq_client = Groq(api_key=api_key)

# ============ LOAD MODEL ============
if not IS_PRODUCTION:
    logger.info("Loading Model...")

model = SentenceTransformer("BAAI/bge-small-en-v1.5", device='cpu')

if not IS_PRODUCTION:
    logger.info("Model Loaded!")

# ============ LOAD DATA ============
def load_data():
    if not IS_PRODUCTION:
        logger.info("Loading Embeddings...")
    
    possible_paths = [
        "embeddings.pkl",
        os.path.join(os.path.dirname(__file__), "embeddings.pkl"),
        "/app/embeddings.pkl",
        "./embeddings.pkl"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            if not IS_PRODUCTION:
                logger.info(f"Found embeddings at: {path}")
            df = pd.read_pickle(path)
            if not IS_PRODUCTION:
                logger.info(f"Loaded {len(df)} embeddings")
            return df
    
    logger.error("embeddings.pkl not found!")
    raise FileNotFoundError("Embeddings file not found")

try:
    df = load_data()
except Exception as e:
    logger.error(f"Error loading data: {e}")
    df = None

# ============ RAG LOGIC ============
def create_embedding(text):
    return model.encode(
        text,
        show_progress_bar=False,
        convert_to_numpy=True
    ).tolist()

def detect_language(text):
    hindi_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    total_chars = len([c for c in text if c.isalpha()])
    if total_chars == 0:
        return "english"
    return "hindi" if hindi_chars / total_chars > 0.3 else "english"

def get_rag_response(incoming_query):
    if df is None:
        return "System not ready. Embeddings not loaded."
    
    try:
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
    
    except Exception as e:
        logger.error(f"RAG Error: {e}")
        return "Error processing request. Please try again."

# ============ MODELS ============
class ChatRequest(BaseModel):
    message: str

# ============ ROUTES ============
@app.get("/favicon.ico")
async def favicon():
    favicon_path = os.path.join("static", "favicon.ico")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    return JSONResponse(status_code=204, content={})

@app.get("/")
async def root():
    return {
        "status": "running",
        "environment": "railway" if IS_RAILWAY else "local",
        "model_loaded": model is not None,
        "embeddings_loaded": df is not None,
        "embeddings_count": len(df) if df is not None else 0
    }

@app.get("/ui", response_class=HTMLResponse)
async def get_index():
    try:
        index_path = os.path.join(os.path.dirname(__file__), "ragebase-ui", "index.html")
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            return HTMLResponse(
                content="<h1>UI Not Found</h1><p>Place your HTML in ragebase-ui/index.html</p>",
                status_code=404
            )
    except Exception as e:
        return HTMLResponse(
            content=f"<h1>Error Loading UI</h1><p>{str(e)}</p>",
            status_code=500
        )

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Empty message")
        
        if df is None:
            raise HTTPException(status_code=503, detail="System not ready")
        
        reply = get_rag_response(request.message)
        return {"reply": reply}
    
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Chat Error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "reply": "I'm sorry, I'm having trouble. Please try again later.",
                "error": str(e) if not IS_PRODUCTION else None
            }
        )

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if df is not None else "degraded",
        "model": "loaded" if model is not None else "not loaded",
        "embeddings": "loaded" if df is not None else "not loaded"
    }

# ============ STARTUP & SHUTDOWN (Silent in Production) ============
@app.on_event("startup")
async def startup_event():
    if not IS_PRODUCTION:
        logger.info("="*50)
        logger.info("Sigma RAG API Starting...")
        logger.info(f"Environment: {'Railway' if IS_RAILWAY else 'Local'}")
        logger.info(f"Model: {'Loaded' if model else 'Failed'}")
        logger.info(f"Embeddings: {'Loaded' if df is not None else 'Failed'}")
        logger.info("="*50)

@app.on_event("shutdown")
async def shutdown_event():
    if not IS_PRODUCTION:
        logger.info("Shutting down gracefully...")

# ============ RUN SERVER ============
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    
    if not IS_PRODUCTION:
        logger.info(f"Starting server on port {port}...")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="error" if IS_PRODUCTION else "info",
        access_log=False  # Access logs completely off
    )