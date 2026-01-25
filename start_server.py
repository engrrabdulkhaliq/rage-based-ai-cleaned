#!/usr/bin/env python
"""
Lightweight RAG server wrapper that starts Flask ASAP
"""
import sys
import os

# Add vscode dir to path for embeddings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.vscode'))

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Lazy load heavy dependencies
_resources = {
    'model': None,
    'gemini_client': None,
    'df': None
}

def load_resources():
    if _resources['model'] is None:
        print("⏳ Loading SentenceTransformer model...")
        from sentence_transformers import SentenceTransformer
        _resources['model'] = SentenceTransformer("BAAI/bge-small-en-v1.5")
    
    if _resources['gemini_client'] is None:
        print("⏳ Initializing Gemini client...")
        from google import genai
        _resources['gemini_client'] = genai.Client(api_key="AIzaSyDBTB1DDHMvT2ZlWaGlsPJlsaZlsGXIulk")
    
    if _resources['df'] is None:
        print("⏳ Loading embeddings...")
        import pandas as pd
        file_path = os.path.join(os.path.dirname(__file__), ".vscode", "embeddings.pkl")
        if os.path.exists(file_path):
            _resources['df'] = pd.read_pickle(file_path)
            print(f"✅ Loaded {len(_resources['df'])} embeddings")
        else:
            import pandas as pd
            _resources['df'] = pd.DataFrame()
            print(f"⚠️  No embeddings found")

@app.before_request
def init_resources():
    if _resources['model'] is None:
        load_resources()

def create_embedding(text):
    return _resources['model'].encode(text).tolist()

def detect_language(text):
    hindi_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    total_chars = len([c for c in text if c.isalpha()])
    if total_chars == 0:
        return "english"
    return "hindi" if hindi_chars / total_chars > 0.3 else "english"

def get_rag_response(incoming_query):
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from google.genai import types
    
    df = _resources['df']
    model = _resources['model']
    gemini_client = _resources['gemini_client']
    
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
Below are video subtitle chunks with video number, title, and timestamp:
{chunks_context}

User question:
"{incoming_query}"

Instructions:
- Answer ONLY using the provided chunks
- Tell clearly WHICH video and WHICH timestamp (MM:SS format)
- Be human and teacher-like
- If question is unrelated, politely say you can only answer course-related questions
"""
    
    system_instruction = (
        "You are a helpful teaching assistant for the Sigma Web Development course. "
        "Always convert timestamps to MM:SS format. "
        "Never mention raw seconds."
    )
    
    full_prompt = f"{system_instruction}\n\n{prompt}"
    
    response = gemini_client.models.generate_content(
        model='gemini-1.5-flash-8b',
        contents=full_prompt,
        config=types.GenerateContentConfig(
            temperature=0.4,
            max_output_tokens=500,
        )
    )
    
    return response.text.strip()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return "", 200
    
    try:
        data = request.get_json()
        user_message = data.get("message", "")
        print("Received message:", user_message)
        
        answer = get_rag_response(user_message)
        return jsonify({"reply": answer})
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 RAG AI Teaching Assistant Server")
    print("="*60)
    print("🔗 Server: http://localhost:5000")
    print("📍 Health Check: http://localhost:5000/health")
    print("💬 Chat API: POST http://localhost:5000/chat")
    print("="*60 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000, use_reloader=False, threaded=True)
