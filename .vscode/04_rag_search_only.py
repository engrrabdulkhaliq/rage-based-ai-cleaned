from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

app = Flask(__name__)
CORS(app)

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

groq_client = Groq(
    api_key="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"
)

file_path = os.path.join(os.path.dirname(__file__), "embeddings.pkl")
df = pd.read_pickle(file_path)

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
    
    completion = groq_client.chat.completions.create(
        model="moonshotai/kimi-k2-instruct-0905",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful teaching assistant for the Sigma Web Development course. "
                    "Always convert timestamps to MM:SS format. "
                    "Never mention raw seconds."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        max_tokens=500
    )
    
    return completion.choices[0].message.content.strip()

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "message": "RAG-based AI Chatbot API",
        "endpoints": {
            "chat": "/chat (POST)"
        }
    })

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
    app.run(debug=True, host="0.0.0.0", port=5000)
