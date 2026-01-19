from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
import ast
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

app = Flask(__name__)
CORS(app)

model = SentenceTransformer('BAAI/bge-small-en-v1.5')
openai_client = OpenAI(api_key="sk-AfX0rModIomRG94i_uu6EmTxDke-lTJjrd-ypIa0mVT3BlbkFJGWF5LOwnni53vOR9uvHv2FU_H98V4-DHOqp87EP1gA")
df = pd.read_pickle("embeddings.pkl")

def create_embedding(text):
    return model.encode([text])[0].tolist()

def detect_language(text):
    hindi_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    total_chars = len([c for c in text if c.isalpha()])
    if total_chars == 0:
        return 'english'
    return 'hindi' if hindi_chars / total_chars > 0.3 else 'english'

def get_rag_response(incoming_query):
    query_lang = detect_language(incoming_query)
    filtered_df = df[df['language'] == query_lang].reset_index(drop=True)
    
    if filtered_df.empty:
        return "No chunks found for this language."
    
    query_embedding = create_embedding(incoming_query)
    similarities = cosine_similarity(
        np.vstack(filtered_df['embedding'].values),
        [query_embedding]
    ).flatten()
    
    top_k = 3
    top_idx = similarities.argsort()[::-1][:top_k]
    results = filtered_df.loc[top_idx]
    
    chunks_context = ""
    for i, (idx, row) in enumerate(results.iterrows(), 1):
        video_num = row.get('number', 'N/A')
        title = row.get('title', 'Unknown')
        timestamp = row['timestamp']
        text = row['text']
        
        chunks_context += f"""
Chunk {i}:
- Video Number: {video_num}
- Title: {title}
- Timestamp: {timestamp}
- Content: {text}
"""
    
    prompt = f"""I am teaching web development in my Sigma web development course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:
{chunks_context}
User asked this question: "{incoming_query}"
User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course."""
    
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful teaching assistant for the Sigma Web Development course. Answer based ONLY on the provided chunks and user's specific question. IMPORTANT: Always convert timestamps to minutes:seconds format ( seconds = 1:25, seconds = 14:09). Never mention raw seconds, always use MM:SS format."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500,
        presence_penalty=0.6,
        frequency_penalty=0.3
    )
    
    return response.choices[0].message.content

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        answer = get_rag_response(user_message)
        
        return jsonify({
            'reply': answer
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
    
# df = pd.read_csv("embeddings.csv")
# def fix_embedding(x):
#     if isinstance(x, str):
#         return ast.literal_eval(x)
#     return x

# df['embedding'] = df['embedding'].apply(fix_embedding)
# df['embedding'] = df['embedding'].apply(lambda x: np.array(x, dtype=np.float32))
# joblib.dump(df, "embeddings.joblib")
# print("✅ Embeddings successfully stored in joblib")

# df_test = joblib.load("embeddings.joblib")

# print(type(df_test['embedding'][0]))
# print(df_test['embedding'][0].shape)