import streamlit as st
import numpy as np
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
api_key = os.getenv("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")

if not api_key:
    raise ValueError("API_KEY environment variable is missing!")

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Sigma RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Sigma Web Dev RAG Chatbot")
st.write("Ask questions related to **Sigma Web Development Course**")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("BAAI/bge-small-en-v1.5")

model = load_model()

# ---------------- GROQ CLIENT ----------------
groq_client = Groq(
    api_key="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"  
)

# ---------------- LOAD EMBEDDINGS ----------------
@st.cache_data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), "embeddings.pkl")
    return pd.read_pickle(file_path)

df = load_data()

# ---------------- UTILS ----------------
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
                "content": (
                    "You are a helpful teaching assistant. "
                    "Always convert timestamps to MM:SS format."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        max_tokens=500
    )

    return completion.choices[0].message.content.strip()

# ---------------- UI ----------------
user_question = st.text_input("Ask your question:")

if st.button("Ask"):
    if not user_question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                answer = get_rag_response(user_question)
                st.success("Answer:")
                st.write(answer)
            except Exception as e:
                st.error("Something went wrong")
                st.code(str(e))
