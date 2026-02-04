import streamlit as st
import numpy as np
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Sigma Web Development Course",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- SESSION STATE ----------------
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'query_counter' not in st.session_state:
    st.session_state.query_counter = 0

# ---------------- LAZY LOADING ----------------
@st.cache_resource
def get_model():
    """Lazy load model only when needed"""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")  # Faster, lighter model

@st.cache_data
def get_data():
    """Load embeddings data"""
    path = os.path.join(os.path.dirname(__file__), "embeddings.pkl")
    if os.path.exists(path):
        return pd.read_pickle(path)
    return pd.DataFrame()

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --bg-primary: #0a0a1a;
        --bg-secondary: #12122a;
        --bg-gradient: linear-gradient(135deg, #0a0a1a 0%, #1a1035 50%, #0d1025 100%);
        --text-primary: #f0f0f5;
        --text-secondary: #a0a0b5;
        --text-muted: #6b6b80;
        --accent-primary: #7c3aed;
        --accent-secondary: #a855f7;
        --accent-glow: rgba(124, 58, 237, 0.3);
        --assistant-bubble: rgba(30, 30, 60, 0.8);
        --user-bubble: rgba(124, 58, 237, 0.15);
        --border-color: rgba(124, 58, 237, 0.2);
        --border-glow: rgba(168, 85, 247, 0.3);
        --glass-bg: rgba(20, 20, 40, 0.6);
        --header-bg: rgba(10, 10, 26, 0.85);
        --input-bg: rgba(20, 20, 45, 0.9);
    }

    /* Reset and Base */
    * {
        box-sizing: border-box;
    }

    .stApp {
        background: var(--bg-gradient);
        background-attachment: fixed;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        min-height: 100vh;
    }

    /* Background Effects */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(ellipse at 20% 20%, rgba(124, 58, 237, 0.15) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 80%, rgba(99, 102, 241, 0.1) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 50%, rgba(168, 85, 247, 0.05) 0%, transparent 70%);
        pointer-events: none;
        z-index: 0;
    }

    [data-testid="stHeader"] {display:none;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}

    /* Header */
    .chat-header {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 68px;
        background: var(--header-bg);
        border-bottom: 1px solid var(--border-color);
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 24px;
        z-index: 1000;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
    }

    .chat-header h1 {
        font-size: 1.2rem;
        font-weight: 600;
        margin: 0;
        color: var(--text-primary);
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, var(--text-primary) 0%, var(--accent-secondary) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* Clear Button */
    .clear-btn {
        background: rgba(124, 58, 237, 0.1);
        border: 1px solid var(--border-color);
        color: var(--text-primary);
        padding: 8px 16px;
        border-radius: 8px;
        font-size: 13px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
        font-family: 'Inter', sans-serif;
    }

    .clear-btn:hover {
        background: rgba(124, 58, 237, 0.2);
        border-color: var(--accent-primary);
    }

    /* Chat Container */
    .chat-wrapper {
        max-width: 820px;
        margin: 0 auto;
        padding: 90px 24px 160px 24px;
        position: relative;
        z-index: 1;
    }

    /* Message Styling */
    .message-container {
        display: flex;
        align-items: flex-start;
        gap: 16px;
        margin-bottom: 32px;
        opacity: 0;
        transform: translateY(20px);
        animation: fadeInUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }

    @keyframes fadeInUp {
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .avatar-wrapper {
        flex-shrink: 0;
        width: 42px;
        height: 42px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        border: 1px solid var(--border-color);
    }

    .assistant-avatar { 
        background: linear-gradient(135deg, #10a37f 0%, #059669 100%);
        color: white; 
    }
    
    .user-avatar { 
        background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
        color: white; 
    }

    .message-body {
        flex-grow: 1;
        max-width: calc(100% - 58px);
    }

    .message-info {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 8px;
    }

    .message-label {
        font-size: 13px;
        font-weight: 600;
        color: var(--text-secondary);
    }

    .message-time {
        font-size: 11px;
        font-weight: 400;
        color: var(--text-muted);
    }

    .message-bubble {
        padding: 16px 20px;
        border-radius: 16px;
        font-size: 15px;
        line-height: 1.7;
        border: 1px solid var(--border-color);
        word-wrap: break-word;
        color: var(--text-primary);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }

    .assistant-bubble { 
        background: var(--assistant-bubble);
        box-shadow: 0 4px 24px rgba(0,0,0,0.2);
    }
    
    .user-bubble { 
        background: var(--user-bubble);
        border-color: rgba(124, 58, 237, 0.3);
        box-shadow: 0 4px 24px rgba(124, 58, 237, 0.1);
    }

    /* Typing Indicator */
    .typing-indicator {
        display: inline-flex;
        gap: 4px;
        padding: 16px 20px;
        border-radius: 16px;
        background: var(--assistant-bubble);
        border: 1px solid var(--border-color);
        box-shadow: 0 4px 24px rgba(0,0,0,0.2);
    }

    .typing-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--text-secondary);
        animation: typing 1.4s infinite;
    }

    .typing-dot:nth-child(2) {
        animation-delay: 0.2s;
    }

    .typing-dot:nth-child(3) {
        animation-delay: 0.4s;
    }

    @keyframes typing {
        0%, 60%, 100% {
            transform: translateY(0);
            opacity: 0.7;
        }
        30% {
            transform: translateY(-10px);
            opacity: 1;
        }
    }

    /* Fixed Input Box */
    .fixed-input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 24px 24px 36px 24px;
        background: linear-gradient(to top, var(--bg-primary) 40%, transparent);
        z-index: 900;
    }

    .input-max-width {
        max-width: 820px;
        margin: 0 auto;
        position: relative;
    }

    /* Streamlit Input Overrides */
    .stTextInput {
        margin-top: 0 !important;
    }
    
    .stTextInput > div > div > input {
        border-radius: 16px !important;
        border: 1px solid var(--border-color) !important;
        padding: 16px 20px !important;
        background: var(--input-bg) !important;
        color: var(--text-primary) !important;
        box-shadow: 0 4px 30px rgba(0,0,0,0.3), 0 0 0 1px rgba(124, 58, 237, 0.1) !important;
        font-size: 15px !important;
        transition: all 0.3s ease !important;
        backdrop-filter: blur(10px) !important;
        caret-color: var(--accent-primary) !important;
    }

    .stTextInput > div > div > input::placeholder {
        color: var(--text-muted) !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: var(--accent-primary) !important;
        box-shadow: 0 4px 30px rgba(0,0,0,0.3), 0 0 0 3px var(--accent-glow) !important;
        outline: none !important;
    }

    /* Form Button */
    .stFormSubmitButton > button {
        background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px var(--accent-glow) !important;
    }

    .stFormSubmitButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 30px var(--accent-glow) !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { 
        background: rgba(124, 58, 237, 0.3); 
        border-radius: 10px; 
    }
    ::-webkit-scrollbar-thumb:hover { 
        background: rgba(124, 58, 237, 0.5); 
    }

    /* Spinner Override */
    .stSpinner > div {
        border-top-color: var(--accent-primary) !important;
    }

    /* Hide Streamlit elements */
    .stButton {
        position: absolute;
        right: 0;
        top: 0;
    }
    
</style>
""", unsafe_allow_html=True)

# ---------------- BACKEND LOGIC ----------------
def get_rag_response(query):
    """Fast RAG response without heavy loading"""
    # Casual Interaction Handling
    greetings = ["hi", "hello", "hey", "hola", "sup", "yo", "morning", "afternoon"]
    if query.lower().strip() in greetings:
        return "Hey! I'm your Sigma Assistant. I'm here to help you master web development. What can I help you with today?"

    try:
        # Lazy load only when needed
        model = get_model()
        df = get_data()
        
        # API Key from environment only
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return "API key not configured. Please set the GROQ_API_KEY environment variable."
        
        client = Groq(api_key=api_key)

        # Simple Language Detection
        hindi_chars = sum(1 for c in query if '\u0900' <= c <= '\u097F')
        total_chars = len([c for c in query if c.isalpha()])
        lang = "hindi" if total_chars > 0 and (hindi_chars / total_chars) > 0.3 else "english"
        
        # RAG Search
        filtered_df = df[df["language"] == lang].reset_index(drop=True) if not df.empty else df
        if filtered_df.empty:
            filtered_df = df.reset_index(drop=True)

        if not filtered_df.empty:
            q_emb = model.encode(query).tolist()
            sims = cosine_similarity(np.vstack(filtered_df["embedding"].values), [q_emb]).flatten()
            top_indices = sims.argsort()[::-1][:3]
            top_results = filtered_df.loc[top_indices]

            context_str = ""
            for _, row in top_results.iterrows():
                context_str += f"Video {row.get('number', 'N/A')} [{row['timestamp']}]: {row['text']}\n"
        else:
            context_str = "No course context available."

        # Conversational Prompt
        system_msg = (
            "You are a friendly and professional AI teaching assistant for the Sigma Web Development course. "
            "Your responses should feel natural, intelligent, and human-like, similar to Claude or ChatGPT. "
            "NEVER mention 'datasets', 'chunks', 'subtitles', or 'internal indexing'. "
            "Use the context provided to answer accurately. Always credit specific video numbers and timestamps (MM:SS) "
            "when providing information from the course. If a question is unrelated to the course, answer politely using "
            "your general knowledge while maintaining your identity as the Sigma assistant."
        )
        
        chat_completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Context:\n{context_str}\n\nUser Question: {query}"}
            ],
            temperature=0.75,
            max_tokens=1000
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"I apologize, but I encountered an error while processing that. Let's try again! (Error: {str(e)})"

# ---------------- UI RENDERING ----------------

# Header
st.markdown("""
<div class="chat-header">
    <h1>Sigma Web Development Course</h1>
    <form method="post">
        <button class="clear-btn" name="clear_chat">Clear</button>
    </form>
</div>
""", unsafe_allow_html=True)

# Clear button logic (NO UI CHANGE)
if "clear_chat" in st.session_state:
    st.session_state.messages = []
    st.session_state.query_counter = 0
    del st.session_state["clear_chat"]
    st.rerun()


# Chat Content Container
st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)

# Initial Welcome Message if empty
if not st.session_state.messages:
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hi there! I'm your AI guide for the Sigma Web Development course. I can help you understand complex concepts, find specific moments in our videos, or just chat about your coding journey. How can I help you today?",
        "time": datetime.now().strftime("%H:%M")
    })

# Render Messages
for msg in st.session_state.messages:
    role = msg["role"]
    is_user = role == "user"
    avatar_class = "user-avatar" if is_user else "assistant-avatar"
    avatar_icon = "👤" if is_user else "🤖"
    label = "You" if is_user else "Assistant"
    bubble_class = "user-bubble" if is_user else "assistant-bubble"
    
    st.markdown(f"""
    <div class="message-container">
        <div class="avatar-wrapper {avatar_class}">{avatar_icon}</div>
        <div class="message-body">
            <div class="message-info">
                <span class="message-label">{label}</span>
                <span class="message-time">{msg['time']}</span>
            </div>
            <div class="message-bubble {bubble_class}">{msg['content']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True) # End chat-wrapper

# Bottom Fixed Input
st.markdown('<div class="fixed-input-container">', unsafe_allow_html=True)
st.markdown('<div class="input-max-width">', unsafe_allow_html=True)

with st.form(key="chat_input_form", clear_on_submit=True):
    user_query = st.text_input(
        "Message Sigma Assistant...",
        placeholder="Type a message...",
        label_visibility="collapsed",
        key=f"query_input_{st.session_state.query_counter}"
    )
    submit_clicked = st.form_submit_button("Send", use_container_width=True)

if submit_clicked and user_query.strip():
    # Append User Message
    st.session_state.messages.append({
        "role": "user",
        "content": user_query,
        "time": datetime.now().strftime("%H:%M")
    })
    
    # Show typing indicator temporarily
    typing_placeholder = st.empty()
    typing_placeholder.markdown("""
    <div class="message-container">
        <div class="avatar-wrapper assistant-avatar">🤖</div>
        <div class="message-body">
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Get Response
    response_text = get_rag_response(user_query)
    typing_placeholder.empty()
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "time": datetime.now().strftime("%H:%M")
    })
    
    st.session_state.query_counter += 1
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Auto-Scroll to bottom
st.markdown("""
<script>
    function scrollToBottom() {
        window.parent.scrollTo({
            top: window.parent.document.body.scrollHeight,
            behavior: 'smooth'
        });
    }
    setTimeout(scrollToBottom, 100);
</script>
""", unsafe_allow_html=True)