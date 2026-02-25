"""
Streamlit Frontend — Titanic Dataset Chat Agent

Runs fully self-contained (no separate FastAPI server needed).
Works both locally and on Streamlit Cloud.
"""

import os
import sys
import base64
import pandas as pd
from pathlib import Path
import streamlit as st

# ─── Path Setup (works locally and on Streamlit Cloud) ───────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backend.agent import create_agent, run_query

# ─── Page Configuration ─────────────────────────────────────────
st.set_page_config(
    page_title="Titanic Data Agent",
    page_icon="🚢",
    layout="centered",
)

# ─── API Key: Streamlit Secrets → env var fallback ───────────────
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

if not os.getenv("GROQ_API_KEY"):
    st.error(
        "🔑 **GROQ_API_KEY not found.**\n\n"
        "- **Streamlit Cloud:** Add it in App Settings → Secrets\n"
        "- **Local:** Add it to `.env` or `.streamlit/secrets.toml`"
    )
    st.stop()

# ─── Load Data & Agent ONCE (cached across reruns) ───────────────
@st.cache_resource(show_spinner="🚀 Loading Titanic dataset and initializing agent...")
def load_agent():
    data_path = ROOT / "backend" / "data" / "titanic.csv"
    df = pd.read_csv(data_path)
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    if "Cabin" in df.columns:
        df = df.drop(columns=["Cabin"])
    return create_agent(df), df

agent_executor, df = load_agent()

# ─── Custom Styling ──────────────────────────────────────────────
st.markdown("""
<style>
    .stChatMessage {
        border-radius: 12px;
    }
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
    }
    .main-header p {
        color: #6b7280;
        font-size: 1.1rem;
    }
    .example-queries {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🚢 Titanic Data Agent</h1>
    <p>Ask questions about the Titanic dataset in plain English</p>
</div>
""", unsafe_allow_html=True)

# ─── Example Queries ─────────────────────────────────────────────
with st.expander("💡 Example questions you can ask", expanded=False):
    st.markdown("""
    - *"What percentage of passengers were male?"*
    - *"Show me a histogram of passenger ages"*
    - *"What was the average ticket fare?"*
    - *"How many passengers embarked from each port?"*
    - *"What was the survival rate by passenger class?"*
    - *"Show a bar chart of survival by gender"*
    """)

# ─── Chat History ────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show code if available
        if message.get("code"):
            with st.expander("🛠️ View Agent's Python Code"):
                st.code(message["code"], language="python")
                
        # Show image if available
        if message.get("image"):
            image_bytes = base64.b64decode(message["image"])
            st.image(image_bytes, use_container_width=True)

# ─── Chat Input ──────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about the Titanic dataset..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Send to backend and display response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Analyzing the dataset..."):
            try:
                result = run_query(agent_executor, prompt)
                answer_text = result.get("text", "No response received.")
                answer_image = result.get("image")
                answer_code = result.get("code")

                st.markdown(answer_text)

                if answer_code:
                    with st.expander("🛠️ View Agent's Python Code"):
                        st.code(answer_code, language="python")

                if answer_image:
                    image_bytes = base64.b64decode(answer_image)
                    st.image(image_bytes, use_container_width=True)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_text,
                    "image": answer_image,
                    "code": answer_code,
                })

            except Exception as e:
                error_msg = f"⚠️ Unexpected error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })

# ─── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🚢 About")
    st.markdown(
        "This chatbot analyzes the **Titanic dataset** (891 passengers) "
        "using a **LangChain Pandas Agent** powered by Groq LLM."
    )
    st.markdown("---")
    st.markdown("### 📊 Dataset Info")
    st.markdown(f"**Rows:** {df.shape[0]}  |  **Columns:** {df.shape[1]}")
    st.markdown("### 📋 Columns")
    st.markdown("""
    | Column | Type |
    |--------|------|
    | PassengerId | int |
    | Survived | 0/1 |
    | Pclass | 1/2/3 |
    | Name | string |
    | Sex | male/female |
    | Age | float |
    | SibSp | int |
    | Parch | int |
    | Ticket | string |
    | Fare | float |
    | Embarked | C/Q/S |
    """)
    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
