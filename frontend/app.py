"""
Streamlit Frontend — Titanic Dataset Chat Agent

A clean chat interface that:
- Accepts natural language questions
- Sends them to the FastAPI backend
- Renders text answers and matplotlib visualizations
"""

import base64
import requests
import streamlit as st
from io import BytesIO

# ─── Page Configuration ─────────────────────────────────────────
st.set_page_config(
    page_title="Titanic Data Agent",
    page_icon="🚢",
    layout="centered",
)

# ─── Backend URL ─────────────────────────────────────────────────
BACKEND_URL = "http://localhost:8000"

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
                response = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={"question": prompt},
                    timeout=120,  # LLM can take time
                )

                if response.status_code == 200:
                    data = response.json()
                    answer_text = data.get("text", "No response received.")
                    answer_image = data.get("image")
                    answer_code = data.get("code")

                    st.markdown(answer_text)
                    
                    if answer_code:
                        with st.expander("🛠️ View Agent's Python Code"):
                            st.code(answer_code, language="python")

                    if answer_image:
                        image_bytes = base64.b64decode(answer_image)
                        st.image(image_bytes, use_container_width=True)

                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer_text,
                        "image": answer_image,
                        "code": answer_code,
                    })
                else:
                    error_msg = f"⚠️ Backend error (status {response.status_code})"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                    })

            except requests.exceptions.ConnectionError:
                error_msg = (
                    "⚠️ Cannot connect to the backend. "
                    "Make sure the FastAPI server is running:\n\n"
                    "```bash\nuvicorn backend.main:app --port 8000\n```"
                )
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })
            except requests.exceptions.Timeout:
                error_msg = "⚠️ Request timed out. The query may be too complex. Try a simpler question."
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
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
    st.markdown("### 📊 Dataset Columns")
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
    | Cabin | string |
    | Embarked | C/Q/S |
    """)
    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
