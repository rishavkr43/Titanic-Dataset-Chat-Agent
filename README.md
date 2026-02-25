# 🚢 Titanic Data Agent

A full-stack AI-powered chat interface to explore the Titanic dataset using natural language. Ask questions in plain English and get instant answers, charts, and generated code — powered by **Groq LLaMA 3.3 70B**, **LangChain**, and **FastAPI + Streamlit**.

---

## 🧱 Tech Stack

| Layer     | Technology                          |
|-----------|-------------------------------------|
| LLM       | Groq — `llama-3.3-70b-versatile`    |
| Agent     | LangChain Experimental Pandas Agent |
| Backend   | FastAPI + Uvicorn                   |
| Frontend  | Streamlit                           |
| Data      | Titanic CSV (891 rows)              |

---

## 📁 Project Structure

```
├── backend/
│   ├── main.py          # FastAPI app — exposes POST /chat endpoint
│   ├── agent.py         # LangChain Pandas agent setup & query runner
│   └── data/
│       └── titanic.csv  # Titanic dataset
├── frontend/
│   └── app.py           # Streamlit chat UI
├── .env                 # Environment variables (GROQ_API_KEY)
├── .env.example         # Example env file (safe to commit)
├── requirements.txt     # Python dependencies
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd rishavproject
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## 🚀 Running the App

You need **two terminals** — one for the backend, one for the frontend.

### Terminal 1 — Start the FastAPI backend

```bash
# Windows
venv\Scripts\python.exe -m uvicorn backend.main:app --reload --port 8000

# macOS/Linux
python -m uvicorn backend.main:app --reload --port 8000
```

Backend runs at: [http://localhost:8000](http://localhost:8000)  
API docs at: [http://localhost:8000/docs](http://localhost:8000/docs)

### Terminal 2 — Start the Streamlit frontend

```bash
# Windows
venv\Scripts\python.exe -m streamlit run frontend/app.py

# macOS/Linux
streamlit run frontend/app.py
```

Frontend runs at: [http://localhost:8501](http://localhost:8501)

---

## 💬 Example Questions

- *How many passengers survived?*
- *What was the survival rate by gender?*
- *Show a bar chart of survival by passenger class*
- *What is the average age of survivors vs non-survivors?*
- *Which port of embarkation had the highest survival rate?*

---

## 🔌 API Reference

### `POST /chat`

**Request:**
```json
{
  "question": "What was the survival rate by gender?"
}
```

**Response:**
```json
{
  "text": "Female survival rate: 74.2%, Male survival rate: 18.9%",
  "image": "<base64-encoded PNG or null>",
  "code": "df.groupby('Sex')['Survived'].mean()"
}
```
