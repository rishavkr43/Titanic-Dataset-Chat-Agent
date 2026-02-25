"""
FastAPI Backend — Titanic Dataset Chat Agent

Responsibilities:
- Load Titanic CSV once at startup
- Initialize LangChain Pandas agent
- Expose POST /chat endpoint for queries
- Return text + optional base64 plot image
"""

import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.agent import create_agent, run_query

# ─── Load environment variables ──────────────────────────────────
load_dotenv()

# ─── Validate API key ───────────────────────────────────────────
if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError(
        "GROQ_API_KEY not set! Get a free key at https://console.groq.com/keys "
        "and add it to your .env file."
    )

# ─── App setup ───────────────────────────────────────────────────
app = FastAPI(
    title="Titanic Chat Agent",
    description="Natural language interface for Titanic dataset analysis",
    version="1.0.0",
)

# Allow Streamlit (different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Request/Response models ─────────────────────────────────────

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    text: str
    image: str | None = None
    code: str | None = None

# ─── Load data and create agent (ONCE at startup) ───────────────

DATA_PATH = Path(__file__).parent / "data" / "titanic.csv"

# Load the dataset
df = pd.read_csv(DATA_PATH)

# Phase 2 Improvement: Pre-clean data for the agent
# 1. Fill missing Age with median
df["Age"] = df["Age"].fillna(df["Age"].median())
# 2. Fill missing Embarked with mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
# 3. Drop Cabin column (too many missing values, confuses agent)
if "Cabin" in df.columns:
    df = df.drop(columns=["Cabin"])

print(f"✅ Loaded and cleaned Titanic dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# Create the agent
agent_executor = create_agent(df)
print("✅ LangChain Pandas agent initialized")


# ─── Endpoints ───────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "dataset_rows": len(df),
        "dataset_columns": list(df.columns),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Process a natural language question about the Titanic dataset.

    The agent will:
    1. Interpret the question
    2. Generate Python/Pandas code
    3. Execute it against the real DataFrame
    4. Return computed answer + any generated plot
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    result = run_query(agent_executor, request.question)
    return ChatResponse(text=result["text"], image=result["image"])
