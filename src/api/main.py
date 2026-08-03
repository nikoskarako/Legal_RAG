"""FastAPI wrapper around the dense RAG pipeline.

Run from src/api/ with:  uvicorn main:app --host 0.0.0.0 --port 8001
"""
import json
import os
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402,F401  (puts src/rag on sys.path)

import chat_openrouter  # noqa: E402

app = FastAPI(title="Legal ChatBot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    model: str = "deepseek/deepseek-chat-v3-0324"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
def query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    settings = json.dumps({"model": req.model})
    result_json = chat_openrouter.process_query(req.query, settings)
    return json.loads(result_json)
