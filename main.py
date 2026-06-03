"""
On-Call Assistant — FastAPI Application

Three phases:
  /v1 — Keyword search engine
  /v2 — Semantic search
  /v3 — On-Call Assistant Agent (ReAct)
"""

import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json

from engine.keyword_search import keyword_search, add_document
from engine.semantic_search import semantic_search
from engine.agent import agent as react_agent

app = FastAPI(title="On-Call Assistant", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files (CSS, JS)
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# ── Models ────────────────────────────────────────────────────────────

class DocumentUpload(BaseModel):
    id: str
    html: str

class ChatRequest(BaseModel):
    query: str
    history: list[dict] = []

# ── Phase 1: Keyword Search (/v1) ─────────────────────────────────────

@app.get("/v1", response_class=HTMLResponse)
async def v1_page():
    """Serve the search page for Phase 1."""
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.post("/v1/documents", status_code=201)
async def v1_add_document(doc: DocumentUpload):
    """Add or update a document."""
    result = add_document(doc.id, doc.html)
    return result


@app.get("/v1/search")
async def v1_search(q: str = ""):
    """Keyword search."""
    results = keyword_search(q)
    return results


# ── Phase 2: Semantic Search (/v2) ────────────────────────────────────

@app.get("/v2", response_class=HTMLResponse)
async def v2_page():
    """Serve the semantic search page for Phase 2."""
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/v2/search")
async def v2_search(q: str = ""):
    """Semantic search with LLM re-ranking."""
    results = await semantic_search(q)
    return results


# ── Phase 3: Agent (/v3) ──────────────────────────────────────────────

@app.get("/v3", response_class=HTMLResponse)
async def v3_page():
    """Serve the agent chat page for Phase 3."""
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.post("/v3/chat")
async def v3_chat(request: ChatRequest):
    """Stream agent chat response via SSE."""
    async def event_stream():
        async for event in react_agent.chat(request.query, request.history):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Static files ──────────────────────────────────────────────────────

@app.get("/api/documents")
async def list_documents():
    """List all available documents."""
    from engine.document_store import store
    return store.list_documents()


# ── Root redirect ─────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
