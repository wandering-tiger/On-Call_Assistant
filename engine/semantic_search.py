"""Phase 2: Semantic search using DeepSeek LLM for relevance ranking.

Strategy:
1. Use keyword search (Phase 1) as first-pass retrieval to get candidates
2. Use DeepSeek chat API to evaluate semantic relevance of each candidate
3. Return re-ranked results with LLM-based relevance scores

This approach works well for small-to-medium document collections and
doesn't require a dedicated embeddings API.
"""

import json
import asyncio
from typing import Optional

import httpx

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_CHAT_MODEL
from engine.document_store import store

SEMANTIC_RANKING_PROMPT = """You are a search relevance judge. Given a search query and a document, rate how relevant the document is to the query on a scale of 0.0 to 1.0.

Query: {query}

Document Title: {title}
Document Content (excerpt): {content}

Return ONLY a JSON object with the following format:
{{"score": 0.XX, "reason": "brief explanation in Chinese"}}

The score should reflect semantic relevance, not just keyword matching. Consider:
- Does the document topic match the query intent?
- Would this document help answer the query?
- Is the document about the same domain/area as the query?

JSON:"""


def _build_semantic_request(query: str, doc_id: str) -> dict:
    """Build the API request for semantic relevance scoring."""
    doc = store.documents.get(doc_id)
    if not doc:
        return {"score": 0.0, "reason": "Document not found"}

    # Use first 2000 chars as excerpt
    content = doc["text"][:2000]

    return {
        "model": DEEPSEEK_CHAT_MODEL,
        "messages": [
            {
                "role": "user",
                "content": SEMANTIC_RANKING_PROMPT.format(
                    query=query,
                    title=doc["title"],
                    content=content,
                ),
            }
        ],
        "temperature": 0.1,
        "max_tokens": 200,
        "stream": False,
    }


async def _score_document(
    client: httpx.AsyncClient,
    query: str,
    doc_id: str,
) -> tuple[str, float, str]:
    """Score a single document's relevance to the query. Returns (doc_id, score, reason)."""
    request = _build_semantic_request(query, doc_id)

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = await client.post(
            f"{DEEPSEEK_BASE_URL}/v1/chat/completions",
            headers=headers,
            json=request,
            timeout=30.0,
        )
        if response.status_code != 200:
            return doc_id, 0.0, f"API error: {response.status_code}"

        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()

        # Parse JSON from response
        # Handle potential markdown code block wrapping
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        result = json.loads(content)
        score = float(result.get("score", 0.0))
        reason = result.get("reason", "")
        return doc_id, max(0.0, min(1.0, score)), reason

    except (json.JSONDecodeError, KeyError, IndexError) as e:
        return doc_id, 0.0, f"Parse error: {e}"
    except Exception as e:
        return doc_id, 0.0, str(e)


async def semantic_search(
    query: str,
    max_results: int = 10,
    candidate_multiplier: int = 3,
) -> dict:
    """
    Perform semantic search with LLM re-ranking.
    1. Get candidates from keyword search (more than needed)
    2. Re-rank using DeepSeek LLM
    3. Return top results
    """
    # Step 1: Get keyword search candidates
    candidates = store.search(query, max_results=max_results * candidate_multiplier)

    if not candidates:
        return {"query": query, "results": []}

    # Step 2: Re-rank with LLM
    async with httpx.AsyncClient() as client:
        tasks = [
            _score_document(client, query, c["id"]) for c in candidates
        ]
        scored = await asyncio.gather(*tasks)

    # Step 3: Sort by semantic score and build results
    doc_scores = {doc_id: (score, reason) for doc_id, score, reason in scored}

    results = []
    for candidate in candidates:
        doc_id = candidate["id"]
        score, reason = doc_scores.get(doc_id, (0.0, ""))
        if score > 0.05:  # Filter very low relevance
            results.append({
                "id": candidate["id"],
                "title": candidate["title"],
                "snippet": candidate["snippet"],
                "score": round(score, 4),
                "reason": reason,
            })

    # Sort by semantic score
    results.sort(key=lambda x: x["score"], reverse=True)
    results = results[:max_results]

    return {"query": query, "results": results}


def semantic_search_sync(query: str, max_results: int = 10) -> dict:
    """Synchronous wrapper for semantic search."""
    return asyncio.run(semantic_search(query, max_results))
