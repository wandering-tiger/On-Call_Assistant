"""Phase 2: Semantic search using DeepSeek LLM for direct document ranking.

Strategy (single-stage LLM ranking):
1. Build document summaries (title + first 800 chars of content) for every SOP
2. In ONE LLM call, present all documents with their summaries + the query
3. Ask LLM to select and rank the most relevant documents
4. Parse the ranked list and return results

This avoids the keyword-search bottleneck: the LLM sees ALL documents
and judges relevance semantically, so colloquial queries like "服务器挂了"
correctly match backend/SRE docs even though "挂了" never appears in them.
"""

import json
import asyncio
from typing import Optional

import httpx

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_CHAT_MODEL
from engine.document_store import store

# ── Document summaries ────────────────────────────────────────────────

def _build_doc_summaries() -> list[dict]:
    """Build title + summary for every indexed document."""
    summaries = []
    for doc_id, doc in store.documents.items():
        # First 800 chars of plain text as the summary
        text = doc["text"][:800].replace("\n", " ").strip()
        summaries.append({
            "id": doc_id,
            "title": doc["title"],
            "summary": text,
        })
    return summaries


# ── LLM Ranking Prompt ────────────────────────────────────────────────

RANKING_SYSTEM = """你是一个专业的搜索引擎排序专家。用户会用自然语言描述他们遇到的运维问题，你需要从所有 SOP 文档中找出最相关的。

## 规则
1. 理解用户查询的真实意图（用户可能使用口语化表达，如"挂了"="服务不可用/宕机"）
2. 根据文档的主题领域和内容，判断哪些文档能最好地回答用户的问题
3. 按相关度从高到低排序，只返回相关度 > 0.3 的文档
4. 返回严格的 JSON 格式，不要包含任何其他文字"""


def _build_ranking_prompt(query: str) -> str:
    """Build the prompt with all document summaries for ranking."""
    summaries = _build_doc_summaries()

    docs_text = ""
    for i, s in enumerate(summaries, 1):
        docs_text += f"[{i}] ID: {s['id']} | {s['title']}\n   摘要: {s['summary']}\n\n"

    return f"""以下是所有可用的 SOP 文档：

{docs_text}
---
用户查询："{query}"

请判断哪些文档与用户查询最相关。按相关度从高到低排序，返回 JSON：

{{"results": [
  {{"id": "sop-xxx", "score": 0.95, "reason": "一句话中文解释为什么相关"}},
  ...
]}}

注意：
- score 范围 0.0～1.0，只包含 score > 0.3 的文档
- "挂了" = 服务宕机/不可用/崩溃
- "被攻击" = 安全事件/入侵/DDoS
- 口语化查询要理解其真实意图"""


# ── Single-call LLM ranking ──────────────────────────────────────────

async def semantic_search(query: str, max_results: int = 10) -> dict:
    """
    Semantic search via single LLM ranking call.
    The LLM sees ALL documents and ranks them by semantic relevance.
    """
    prompt = _build_ranking_prompt(query)

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    request_body = {
        "model": DEEPSEEK_CHAT_MODEL,
        "messages": [
            {"role": "system", "content": RANKING_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 2048,
        "stream": False,
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{DEEPSEEK_BASE_URL}/v1/chat/completions",
                headers=headers,
                json=request_body,
                timeout=60.0,
            )

            if response.status_code != 200:
                return {"query": query, "results": [], "error": f"API error: {response.status_code}"}

            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()

            # Parse JSON from response
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            parsed = json.loads(content)
            ranked = parsed.get("results", [])

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            return {"query": query, "results": [], "error": f"LLM 响应解析失败: {e}"}
        except Exception as e:
            return {"query": query, "results": [], "error": str(e)}

    # Filter, sort and format results
    results = []
    for item in ranked:
        doc_id = item.get("id", "")
        doc = store.documents.get(doc_id)
        if not doc:
            # Try adding .html
            doc = store.documents.get(doc_id + ".html") if not doc_id.endswith(".html") else None
            if not doc:
                continue

        score = max(0.0, min(1.0, float(item.get("score", 0.0))))
        if score <= 0.3:
            continue

        results.append({
            "id": doc["id"],
            "title": doc["title"],
            "snippet": doc["text"][:200] + ("..." if len(doc["text"]) > 200 else ""),
            "score": round(score, 4),
            "reason": item.get("reason", ""),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    results = results[:max_results]

    return {"query": query, "results": results}


def semantic_search_sync(query: str, max_results: int = 10) -> dict:
    """Synchronous wrapper for semantic search."""
    return asyncio.run(semantic_search(query, max_results))
