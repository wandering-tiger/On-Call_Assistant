"""
On-Call Assistant — FastAPI Application

Three phases:
  /v1 — Keyword search engine
  /v2 — Semantic search
  /v3 — On-Call Assistant Agent (ReAct)
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
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
    return HTMLResponse(content=_get_index_html("v1"))


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
    """Serve the search page for Phase 2."""
    return HTMLResponse(content=_get_index_html("v2"))


@app.get("/v2/search")
async def v2_search(q: str = ""):
    """Semantic search with LLM re-ranking."""
    results = await semantic_search(q)
    return results


# ── Phase 3: Agent (/v3) ──────────────────────────────────────────────

@app.get("/v3", response_class=HTMLResponse)
async def v3_page():
    """Serve the agent chat page for Phase 3."""
    return HTMLResponse(content=_get_index_html("v3"))


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
    return HTMLResponse(content=_get_index_html("v1"))


# ── HTML template ─────────────────────────────────────────────────────

def _get_index_html(active_tab: str = "v1") -> str:
    """Return the single-page app HTML with the active tab set."""
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>On-Call Assistant</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        :root {{
            --bg: #0f172a;
            --surface: #1e293b;
            --surface2: #334155;
            --border: #475569;
            --text: #f1f5f9;
            --text2: #94a3b8;
            --accent: #38bdf8;
            --accent2: #0284c7;
            --success: #34d399;
            --warning: #fbbf24;
            --danger: #f87171;
            --radius: 10px;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            line-height: 1.6;
        }}

        /* ── Header ── */
        .header {{
            background: var(--surface);
            border-bottom: 1px solid var(--border);
            padding: 0 24px;
            display: flex;
            align-items: center;
            height: 60px;
            gap: 32px;
            position: sticky;
            top: 0;
            z-index: 100;
            backdrop-filter: blur(12px);
        }}
        .header .logo {{
            font-size: 20px;
            font-weight: 700;
            color: var(--accent);
            white-space: nowrap;
            letter-spacing: -0.5px;
        }}
        .header .logo span {{ color: var(--text2); font-weight: 400; }}

        /* ── Navigation Tabs ── */
        .nav-tabs {{
            display: flex;
            gap: 4px;
            background: var(--surface2);
            border-radius: 8px;
            padding: 3px;
        }}
        .nav-tab {{
            padding: 8px 20px;
            border: none;
            background: transparent;
            color: var(--text2);
            cursor: pointer;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.2s;
            white-space: nowrap;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
        }}
        .nav-tab:hover {{ color: var(--text); background: rgba(255,255,255,0.05); }}
        .nav-tab.active {{ background: var(--accent); color: #0f172a; font-weight: 600; }}

        /* ── Main Content ── */
        .main {{
            max-width: 900px;
            margin: 32px auto;
            padding: 0 24px;
        }}

        /* ── Search Bar ── */
        .search-section {{
            margin-bottom: 32px;
        }}
        .search-box {{
            display: flex;
            gap: 12px;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 6px;
            transition: border-color 0.2s;
        }}
        .search-box:focus-within {{
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(56,189,248,0.1);
        }}
        .search-input {{
            flex: 1;
            background: transparent;
            border: none;
            color: var(--text);
            font-size: 16px;
            padding: 12px 16px;
            outline: none;
            font-family: inherit;
        }}
        .search-input::placeholder {{ color: var(--text2); }}
        .search-btn {{
            padding: 12px 28px;
            background: var(--accent);
            color: #0f172a;
            border: none;
            border-radius: 8px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            white-space: nowrap;
        }}
        .search-btn:hover {{ background: #7dd3fc; }}
        .search-btn:disabled {{ opacity: 0.5; cursor: not-allowed; }}

        .search-stats {{
            color: var(--text2);
            font-size: 14px;
            margin-top: 12px;
            padding-left: 12px;
        }}

        /* ── Results ── */
        .results-list {{
            display: flex;
            flex-direction: column;
            gap: 16px;
        }}
        .result-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 20px 24px;
            transition: border-color 0.2s;
        }}
        .result-card:hover {{ border-color: var(--accent); }}
        .result-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 16px;
            margin-bottom: 10px;
        }}
        .result-title {{
            font-size: 17px;
            font-weight: 600;
            color: var(--accent);
        }}
        .result-score {{
            font-size: 13px;
            color: var(--text2);
            background: var(--surface2);
            padding: 3px 10px;
            border-radius: 20px;
            white-space: nowrap;
        }}
        .result-snippet {{
            font-size: 14px;
            color: var(--text2);
            line-height: 1.7;
        }}
        .result-snippet mark {{
            background: rgba(251,191,36,0.3);
            color: var(--warning);
            padding: 1px 4px;
            border-radius: 2px;
        }}
        .result-reason {{
            font-size: 13px;
            color: var(--success);
            margin-top: 8px;
            padding-top: 8px;
            border-top: 1px solid var(--border);
        }}

        /* ── Chat Interface (Phase 3) ── */
        .chat-container {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            overflow: hidden;
            display: flex;
            flex-direction: column;
            height: calc(100vh - 200px);
            min-height: 500px;
        }}
        .chat-messages {{
            flex: 1;
            overflow-y: auto;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }}
        .chat-message {{
            max-width: 85%;
            animation: fadeIn 0.3s ease;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(8px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .chat-message.user {{
            align-self: flex-end;
        }}
        .chat-message.user .msg-content {{
            background: var(--accent2);
            color: var(--text);
            border-radius: 16px 16px 4px 16px;
            padding: 12px 18px;
            font-size: 15px;
        }}
        .chat-message.assistant {{
            align-self: flex-start;
        }}
        .chat-message.assistant .msg-content {{
            background: var(--surface2);
            color: var(--text);
            border-radius: 16px 16px 16px 4px;
            padding: 12px 18px;
            font-size: 15px;
            line-height: 1.7;
            white-space: pre-wrap;
            word-break: break-word;
        }}

        /* ── Agent Steps ── */
        .agent-step {{
            align-self: flex-start;
            max-width: 90%;
        }}
        .step-thought {{
            background: rgba(56,189,248,0.08);
            border-left: 3px solid var(--accent);
            padding: 10px 16px;
            border-radius: 0 8px 8px 0;
            font-size: 14px;
            color: var(--text2);
            margin-bottom: 8px;
            font-style: italic;
        }}
        .step-action {{
            background: rgba(251,191,36,0.1);
            border: 1px solid rgba(251,191,36,0.2);
            padding: 10px 16px;
            border-radius: 8px;
            font-size: 13px;
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 4px;
        }}
        .step-action .action-icon {{
            font-size: 18px;
        }}
        .step-action .action-label {{
            color: var(--warning);
            font-weight: 600;
        }}
        .step-action .action-filename {{
            color: var(--accent);
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            background: rgba(56,189,248,0.1);
            padding: 2px 8px;
            border-radius: 4px;
        }}
        .step-observation {{
            background: var(--surface2);
            border-radius: 8px;
            padding: 10px 16px;
            font-size: 12px;
            color: var(--text2);
            max-height: 120px;
            overflow-y: auto;
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            white-space: pre-wrap;
            word-break: break-all;
            margin-bottom: 8px;
        }}
        .step-observation summary {{
            color: var(--text2);
            cursor: pointer;
            font-family: inherit;
            font-size: 13px;
        }}

        /* ── Chat Input ── */
        .chat-input-area {{
            display: flex;
            gap: 12px;
            padding: 16px 20px;
            border-top: 1px solid var(--border);
            background: var(--surface2);
        }}
        .chat-input {{
            flex: 1;
            background: var(--surface);
            border: 1px solid var(--border);
            color: var(--text);
            font-size: 15px;
            padding: 12px 16px;
            border-radius: 8px;
            outline: none;
            font-family: inherit;
            resize: none;
            min-height: 48px;
        }}
        .chat-input:focus {{
            border-color: var(--accent);
        }}
        .chat-send {{
            padding: 12px 24px;
            background: var(--accent);
            color: #0f172a;
            border: none;
            border-radius: 8px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            align-self: flex-end;
        }}
        .chat-send:hover {{ background: #7dd3fc; }}
        .chat-send:disabled {{ opacity: 0.5; cursor: not-allowed; }}

        /* ── Status indicators ── */
        .loading-indicator {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            color: var(--text2);
            font-size: 14px;
        }}
        .loading-dots {{
            display: flex;
            gap: 4px;
        }}
        .loading-dots span {{
            width: 6px;
            height: 6px;
            background: var(--accent);
            border-radius: 50%;
            animation: bounce 1.4s infinite ease-in-out;
        }}
        .loading-dots span:nth-child(1) {{ animation-delay: 0s; }}
        .loading-dots span:nth-child(2) {{ animation-delay: 0.2s; }}
        .loading-dots span:nth-child(3) {{ animation-delay: 0.4s; }}
        @keyframes bounce {{
            0%, 80%, 100% {{ transform: scale(0); }}
            40% {{ transform: scale(1); }}
        }}

        .empty-state {{
            text-align: center;
            padding: 64px 24px;
            color: var(--text2);
        }}
        .empty-state .icon {{
            font-size: 48px;
            margin-bottom: 16px;
            opacity: 0.5;
        }}
        .empty-state h3 {{
            font-size: 18px;
            color: var(--text);
            margin-bottom: 8px;
        }}
        .empty-state p {{
            font-size: 14px;
            max-width: 400px;
            margin: 0 auto;
        }}

        /* ── Responsive ── */
        @media (max-width: 640px) {{
            .header {{ padding: 0 16px; gap: 12px; }}
            .header .logo {{ font-size: 16px; }}
            .nav-tab {{ padding: 6px 14px; font-size: 13px; }}
            .main {{ padding: 0 16px; margin-top: 20px; }}
            .result-header {{ flex-direction: column; gap: 4px; }}
        }}

        /* ── Scrollbar ── */
        ::-webkit-scrollbar {{ width: 6px; }}
        ::-webkit-scrollbar-track {{ background: transparent; }}
        ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: var(--text2); }}
    </style>
</head>
<body>
    <header class="header">
        <div class="logo">🛟 On-Call <span>Assistant</span></div>
        <nav class="nav-tabs">
            <a href="/v1" class="nav-tab {'active' if active_tab == 'v1' else ''}" data-tab="v1">🔍 关键词搜索</a>
            <a href="/v2" class="nav-tab {'active' if active_tab == 'v2' else ''}" data-tab="v2">🧠 语义搜索</a>
            <a href="/v3" class="nav-tab {'active' if active_tab == 'v3' else ''}" data-tab="v3">🤖 Agent 助手</a>
        </nav>
    </header>

    <main class="main">
        <div id="view-v1" class="view" style="display: {'block' if active_tab == 'v1' else 'none'}">
            <div class="search-section">
                <div class="search-box">
                    <input type="text" class="search-input" id="v1-input"
                           placeholder="输入关键词搜索 SOP 文档，例如：OOM、故障、CDN..."
                           autocomplete="off">
                    <button class="search-btn" id="v1-btn" onclick="searchV1()">搜索</button>
                </div>
                <div class="search-stats" id="v1-stats"></div>
            </div>
            <div class="results-list" id="v1-results">
                <div class="empty-state">
                    <div class="icon">🔍</div>
                    <h3>关键词搜索引擎</h3>
                    <p>基于 TF-IDF 的全文检索，支持中英文混合搜索。输入关键词开始查找相关 SOP 文档。</p>
                </div>
            </div>
        </div>

        <div id="view-v2" class="view" style="display: {'block' if active_tab == 'v2' else 'none'}">
            <div class="search-section">
                <div class="search-box">
                    <input type="text" class="search-input" id="v2-input"
                           placeholder="用自然语言描述问题，例如：服务器挂了怎么办..."
                           autocomplete="off">
                    <button class="search-btn" id="v2-btn" onclick="searchV2()">语义搜索</button>
                </div>
                <div class="search-stats" id="v2-stats"></div>
            </div>
            <div class="results-list" id="v2-results">
                <div class="empty-state">
                    <div class="icon">🧠</div>
                    <h3>语义搜索引擎</h3>
                    <p>基于 LLM 的语义相关性排序，理解查询意图。即使关键词不完全匹配也能找到相关文档。</p>
                </div>
            </div>
        </div>

        <div id="view-v3" class="view" style="display: {'block' if active_tab == 'v3' else 'none'}">
            <div class="chat-container">
                <div class="chat-messages" id="chat-messages">
                    <div class="empty-state" id="chat-empty">
                        <div class="icon">🤖</div>
                        <h3>On-Call Agent 助手</h3>
                        <p>一个基于 ReAct 框架的智能 Agent，可以自动查阅 SOP 文档来回答您的值班问题。试试问："数据库主从延迟超过30秒怎么处理？"</p>
                    </div>
                </div>
                <div class="chat-input-area">
                    <textarea class="chat-input" id="chat-input"
                              placeholder="输入您的 On-Call 问题..."
                              rows="1"
                              onkeydown="if(event.key==='Enter' && !event.shiftKey && !event.ctrlKey){{event.preventDefault();sendChat();}}"></textarea>
                    <button class="chat-send" id="chat-send-btn" onclick="sendChat()">发送</button>
                </div>
            </div>
        </div>
    </main>

    <script>
        // ── Tab navigation ──
        document.querySelectorAll('.nav-tab').forEach(tab => {{
            tab.addEventListener('click', function(e) {{
                // Allow default navigation (the tabs are links)
            }});
        }});

        // ── Phase 1: Keyword Search ──
        async function searchV1() {{
            const input = document.getElementById('v1-input');
            const btn = document.getElementById('v1-btn');
            const stats = document.getElementById('v1-stats');
            const resultsDiv = document.getElementById('v1-results');
            const query = input.value.trim();

            if (!query) return;

            btn.disabled = true;
            btn.textContent = '搜索中...';
            resultsDiv.innerHTML = '<div class="loading-indicator"><div class="loading-dots"><span></span><span></span><span></span></div>搜索中...</div>';

            try {{
                const resp = await fetch(`/v1/search?q=${{encodeURIComponent(query)}}`);
                const data = await resp.json();
                stats.textContent = `共找到 ${{data.results.length}} 条结果`;
                if (data.results.length === 0) {{
                    resultsDiv.innerHTML = '<div class="empty-state"><div class="icon">📭</div><h3>未找到结果</h3><p>尝试使用不同的关键词搜索</p></div>';
                }} else {{
                    resultsDiv.innerHTML = data.results.map(r => `
                        <div class="result-card">
                            <div class="result-header">
                                <div class="result-title">${{escapeHtml(r.title)}}</div>
                                <div class="result-score">相关性: ${{r.score.toFixed(2)}} | ${{r.id}}</div>
                            </div>
                            <div class="result-snippet">${{r.snippet}}</div>
                        </div>
                    `).join('');
                }}
            }} catch (e) {{
                resultsDiv.innerHTML = '<div class="empty-state"><div class="icon">⚠️</div><h3>搜索失败</h3><p>' + e.message + '</p></div>';
            }} finally {{
                btn.disabled = false;
                btn.textContent = '搜索';
            }}
        }}

        // ── Phase 2: Semantic Search ──
        async function searchV2() {{
            const input = document.getElementById('v2-input');
            const btn = document.getElementById('v2-btn');
            const stats = document.getElementById('v2-stats');
            const resultsDiv = document.getElementById('v2-results');
            const query = input.value.trim();

            if (!query) return;

            btn.disabled = true;
            btn.textContent = '搜索中...';
            resultsDiv.innerHTML = '<div class="loading-indicator"><div class="loading-dots"><span></span><span></span><span></span></div>语义搜索中（LLM 排序）...</div>';

            try {{
                const resp = await fetch(`/v2/search?q=${{encodeURIComponent(query)}}`);
                const data = await resp.json();
                stats.textContent = `共找到 ${{data.results.length}} 条结果`;
                if (data.results.length === 0) {{
                    resultsDiv.innerHTML = '<div class="empty-state"><div class="icon">📭</div><h3>未找到相关结果</h3><p>尝试使用不同的描述方式搜索</p></div>';
                }} else {{
                    resultsDiv.innerHTML = data.results.map(r => `
                        <div class="result-card">
                            <div class="result-header">
                                <div class="result-title">${{escapeHtml(r.title)}}</div>
                                <div class="result-score">语义相关度: ${{r.score.toFixed(4)}} | ${{r.id}}</div>
                            </div>
                            <div class="result-snippet">${{r.snippet}}</div>
                            ${{r.reason ? `<div class="result-reason">💡 ${{escapeHtml(r.reason)}}</div>` : ''}}
                        </div>
                    `).join('');
                }}
            }} catch (e) {{
                resultsDiv.innerHTML = '<div class="empty-state"><div class="icon">⚠️</div><h3>搜索失败</h3><p>' + e.message + '</p></div>';
            }} finally {{
                btn.disabled = false;
                btn.textContent = '语义搜索';
            }}
        }}

        // ── Phase 3: Agent Chat ──
        let isAgentProcessing = false;

        async function sendChat() {{
            if (isAgentProcessing) return;

            const input = document.getElementById('chat-input');
            const btn = document.getElementById('chat-send-btn');
            const messagesDiv = document.getElementById('chat-messages');
            const emptyDiv = document.getElementById('chat-empty');
            const query = input.value.trim();

            if (!query) return;

            // Hide empty state
            if (emptyDiv) emptyDiv.style.display = 'none';

            // Add user message
            addChatMessage('user', query);
            input.value = '';
            input.style.height = 'auto';

            isAgentProcessing = true;
            btn.disabled = true;
            btn.textContent = '思考中...';

            // Add loading indicator
            const loadingId = 'loading-' + Date.now();
            messagesDiv.insertAdjacentHTML('beforeend', `
                <div id="${{loadingId}}" class="loading-indicator">
                    <div class="loading-dots"><span></span><span></span><span></span></div>
                    Agent 思考中...
                </div>
            `);
            scrollToBottom(messagesDiv);

            try {{
                const resp = await fetch('/v3/chat', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ query: query, history: [] }}),
                }});

                const reader = resp.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                let currentThought = null;
                let currentAction = null;

                while (true) {{
                    const {{ done, value }} = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, {{ stream: true }});
                    const lines = buffer.split('\\n');
                    buffer = lines.pop() || '';

                    for (const line of lines) {{
                        if (!line.startsWith('data: ') || line === 'data: [DONE]') continue;
                        try {{
                            const event = JSON.parse(line.slice(6));

                            if (event.type === 'thought') {{
                                currentThought = event.content;
                                // Show thought
                                addAgentStep('thought', event.content);
                            }} else if (event.type === 'action') {{
                                addAgentStep('action', event.input);
                            }} else if (event.type === 'observation') {{
                                addAgentStep('observation', event.content);
                            }} else if (event.type === 'answer') {{
                                addChatMessage('assistant', event.content);
                            }} else if (event.type === 'error') {{
                                addChatMessage('assistant', '❌ ' + event.content);
                            }}
                        }} catch (e) {{}}
                    }}
                    scrollToBottom(messagesDiv);
                }}

                // Process remaining buffer
                if (buffer.startsWith('data: ') && buffer !== 'data: [DONE]') {{
                    try {{
                        const event = JSON.parse(buffer.slice(6));
                        if (event.type === 'answer') {{
                            addChatMessage('assistant', event.content);
                        }}
                    }} catch (e) {{}}
                }}

            }} catch (e) {{
                addChatMessage('assistant', '❌ 连接失败: ' + e.message);
            }} finally {{
                // Remove loading
                const loading = document.getElementById(loadingId);
                if (loading) loading.remove();
                isAgentProcessing = false;
                btn.disabled = false;
                btn.textContent = '发送';
                scrollToBottom(messagesDiv);
            }}
        }}

        function addChatMessage(type, content) {{
            const messagesDiv = document.getElementById('chat-messages');
            const div = document.createElement('div');
            div.className = 'chat-message ' + type;
            div.innerHTML = '<div class="msg-content">' + escapeHtml(content) + '</div>';
            messagesDiv.appendChild(div);
        }}

        function addAgentStep(type, content) {{
            const messagesDiv = document.getElementById('chat-messages');

            if (type === 'thought') {{
                const div = document.createElement('div');
                div.className = 'agent-step';
                div.innerHTML = '<div class="step-thought">💭 ' + escapeHtml(content) + '</div>';
                messagesDiv.appendChild(div);
            }} else if (type === 'action') {{
                const div = document.createElement('div');
                div.className = 'agent-step';
                div.innerHTML = `
                    <div class="step-action">
                        <span class="action-icon">📖</span>
                        <span class="action-label">读取文档:</span>
                        <span class="action-filename">${{escapeHtml(content)}}</span>
                    </div>`;
                messagesDiv.appendChild(div);
            }} else if (type === 'observation') {{
                const div = document.createElement('div');
                div.className = 'agent-step';
                const truncated = content.length > 500 ? content.substring(0, 500) + '...' : content;
                div.innerHTML = `
                    <details class="step-observation">
                        <summary>📄 文档内容 (点击展开)</summary>
                        ${{escapeHtml(content)}}
                    </details>`;
                messagesDiv.appendChild(div);
            }}
        }}

        function scrollToBottom(el) {{
            el.scrollTop = el.scrollHeight;
        }}

        function escapeHtml(text) {{
            const map = {{ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' }};
            return String(text).replace(/[&<>"']/g, m => map[m]);
        }}

        // ── Auto-resize textarea ──
        document.getElementById('chat-input').addEventListener('input', function() {{
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 150) + 'px';
        }});

        // ── Enter key for search inputs ──
        document.getElementById('v1-input').addEventListener('keydown', function(e) {{
            if (e.key === 'Enter') searchV1();
        }});
        document.getElementById('v2-input').addEventListener('keydown', function(e) {{
            if (e.key === 'Enter') searchV2();
        }});

        // ── Focus input on load ──
        document.getElementById('{active_tab}-input')?.focus();
    </script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
