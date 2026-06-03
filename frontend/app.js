// ── Tab activation based on URL path ──
function initTab() {
    const path = window.location.pathname;
    let activeTab = 'v1';
    if (path.startsWith('/v2')) activeTab = 'v2';
    else if (path.startsWith('/v3')) activeTab = 'v3';

    // Activate nav tab
    const tab = document.querySelector(`.nav-tab[data-tab="${activeTab}"]`);
    if (tab) tab.classList.add('active');

    // Show corresponding view
    const view = document.getElementById(`view-${activeTab}`);
    if (view) view.style.display = 'block';

    // Focus input
    const input = document.getElementById(`${activeTab}-input`);
    if (input) input.focus();
}

// ── Phase 1: Keyword Search ──
async function searchV1() {
    const input = document.getElementById('v1-input');
    const btn = document.getElementById('v1-btn');
    const stats = document.getElementById('v1-stats');
    const resultsDiv = document.getElementById('v1-results');
    const query = input.value.trim();

    if (!query) return;

    btn.disabled = true;
    btn.textContent = '搜索中...';
    resultsDiv.innerHTML = '<div class="loading-indicator"><div class="loading-dots"><span></span><span></span><span></span></div>搜索中...</div>';

    try {
        const resp = await fetch(`/v1/search?q=${encodeURIComponent(query)}`);
        const data = await resp.json();
        stats.textContent = `共找到 ${data.results.length} 条结果`;
        if (data.results.length === 0) {
            resultsDiv.innerHTML = '<div class="empty-state"><div class="icon">📭</div><h3>未找到结果</h3><p>尝试使用不同的关键词搜索</p></div>';
        } else {
            resultsDiv.innerHTML = data.results.map(r => `
                <div class="result-card">
                    <div class="result-header">
                        <div class="result-title">${escapeHtml(r.title)}</div>
                        <div class="result-score">相关性: ${r.score.toFixed(2)} | ${r.id}</div>
                    </div>
                    <div class="result-snippet">${r.snippet}</div>
                </div>
            `).join('');
        }
    } catch (e) {
        resultsDiv.innerHTML = '<div class="empty-state"><div class="icon">⚠️</div><h3>搜索失败</h3><p>' + e.message + '</p></div>';
    } finally {
        btn.disabled = false;
        btn.textContent = '搜索';
    }
}

// ── Phase 2: Semantic Search ──
async function searchV2() {
    const input = document.getElementById('v2-input');
    const btn = document.getElementById('v2-btn');
    const stats = document.getElementById('v2-stats');
    const resultsDiv = document.getElementById('v2-results');
    const query = input.value.trim();

    if (!query) return;

    btn.disabled = true;
    btn.textContent = '搜索中...';
    resultsDiv.innerHTML = '<div class="loading-indicator"><div class="loading-dots"><span></span><span></span><span></span></div>语义搜索中（LLM 排序）...</div>';

    try {
        const resp = await fetch(`/v2/search?q=${encodeURIComponent(query)}`);
        const data = await resp.json();
        stats.textContent = `共找到 ${data.results.length} 条结果`;
        if (data.results.length === 0) {
            resultsDiv.innerHTML = '<div class="empty-state"><div class="icon">📭</div><h3>未找到相关结果</h3><p>尝试使用不同的描述方式搜索</p></div>';
        } else {
            resultsDiv.innerHTML = data.results.map(r => `
                <div class="result-card">
                    <div class="result-header">
                        <div class="result-title">${escapeHtml(r.title)}</div>
                        <div class="result-score">语义相关度: ${r.score.toFixed(4)} | ${r.id}</div>
                    </div>
                    <div class="result-snippet">${r.snippet}</div>
                    ${r.reason ? `<div class="result-reason">💡 ${escapeHtml(r.reason)}</div>` : ''}
                </div>
            `).join('');
        }
    } catch (e) {
        resultsDiv.innerHTML = '<div class="empty-state"><div class="icon">⚠️</div><h3>搜索失败</h3><p>' + e.message + '</p></div>';
    } finally {
        btn.disabled = false;
        btn.textContent = '语义搜索';
    }
}

// ── Phase 3: Agent Chat ──
let isAgentProcessing = false;

async function sendChat() {
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
        <div id="${loadingId}" class="loading-indicator">
            <div class="loading-dots"><span></span><span></span><span></span></div>
            Agent 思考中...
        </div>
    `);
    scrollToBottom(messagesDiv);

    try {
        const resp = await fetch('/v3/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query, history: [] }),
        });

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (!line.startsWith('data: ') || line === 'data: [DONE]') continue;
                try {
                    const event = JSON.parse(line.slice(6));

                    if (event.type === 'thought') {
                        addAgentStep('thought', event.content);
                    } else if (event.type === 'action') {
                        addAgentStep('action', event.input);
                    } else if (event.type === 'observation') {
                        addAgentStep('observation', event.content);
                    } else if (event.type === 'answer') {
                        addChatMessage('assistant', event.content);
                    } else if (event.type === 'error') {
                        addChatMessage('assistant', '❌ ' + event.content);
                    }
                } catch (e) {}
            }
            scrollToBottom(messagesDiv);
        }

        // Process remaining buffer
        if (buffer.startsWith('data: ') && buffer !== 'data: [DONE]') {
            try {
                const event = JSON.parse(buffer.slice(6));
                if (event.type === 'answer') {
                    addChatMessage('assistant', event.content);
                }
            } catch (e) {}
        }

    } catch (e) {
        addChatMessage('assistant', '❌ 连接失败: ' + e.message);
    } finally {
        // Remove loading
        const loading = document.getElementById(loadingId);
        if (loading) loading.remove();
        isAgentProcessing = false;
        btn.disabled = false;
        btn.textContent = '发送';
        scrollToBottom(messagesDiv);
    }
}

function addChatMessage(type, content) {
    const messagesDiv = document.getElementById('chat-messages');
    const div = document.createElement('div');
    div.className = 'chat-message ' + type;
    if (type === 'assistant') {
        // Render markdown for assistant messages
        const rendered = typeof marked !== 'undefined'
            ? marked.parse(content, { breaks: true })
            : '<p>' + escapeHtml(content) + '</p>';
        div.innerHTML = '<div class="msg-content">' + rendered + '</div>';
    } else {
        div.innerHTML = '<div class="msg-content">' + escapeHtml(content) + '</div>';
    }
    messagesDiv.appendChild(div);
}

function addAgentStep(type, content) {
    const messagesDiv = document.getElementById('chat-messages');

    if (type === 'thought') {
        const div = document.createElement('div');
        div.className = 'agent-step';
        div.innerHTML = '<div class="step-thought">💭 ' + escapeHtml(content) + '</div>';
        messagesDiv.appendChild(div);
    } else if (type === 'action') {
        const div = document.createElement('div');
        div.className = 'agent-step';
        div.innerHTML = `
            <div class="step-action">
                <span class="action-icon">📖</span>
                <span class="action-label">读取文档:</span>
                <span class="action-filename">${escapeHtml(content)}</span>
            </div>`;
        messagesDiv.appendChild(div);
    } else if (type === 'observation') {
        const div = document.createElement('div');
        div.className = 'agent-step';
        div.innerHTML = `
            <details class="step-observation">
                <summary>📄 文档内容 (点击展开)</summary>
                ${escapeHtml(content)}
            </details>`;
        messagesDiv.appendChild(div);
    }
}

function scrollToBottom(el) {
    el.scrollTop = el.scrollHeight;
}

function escapeHtml(text) {
    const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
    return String(text).replace(/[&<>"']/g, m => map[m]);
}

// ── Event listeners ──
document.addEventListener('DOMContentLoaded', () => {
    initTab();

    // Auto-resize textarea
    const chatInput = document.getElementById('chat-input');
    if (chatInput) {
        chatInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 150) + 'px';
        });
    }

    // Enter key for search inputs
    const v1Input = document.getElementById('v1-input');
    if (v1Input) {
        v1Input.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') searchV1();
        });
    }
    const v2Input = document.getElementById('v2-input');
    if (v2Input) {
        v2Input.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') searchV2();
        });
    }
});
