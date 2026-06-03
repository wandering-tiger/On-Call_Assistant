# 🛟 On-Call Assistant

基于 **DeepSeek LLM** 的智能 On-Call 助手 Web 应用，为值班工程师提供 SOP 文档检索和故障处理建议。

## 项目架构

```
On-Call_Assistant/
├── main.py                      # FastAPI 应用入口（路由 + 内嵌前端）
├── config.py                    # 配置文件（API Key、路径等）
├── requirements.txt             # Python 依赖
├── engine/
│   ├── __init__.py
│   ├── document_store.py        # 文档存储：HTML 解析、中文分词、倒排索引、TF-IDF
│   ├── keyword_search.py        # Phase 1：关键词搜索引擎
│   ├── semantic_search.py       # Phase 2：语义搜索引擎（LLM 重排序）
│   └── agent.py                 # Phase 3：ReAct Agent（自实现框架）
├── data/                        # 10 份部门 On-Call SOP 文档
│   ├── sop-001.html             # 后端服务
│   ├── sop-002.html             # 数据库 DBA
│   ├── sop-003.html             # 前端 Web
│   ├── sop-004.html             # SRE 基础设施
│   ├── sop-005.html             # 信息安全
│   ├── sop-006.html             # 数据平台
│   ├── sop-007.html             # 移动客户端
│   ├── sop-008.html             # AI 算法
│   ├── sop-009.html             # QA 质量保证
│   └── sop-010.html             # 网络 CDN
└── README.md                    # 本文件
```

## 功能介绍

### Phase 1 — 关键词搜索引擎（`/v1`）

基于 **倒排索引 + TF-IDF** 的全文检索系统。

- **HTML 解析**：使用 BeautifulSoup 提取正文，自动去除 `<script>` 和 `<style>` 标签
- **中文分词**：使用 jieba 分词，支持中英文混合检索
- **TF-IDF 评分**：按词频-逆文档频率计算相关性得分
- **片段高亮**：搜索结果自动生成包含匹配关键词的上下文片段

| API | 方法 | 说明 |
|-----|------|------|
| `/v1` | GET | 搜索页面 |
| `/v1/search?q={query}` | GET | 关键词搜索 |
| `/v1/documents` | POST | 添加/更新文档 |

### Phase 2 — 语义搜索引擎（`/v2`）

基于 **LLM 语义理解** 的智能搜索，即使关键词不完全匹配也能找到相关文档。

- **两阶段检索**：先用关键词搜索获取候选集，再用 DeepSeek LLM 进行语义重排序
- **意图理解**：LLM 判断文档与查询在语义层面的相关程度，而非简单的关键词匹配
- **结果解释**：每条结果附带 LLM 给出的相关性理由

| API | 方法 | 说明 |
|-----|------|------|
| `/v2` | GET | 语义搜索页面 |
| `/v2/search?q={query}` | GET | 语义搜索 |

### Phase 3 — On-Call Agent 助手（`/v3`）

基于 **自实现 ReAct 框架** 的智能 Agent，通过对话方式帮助值班工程师处理故障。

- **ReAct 循环**：Thought（思考）→ Action（调用工具）→ Observation（观察结果）→ 循环直到得出答案
- **readFile 工具**：Agent 只能按文件名读取 SOP 文档，不能列目录或使用通配符
- **流式展示**：通过 SSE（Server-Sent Events）实时展示 Agent 的思考过程、工具调用和文档内容
- **多文档综合**：Agent 可连续查阅多个文档，综合给出完整建议

| API | 方法 | 说明 |
|-----|------|------|
| `/v3` | GET | Agent 对话页面 |
| `/v3/chat` | POST | 发送消息（SSE 流式响应） |

## 快速开始

### 环境要求

- Python 3.11+
- DeepSeek API Key（Phase 2 和 Phase 3 需要，Phase 1 不需要）

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

依赖列表：

| 包名 | 用途 |
|------|------|
| `fastapi` | Web 框架 |
| `uvicorn` | ASGI 服务器 |
| `httpx` | 异步 HTTP 客户端（调用 DeepSeek API） |
| `beautifulsoup4` | HTML 解析 |
| `pydantic` | 数据校验 |
| `jieba` | 中文分词 |
| `numpy` | 数值计算 |
| `python-multipart` | 表单数据支持 |

### 2. 配置 API Key

```bash
# Linux / macOS
export DEEPSEEK_API_KEY="sk-xxxxxxxxxxxxxxxx"

# Windows (PowerShell)
$env:DEEPSEEK_API_KEY="sk-xxxxxxxxxxxxxxxx"

# Windows (CMD)
set DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
```

### 3. 启动服务

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. 打开浏览器

```
http://localhost:8000        # 默认进入 Phase 1 关键词搜索
http://localhost:8000/v1     # Phase 1：关键词搜索引擎
http://localhost:8000/v2     # Phase 2：语义搜索引擎
http://localhost:8000/v3     # Phase 3：Agent 助手
```

## 使用示例

### Phase 1 — 关键词搜索

在搜索框中输入技术关键词：

- `OOM` — 返回后端服务和移动端的 OOM 处理 SOP
- `故障` — 返回大部分 SOP 文档（几乎所有文档都涉及故障处理）
- `CDN` — 返回前端和网络 CDN 的 SOP
- `replication` — 返回空（该词仅在 `<script>` 标签中出现，已被过滤）

### Phase 2 — 语义搜索

用自然语言描述问题：

- `"服务器挂了怎么办"` — 后端服务（sop-001）和 SRE（sop-004）靠前
- `"黑客攻击"` — 信息安全（sop-005）靠前
- `"机器学习模型出问题"` — AI 算法（sop-008）靠前

### Phase 3 — Agent 助手

在对话框中用自然语言提问，Agent 会自动查阅相关 SOP 文档：

- `"数据库主从延迟超过30秒怎么处理？"` → Agent 读取 sop-002，给出处理步骤
- `"服务 OOM 了怎么办？"` → Agent 读取 sop-001，给出排查建议
- `"怀疑有人入侵了系统"` → Agent 读取 sop-005，给出安全事件响应流程
- `"P0 故障的响应流程是什么？"` → Agent 综合多个 SOP 给出完整回答

Agent 的每一步思考、文件读取和观察结果都会实时展示在前端界面上。

## 项目设计说明

### ReAct Agent 框架

Agent 实现了标准的 ReAct（Reasoning + Acting）模式：

```
用户问题 → Thought（思考需要查哪个文档）
         → Action: readFile("sop-001")
         → Observation（获取文档内容）
         → Thought（分析内容，决定是否需要更多文档）
         → Action: readFile("sop-004")
         → Observation（获取更多内容）
         → Thought（信息充足，可以回答）
         → Final Answer（给出完整建议）
```

**特点**：
- 纯自实现，不依赖 LangChain 等第三方 Agent 框架
- 通过提示词工程约束 LLM 输出格式
- 正则解析 LLM 响应，支持多种输出格式
- 最大 8 轮迭代，防止无限循环
- 工具调用过程通过 SSE 实时推送到前端

### 搜索引擎设计

**倒排索引**：`token → {doc_id → [positions]}`，支持快速检索和片段定位。

**TF-IDF 评分**：
- TF（词频）：词在文档中出现的频率
- IDF（逆文档频率）：衡量词的区分度
- 使用余弦相似度计算查询与文档的相关性

**语义重排序**（Phase 2）：
- 第一轮：关键词搜索获取候选（扩大召回）
- 第二轮：DeepSeek LLM 逐条评估语义相关性（精准排序）
- 结果过滤低分文档，按 LLM 评分降序排列
