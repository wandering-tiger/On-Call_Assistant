"""Phase 3: ReAct Agent for On-Call assistance.

Implements a simple ReAct (Reasoning + Acting) framework:
- Thought → Action → Observation → Thought → ... → Final Answer
- Single tool: readFile(fname) to read SOP documents from data/ directory
"""

import os
import re
import json
from typing import AsyncGenerator, Optional

import httpx

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_CHAT_MODEL,
    DATA_DIR,
)
from engine.document_store import store

# ── System Prompt ─────────────────────────────────────────────────────

SYSTEM_PROMPT = """你是一个 On-Call 助手 Agent。你的任务是帮助值班工程师处理故障，通过查阅 SOP（标准操作流程）文档来提供准确的操作建议。

## 可用文档

以下是你可以查阅的 SOP 文档列表（按文件名读取）：

{doc_list}

## 工具

你只有一个工具：

- **readFile(filename)**: 读取 `data/` 目录下的 SOP 文档。参数 filename 是文件名（如 "sop-001" 或 "sop-001.html"），返回文档的完整文本内容。

## 工作方式

请严格按照以下 ReAct 格式进行思考和行动：

Thought: 分析用户问题，判断需要查阅哪个 SOP 文档。解释你的推理过程。
Action: readFile
Action Input: <文件名>

当收到文档内容后，继续思考：

Thought: 分析文档内容，判断是否需要查阅更多文档，或者是否已经可以给出答案。
Action: readFile
Action Input: <另一个文件名>

当你有足够信息回答用户问题时：

Thought: 我已经掌握了足够的信息来回答用户的问题。
Final Answer: <你的完整回答，引用相关 SOP 文档中的具体步骤和建议>

## 规则

1. 必须通过 readFile 工具查阅文档，不能凭空编造信息
2. 每次只读一个文件
3. 如果第一个文件没有完全回答问题，继续查阅其他相关文件
4. 回答要具体、可操作，引用文档中的关键步骤
5. 如果找不到相关信息，诚实告知用户
6. 回答使用中文，保持专业、清晰
7. 不要列出目录或尝试使用通配符——只能单个文件名读取
"""

USER_PROMPT_TEMPLATE = """用户问题：{query}

请按照 ReAct 格式开始分析。"""

# ── Response Parsing ───────────────────────────────────────────────────

def parse_react_response(text: str) -> dict:
    """
    Parse the LLM response to extract Thought, Action, Action Input, or Final Answer.
    Returns dict with type: 'action' or 'answer', plus relevant fields.
    """
    # Try to find Final Answer first
    final_answer_match = re.search(
        r"Final\s*Answer\s*[:：]\s*(.*?)$",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if final_answer_match:
        return {
            "type": "answer",
            "thought": _extract_thought(text),
            "answer": final_answer_match.group(1).strip(),
        }

    # Try to find Action
    action_match = re.search(
        r"Action\s*[:：]\s*readFile\s*\n\s*Action\s*Input\s*[:：]\s*(\S+)",
        text,
        re.IGNORECASE,
    )
    if action_match:
        return {
            "type": "action",
            "thought": _extract_thought(text),
            "action": "readFile",
            "action_input": action_match.group(1).strip().strip("'").strip('"'),
        }

    # Alternative format: Action: readFile("filename")
    alt_match = re.search(
        r'Action\s*[:：]\s*readFile\s*\(\s*["\']?([^"\')]+)["\']?\s*\)',
        text,
        re.IGNORECASE,
    )
    if alt_match:
        return {
            "type": "action",
            "thought": _extract_thought(text),
            "action": "readFile",
            "action_input": alt_match.group(1).strip(),
        }

    # If we can't parse, treat as thought-only (might need to prompt for action)
    thought = _extract_thought(text)
    if thought:
        return {
            "type": "thought_only",
            "thought": thought,
            "raw": text,
        }

    # Fallback: return raw text
    return {
        "type": "unknown",
        "raw": text,
    }


def _extract_thought(text: str) -> str:
    """Extract thought from response text."""
    match = re.search(
        r"Thought\s*[:：]\s*(.*?)(?=\n\s*(?:Action|Final\s*Answer|Thought)\s*[:：]|$)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return ""


# ── Tool Implementation ────────────────────────────────────────────────

def read_file(filename: str) -> str:
    """
    Read a SOP document from the data directory.
    Only allows reading by exact filename, no directory listing or wildcards.
    """
    # Normalize filename
    if not filename.endswith(".html"):
        filename = filename + ".html"

    # Security: prevent path traversal
    basename = os.path.basename(filename)
    if basename != filename:
        return f"错误：不允许路径遍历。只能指定文件名，不能包含路径分隔符。你尝试访问：{filename}"

    # Check for wildcards or directory listing attempts
    if any(c in filename for c in "*?[]"):
        return f"错误：不允许使用通配符。只能指定确切的文件名。"

    filepath = os.path.join(DATA_DIR, basename)

    if not os.path.isfile(filepath):
        # List available files for user
        available = [d["id"] for d in store.list_documents()]
        return f"错误：文件 '{basename}' 不存在。可用的 SOP 文件有：{', '.join(available)}"

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            html = f.read()

        from engine.document_store import extract_text_from_html

        text = extract_text_from_html(html)
        title = store.get_document_title(basename.replace(".html", "")) or basename

        return f"=== {title} ({basename}) ===\n\n{text}"

    except Exception as e:
        return f"读取文件时出错：{str(e)}"


# ── ReAct Agent Loop ───────────────────────────────────────────────────

class ReActAgent:
    """
    Simple ReAct agent for On-Call assistance.
    Uses DeepSeek chat API for reasoning and the readFile tool for document access.
    """

    def __init__(self):
        self._build_doc_list()

    def _build_doc_list(self):
        """Build the list of available documents for the system prompt."""
        docs = store.list_documents()
        lines = []
        for doc in docs:
            lines.append(f"- {doc['id']}.html: {doc['title']}")
        self.doc_list_str = "\n".join(lines)
        self.system_prompt = SYSTEM_PROMPT.format(doc_list=self.doc_list_str)

    async def chat(
        self,
        user_query: str,
        conversation_history: Optional[list[dict]] = None,
    ) -> AsyncGenerator[dict, None]:
        """
        Run the ReAct agent loop, yielding events:
        - {"type": "thought", "content": "..."}
        - {"type": "action", "action": "readFile", "input": "sop-001"}
        - {"type": "observation", "content": "..."}
        - {"type": "answer", "content": "..."}
        - {"type": "error", "content": "..."}
        - {"type": "done"}
        """
        # Build message list
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)

        # Add current user query
        messages.append({
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(query=user_query),
        })

        max_iterations = 8
        iteration = 0

        async with httpx.AsyncClient() as http_client:
            while iteration < max_iterations:
                iteration += 1

                # Call LLM
                response_text = await self._call_llm(http_client, messages)
                if response_text is None:
                    yield {"type": "error", "content": "LLM 调用失败，请检查 API 配置"}
                    yield {"type": "done"}
                    return

                # Parse response
                parsed = parse_react_response(response_text)

                # Yield thought
                if parsed.get("thought"):
                    yield {"type": "thought", "content": parsed["thought"]}

                if parsed["type"] == "answer":
                    yield {"type": "answer", "content": parsed["answer"]}
                    yield {"type": "done"}
                    return

                elif parsed["type"] == "action":
                    action = parsed["action"]
                    action_input = parsed["action_input"]

                    yield {
                        "type": "action",
                        "action": action,
                        "input": action_input,
                    }

                    # Execute tool
                    if action == "readFile":
                        observation = read_file(action_input)
                        yield {"type": "observation", "content": observation}

                        # Add assistant response + observation to messages
                        messages.append({
                            "role": "assistant",
                            "content": response_text,
                        })
                        messages.append({
                            "role": "user",
                            "content": f"Observation:\n{observation}\n\n请继续按照 ReAct 格式思考。如果信息足够，请给出 Final Answer。",
                        })
                    else:
                        yield {
                            "type": "error",
                            "content": f"未知工具：{action}",
                        }
                        yield {"type": "done"}
                        return

                elif parsed["type"] == "thought_only":
                    # Agent provided thought but no action/answer — prompt it
                    messages.append({
                        "role": "assistant",
                        "content": response_text,
                    })
                    messages.append({
                        "role": "user",
                        "content": "请继续：你需要调用 readFile 工具来查阅文档，或者如果已经确定答案，请给出 Final Answer。",
                    })

                else:
                    # Unknown format — try to prompt for proper format
                    messages.append({
                        "role": "assistant",
                        "content": response_text,
                    })
                    messages.append({
                        "role": "user",
                        "content": "请严格使用 ReAct 格式回复（Thought → Action → Observation → Final Answer）。如果你已确定答案，请以 'Final Answer:' 开头。",
                    })

        # Max iterations reached
        yield {
            "type": "answer",
            "content": "抱歉，处理过程超过了最大步骤限制。请尝试更具体地描述您的问题。",
        }
        yield {"type": "done"}

    async def _call_llm(
        self,
        client: httpx.AsyncClient,
        messages: list[dict],
    ) -> Optional[str]:
        """Call DeepSeek chat API."""
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }

        request = {
            "model": DEEPSEEK_CHAT_MODEL,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 4096,
            "stream": False,
        }

        try:
            response = await client.post(
                f"{DEEPSEEK_BASE_URL}/v1/chat/completions",
                headers=headers,
                json=request,
                timeout=120.0,
            )

            if response.status_code != 200:
                error_detail = response.text
                print(f"[Agent] API error: {response.status_code} - {error_detail}")
                return None

            data = response.json()
            return data["choices"][0]["message"]["content"]

        except httpx.TimeoutException:
            print("[Agent] API timeout")
            return None
        except Exception as e:
            print(f"[Agent] API call error: {e}")
            return None


# Singleton
agent = ReActAgent()
