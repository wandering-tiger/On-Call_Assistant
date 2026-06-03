"""Application configuration.

LLM provider setup — choose one of three ways:

1. Set LLM_PROVIDER to a preset name (auto-fills base URL + model):
   export LLM_PROVIDER=deepseek   # default
   export LLM_PROVIDER=openai
   export LLM_PROVIDER=azure
   export LLM_PROVIDER=moonshot
   export LLM_PROVIDER=zhipu
   export LLM_PROVIDER=custom     # requires LLM_BASE_URL + LLM_CHAT_MODEL

2. Or set individual variables directly:
   export LLM_API_KEY=sk-xxx
   export LLM_BASE_URL=https://api.deepseek.com
   export LLM_CHAT_MODEL=deepseek-chat

3. Backward compatible — old DEEPSEEK_* vars still work if LLM_* are not set.
"""

import os

# ── Data & search config ──────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MAX_SEARCH_RESULTS = 20
DEFAULT_SNIPPET_CONTEXT = 60

# ── Provider presets ──────────────────────────────────────────────────

PROVIDER_PRESETS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "chat_model": "deepseek-chat",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "chat_model": "gpt-4o",
    },
    "azure": {
        # Azure requires: LLM_API_KEY, LLM_AZURE_ENDPOINT, LLM_AZURE_DEPLOYMENT
        "base_url": "",  # set via LLM_AZURE_ENDPOINT env var
        "chat_model": "",  # set via LLM_AZURE_DEPLOYMENT env var
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "chat_model": "moonshot-v1-8k",
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "chat_model": "glm-4",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "chat_model": "qwen-plus",
    },
    "custom": {
        "base_url": "",
        "chat_model": "",
    },
}

# ── Resolve LLM config ────────────────────────────────────────────────

_provider = os.environ.get("LLM_PROVIDER", "deepseek").lower()

if _provider in PROVIDER_PRESETS:
    _preset = PROVIDER_PRESETS[_provider]

    # Azure special handling
    if _provider == "azure":
        LLM_BASE_URL = os.environ.get("LLM_AZURE_ENDPOINT", "")
        LLM_CHAT_MODEL = os.environ.get("LLM_AZURE_DEPLOYMENT", "")
    else:
        LLM_BASE_URL = os.environ.get("LLM_BASE_URL", _preset["base_url"])
        LLM_CHAT_MODEL = os.environ.get("LLM_CHAT_MODEL", _preset["chat_model"])
else:
    # Unknown provider name — treat as custom, require explicit config
    LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "")
    LLM_CHAT_MODEL = os.environ.get("LLM_CHAT_MODEL", "")

# API key: LLM_API_KEY → DEEPSEEK_API_KEY (backward compat) → empty
LLM_API_KEY = (
    os.environ.get("LLM_API_KEY")
    or os.environ.get("DEEPSEEK_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
    or ""
)

# ── Backward-compatible aliases ───────────────────────────────────────

DEEPSEEK_API_KEY = LLM_API_KEY
DEEPSEEK_BASE_URL = LLM_BASE_URL
DEEPSEEK_CHAT_MODEL = LLM_CHAT_MODEL
