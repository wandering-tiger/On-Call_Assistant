"""Application configuration."""

import os

# DeepSeek API configuration
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "your-deepseek-api-key")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_CHAT_MODEL = os.environ.get("DEEPSEEK_CHAT_MODEL", "deepseek-chat")

# Data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Search configuration
MAX_SEARCH_RESULTS = 20
DEFAULT_SNIPPET_CONTEXT = 60  # characters around match for snippet
