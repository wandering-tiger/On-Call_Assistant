"""Document store with HTML parsing, inverted index, and TF-IDF scoring."""

import os
import re
import math
from collections import defaultdict
from typing import Optional

from bs4 import BeautifulSoup
import jieba

from config import DATA_DIR

# ── HTML text extraction ──────────────────────────────────────────────

def extract_text_from_html(html: str) -> str:
    """Extract visible text from HTML, excluding <script> and <style> content."""
    soup = BeautifulSoup(html, "html.parser")
    # Remove script and style elements completely
    for tag in soup(["script", "style", "noscript", "meta", "link"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_title_from_html(html: str) -> str:
    """Extract the <title> or <h1> from an HTML document."""
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("title")
    if title_tag and title_tag.get_text(strip=True):
        return title_tag.get_text(strip=True)
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)
    return "Untitled"


# ── Tokenization ──────────────────────────────────────────────────────

# Characters that should be indexed as standalone tokens
SPECIAL_CHARS = set("&<>|!@#$%^(){}[].,;:?/\\-+=_'\"`~*")


def tokenize(text: str) -> list[str]:
    """
    Tokenize Chinese + English mixed text.
    Uses jieba for Chinese word segmentation, then splits English words/punctuation.
    """
    tokens = []
    # First pass: jieba cut for Chinese text
    cut_result = list(jieba.cut(text))

    for token in cut_result:
        token = token.strip()
        if not token:
            continue
        # If token contains mixed content (English + Chinese), split further
        # Keep special characters as individual tokens
        # Split on whitespace first
        parts = re.split(r"(\s+)", token)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            # Split Chinese characters from non-Chinese for better indexing
            sub_tokens = _split_mixed(part)
            tokens.extend(sub_tokens)

    return [t for t in tokens if t]


def _split_mixed(text: str) -> list[str]:
    """Split mixed Chinese/English/numbers/punctuation into tokens."""
    result = []
    # Patterns for different character types
    buf = ""
    for ch in text:
        if ch in SPECIAL_CHARS:
            if buf:
                result.append(buf.lower())
                buf = ""
            result.append(ch)
        elif "一" <= ch <= "鿿" or "㐀" <= ch <= "䶿":
            # Chinese character
            if buf and not _is_cjk(buf[0]):
                result.append(buf.lower())
                buf = ""
            buf += ch
        elif ch.isalnum():
            if buf and _is_cjk(buf[0]):
                result.append(buf.lower())
                buf = ""
            buf += ch
        else:
            if buf:
                result.append(buf.lower())
                buf = ""
            if not ch.isspace():
                result.append(ch)
    if buf:
        result.append(buf.lower())
    return result


def _is_cjk(ch: str) -> bool:
    """Check if character is CJK."""
    cp = ord(ch)
    return (
        (0x4E00 <= cp <= 0x9FFF)
        or (0x3400 <= cp <= 0x4DBF)
        or (0x20000 <= cp <= 0x2A6DF)
        or (0x2A700 <= cp <= 0x2B73F)
        or (0x2B740 <= cp <= 0x2B81F)
        or (0x2B820 <= cp <= 0x2CEAF)
        or (0xF900 <= cp <= 0xFAFF)
        or (0x2F800 <= cp <= 0x2FA1F)
    )


# ── Inverted Index ────────────────────────────────────────────────────

class DocumentStore:
    """
    In-memory document store with inverted index and TF-IDF scoring.
    Supports keyword search with snippet generation.

    Performance characteristics for 100 documents:
    - Inverted index lookup: O(k) where k = number of unique query tokens
    - TF-IDF scoring: O(k × m) where m = candidate docs matched
    - IDF values are pre-computed and cached for fast repeated queries
    - All data is in-memory for sub-millisecond response times
    """

    def __init__(self):
        self.documents: dict[str, dict] = {}  # doc_id -> {id, title, html, text, tokens}
        # Inverted index: token -> {doc_id -> [positions]}
        self.inverted_index: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        # Pre-computed IDF cache: token -> idf_value
        self._idf_cache: dict[str, float] = {}
        self.doc_count = 0
        self._load_all()

    def _load_all(self):
        """Load all HTML files from the data directory."""
        if not os.path.isdir(DATA_DIR):
            return
        for fname in sorted(os.listdir(DATA_DIR)):
            if fname.endswith(".html"):
                doc_id = fname.replace(".html", "")
                fpath = os.path.join(DATA_DIR, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    html = f.read()
                self.add_document(doc_id, html)

    def add_document(self, doc_id: str, html: str) -> dict:
        """Add or update a document. Returns document metadata."""
        text = extract_text_from_html(html)
        title = extract_title_from_html(html)
        tokens = tokenize(text)

        # Build position map for this document
        positions_map: dict[str, list[int]] = defaultdict(list)
        for pos, token in enumerate(tokens):
            positions_map[token].append(pos)

        doc_entry = {
            "id": doc_id,
            "title": title,
            "html": html,
            "text": text,
            "tokens": tokens,
            "positions": dict(positions_map),
        }

        # Remove old index entries if updating
        if doc_id in self.documents:
            old_tokens = self.documents[doc_id]["tokens"]
            for token in set(old_tokens):
                if doc_id in self.inverted_index.get(token, {}):
                    del self.inverted_index[token][doc_id]
                    if not self.inverted_index[token]:
                        del self.inverted_index[token]
        else:
            self.doc_count += 1

        self.documents[doc_id] = doc_entry

        # Add to inverted index
        for token, positions in positions_map.items():
            self.inverted_index[token][doc_id] = positions
            # Invalidate IDF cache for this token
            self._idf_cache.pop(token, None)

        return {"id": doc_id, "title": title}

    def search(
        self,
        query: str,
        max_results: int = 20,
    ) -> list[dict]:
        """
        Keyword search using TF-IDF scoring.
        Returns list of {id, title, snippet, score}.
        """
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # Find candidate documents (any token match)
        candidate_docs: set[str] = set()
        for token in query_tokens:
            if token in self.inverted_index:
                candidate_docs.update(self.inverted_index[token].keys())

        if not candidate_docs:
            return []

        # Compute TF-IDF scores
        scores: dict[str, float] = {}
        for doc_id in candidate_docs:
            score = self._tfidf_score(query_tokens, doc_id)
            if score > 0:
                scores[doc_id] = score

        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked = ranked[:max_results]

        # Build results
        results = []
        for doc_id, score in ranked:
            doc = self.documents[doc_id]
            snippet = self._generate_snippet(doc, query_tokens)
            results.append({
                "id": doc_id,
                "title": doc["title"],
                "snippet": snippet,
                "score": round(score, 4),
            })

        return results

    def _get_idf(self, token: str) -> float:
        """Get IDF value for a token, using pre-computed cache."""
        if token not in self._idf_cache:
            df = len(self.inverted_index.get(token, {}))
            self._idf_cache[token] = math.log((self.doc_count + 1) / (df + 1)) + 1.0
        return self._idf_cache[token]

    def _tfidf_score(self, query_tokens: list[str], doc_id: str) -> float:
        """Compute TF-IDF cosine similarity score (with cached IDF)."""
        doc = self.documents[doc_id]
        doc_len = len(doc["tokens"])
        if doc_len == 0:
            return 0.0

        # Query TF vector (normalised)
        query_tf: dict[str, float] = defaultdict(float)
        for t in query_tokens:
            query_tf[t] += 1.0
        q_len = len(query_tokens)
        for t in query_tf:
            query_tf[t] /= q_len

        # Compute dot product + norms in one pass
        dot_product = 0.0
        query_norm = 0.0
        doc_norm = 0.0
        for t in query_tokens:
            idf = self._get_idf(t)
            qw = query_tf[t] * idf
            query_norm += qw * qw

            if t in doc["positions"]:
                dw = (len(doc["positions"][t]) / doc_len) * idf
            else:
                dw = 0.0
            doc_norm += dw * dw
            dot_product += qw * dw

        if query_norm == 0 or doc_norm == 0:
            return 0.0

        return dot_product / (math.sqrt(query_norm) * math.sqrt(doc_norm))

    def _generate_snippet(
        self, doc: dict, query_tokens: list[str], context: int = 60
    ) -> str:
        """Generate a snippet showing context around the first matching query token."""
        text = doc["text"]
        if not text:
            return ""

        # Find first match position in the raw text
        best_pos = -1
        best_token = ""
        for token in query_tokens:
            # Try case-insensitive matching in the raw text
            idx = text.lower().find(token.lower())
            if idx != -1 and (best_pos == -1 or idx < best_pos):
                best_pos = idx
                best_token = token

        if best_pos == -1:
            # Try position-based matching via tokens
            for token in query_tokens:
                if token in doc["positions"]:
                    # Find the raw text position for this token
                    positions = doc["positions"][token]
                    # Search for the token in text
                    idx = text.lower().find(token.lower())
                    if idx != -1:
                        best_pos = idx
                        best_token = token
                        break
            if best_pos == -1:
                # Return beginning of text
                return text[:context * 2] + ("..." if len(text) > context * 2 else "")

        # Extract context around the match
        start = max(0, best_pos - context)
        end = min(len(text), best_pos + len(best_token) + context)

        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(text) else ""

        snippet = text[start:end]
        # Highlight the match with markers
        match_start = best_pos - start
        match_end = match_start + len(best_token)

        highlighted = (
            snippet[:match_start]
            + "<mark>"
            + snippet[match_start:match_end]
            + "</mark>"
            + snippet[match_end:]
        )

        return prefix + highlighted + suffix

    def get_document_text(self, doc_id: str) -> Optional[str]:
        """Get the plain text of a document by ID."""
        doc = self.documents.get(doc_id)
        if doc:
            return doc["text"]
        return None

    def get_document_title(self, doc_id: str) -> Optional[str]:
        """Get the title of a document by ID."""
        doc = self.documents.get(doc_id)
        if doc:
            return doc["title"]
        return None

    def list_documents(self) -> list[dict]:
        """List all indexed documents."""
        return [
            {"id": doc["id"], "title": doc["title"]}
            for doc in self.documents.values()
        ]

    def get_compact_summary(self, doc_id: str, max_chars: int = 300) -> str:
        """
        Build a compact summary for semantic search ranking.
        Prioritises headings + first meaningful content paragraphs.
        With 100 docs × 300 chars ≈ 30K chars, this fits easily in 32K-token LLM windows.
        """
        doc = self.documents.get(doc_id)
        if not doc:
            return ""

        html = doc["html"]
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "meta", "link"]):
            tag.decompose()

        parts = []
        char_count = 0

        # Extract headings first (they carry the most semantic signal)
        for h in soup.find_all(["h1", "h2", "h3"]):
            text = h.get_text(strip=True)
            if text and text != doc["title"]:
                parts.append(text)
                char_count += len(text)
                if char_count >= max_chars:
                    return " | ".join(parts)[:max_chars]

        # Then add body paragraphs
        for p in soup.find_all("p"):
            text = p.get_text(strip=True)
            if text and len(text) > 15:  # skip very short/empty paragraphs
                parts.append(text)
                char_count += len(text)
                if char_count >= max_chars:
                    return " | ".join(parts)[:max_chars]

        return " | ".join(parts)[:max_chars]

    def get_document_sections(self, doc_id: str) -> list[dict]:
        """Split a document into sections by headings for semantic search."""
        doc = self.documents.get(doc_id)
        if not doc:
            return []

        html = doc["html"]
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "meta", "link"]):
            tag.decompose()

        sections = []
        current_heading = doc["title"]
        current_text = []

        for element in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
            if element.name in ("h1", "h2", "h3", "h4"):
                if current_text:
                    sections.append({
                        "heading": current_heading,
                        "text": " ".join(current_text).strip(),
                    })
                    current_text = []
                current_heading = element.get_text(strip=True)
            else:
                text = element.get_text(strip=True)
                if text:
                    current_text.append(text)

        if current_text:
            sections.append({
                "heading": current_heading,
                "text": " ".join(current_text).strip(),
            })

        return sections


# Singleton instance
store = DocumentStore()
