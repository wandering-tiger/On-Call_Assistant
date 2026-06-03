"""Phase 1: Keyword-based search engine using inverted index and TF-IDF."""

from engine.document_store import store


def keyword_search(query: str, max_results: int = 20) -> dict:
    """
    Perform keyword-based search.
    Returns {query, results: [{id, title, snippet, score}]}.
    """
    results = store.search(query, max_results=max_results)
    return {
        "query": query,
        "results": results,
    }


def add_document(doc_id: str, html: str) -> dict:
    """Add or update a document."""
    return store.add_document(doc_id, html)
