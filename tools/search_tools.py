from langchain_community.tools.tavily_search import TavilySearchAPIWrapper
from langchain_core.tools import tool
from typing import List, Dict, Any

@tool
def tavily_search(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """Busca na web via Tavily para dados clínicos e médicos."""
    try:
        search = TavilySearchAPIWrapper()
        return search.results(query, max_results=max_results)
    except: return []

@tool
def pubmed_mock_search(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """Simula busca no PubMed com artigos mockados."""
    mock_articles = [
        {"title": "Semaglutide for Obesity in CKD", "year": 2024, "conclusion": "Nephroprotective effects detected.", "keywords": ["semaglutide", "ckd"]},
        {"title": "GLP-1 RA Meta-analysis", "year": 2023, "conclusion": "15% reduction in kidney risk.", "keywords": ["glp-1", "ckd"]},
    ]
    query_lower = query.lower()
    return [a for article in mock_articles if any(kw in query_lower for kw in article["keywords"])][:max_results]