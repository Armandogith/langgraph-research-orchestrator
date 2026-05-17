from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from typing import List, Dict, Any

@tool
def tavily_search(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """Busca na web via Tavily para dados clínicos e médicos."""
    try:
        search = TavilySearch(max_results=max_results)
        result = search.invoke(query)
        return result if isinstance(result, list) else [{"content": str(result)}]
    except Exception as e:
        print(f"[tavily_search] Erro: {e}")
        return []

@tool
def pubmed_mock_search(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """Simula busca no PubMed com artigos mockados."""
    mock_articles = [
        {"title": "Semaglutide for Obesity in CKD", "year": 2024, "conclusion": "Nephroprotective effects detected.", "keywords": ["semaglutide", "ckd"]},
        {"title": "GLP-1 RA Meta-analysis", "year": 2023, "conclusion": "15% reduction in kidney risk.", "keywords": ["glp-1", "ckd"]},
        {"title": "GLP-1 Safety in Renal Impairment", "year": 2023, "conclusion": "No significant adverse renal events.", "keywords": ["glp-1", "renal", "semaglutide"]},
        {"title": "Obesity Management in CKD Patients", "year": 2024, "conclusion": "Weight loss improves GFR outcomes.", "keywords": ["obesity", "ckd"]},
        {"title": "Semaglutide Cardiovascular Outcomes", "year": 2022, "conclusion": "Reduced MACE in high-risk patients.", "keywords": ["semaglutide", "cardiovascular"]},
    ]
    query_lower = query.lower()
    filtered = [a for a in mock_articles if any(kw in query_lower for kw in a["keywords"])]
    return filtered[:max_results] if filtered else mock_articles[:max_results]