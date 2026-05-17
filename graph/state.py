from typing import TypedDict, List, Dict, Any, Annotated
import operator

class AgentState(TypedDict):
    query: str
    research_data: Annotated[List[Dict[str, Any]], operator.add]
    critic_feedback: str
    revision_count: int
    final_report: str
    is_sufficient: bool