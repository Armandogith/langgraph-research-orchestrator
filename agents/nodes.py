import os
import json
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from graph.state import AgentState
from tools.search_tools import tavily_search, pubmed_mock_search
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURAÇÃO GLOBAL DO DEEPSEEK ---
llm_deepseek = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    max_tokens=2048,
    temperature=0
)
# ---------------------------------------

class CriticOutput(BaseModel):
    is_sufficient: bool = Field(description="Data is enough?")
    feedback: str = Field(description="Feedback if not enough.")

def orchestrator_node(state: AgentState):
    return {"revision_count": 0, "research_data": [], "is_sufficient": False, "critic_feedback": ""}

async def researcher_node(state: AgentState):
    rev = state.get("revision_count", 0)
    if rev >= 3: 
        return {"research_data": [{"warning": "Max revisions reached."}], "revision_count": rev + 1}
    
    # Usando o DeepSeek com as ferramentas
    llm_with_tools = llm_deepseek.bind_tools([tavily_search, pubmed_mock_search])
    prompt = f"Research query: {state['query']}\nFeedback: {state['critic_feedback']}"
    
    res = llm_with_tools.invoke([
        SystemMessage(content="You are a Clinical Researcher. Use the tools provided to find medical evidence."), 
        HumanMessage(content=prompt)
    ])
    
    new_data = []
    if res.tool_calls:
        for tc in res.tool_calls:
            # Pega a ferramenta correta e executa
            if tc["name"] == "tavily_search":
                tool_result = tavily_search.invoke(tc["args"])
            elif tc["name"] == "pubmed_mock_search":
                tool_result = pubmed_mock_search.invoke(tc["args"])
            else:
                tool_result = []
            
            new_data.append({"tool": tc["name"], "results": tool_result})
            
    return {"research_data": new_data, "revision_count": rev + 1}

async def critic_node(state: AgentState):
    # Usando o DeepSeek para saída estruturada Pydantic
    llm_critic = llm_deepseek.with_structured_output(CriticOutput)
    prompt = f"Evaluate if this research is sufficient for: {state['query']}\nData: {state['research_data']}"
    res = llm_critic.invoke([SystemMessage(content="You are a Medical Critic."), HumanMessage(content=prompt)])
    
    return {"is_sufficient": res.is_sufficient, "critic_feedback": res.feedback}

async def writer_node(state: AgentState):
    # DeepSeek para escrever o Markdown (com um pouco mais de criatividade/temperatura)
    llm_writer = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        temperature=0.2
    )
    prompt = f"Write a Markdown clinical report for: {state['query']}\nEvidence: {state['research_data']}"
    res = llm_writer.invoke([SystemMessage(content="You are a Medical Writer."), HumanMessage(content=prompt)])
    
    return {"final_report": res.content}