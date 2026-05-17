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

class CriticOutput(BaseModel):
    is_sufficient: bool = Field(description="Data is enough?")
    feedback: str = Field(description="Feedback if not enough.")

def orchestrator_node(state: AgentState):
    print("🎯 Orchestrator: Iniciando pipeline...")
    return {"revision_count": 0, "research_data": [], "is_sufficient": False, "critic_feedback": ""}

async def researcher_node(state: AgentState):
    rev = state.get("revision_count", 0)
    print(f"🔬 Researcher: Rodada {rev + 1} de pesquisa...")

    if rev >= 3:
        print("⚠️  Máximo de revisões atingido.")
        return {"research_data": [{"warning": "Max revisions reached."}], "revision_count": rev + 1}

    llm_with_tools = llm_deepseek.bind_tools([tavily_search, pubmed_mock_search])
    prompt = f"Research query: {state['query']}\nPrevious feedback: {state['critic_feedback']}"

    res = llm_with_tools.invoke([
        SystemMessage(content="You are a Clinical Researcher. Use the tools provided to find medical evidence."),
        HumanMessage(content=prompt)
    ])

    new_data = []
    if res.tool_calls:
        for tc in res.tool_calls:
            print(f"   🔧 Usando ferramenta: {tc['name']}")
            if tc["name"] == "tavily_search":
                tool_result = tavily_search.invoke(tc["args"])
            elif tc["name"] == "pubmed_mock_search":
                tool_result = pubmed_mock_search.invoke(tc["args"])
            else:
                tool_result = []
            new_data.append({"tool": tc["name"], "results": tool_result})
    else:
        print("   ℹ️  Sem tool calls — usando resposta direta.")
        new_data.append({"tool": "direct", "results": res.content})

    return {"research_data": new_data, "revision_count": rev + 1}

async def critic_node(state: AgentState):
    print("🧐 Critic: Avaliando qualidade dos dados...")

    prompt = f"""You are a Medical Critic. Evaluate if this research is sufficient for the query.

Query: {state['query']}
Research Data: {state['research_data']}

Respond ONLY with a valid JSON object, no markdown, no explanation, no code blocks:
{{"is_sufficient": true, "feedback": "your feedback here"}}"""

    res = llm_deepseek.invoke([HumanMessage(content=prompt)])

    try:
        # Remove possíveis blocos de markdown caso o modelo retorne
        clean = res.content.strip().replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean)
        is_sufficient = parsed.get("is_sufficient", False)
        feedback = parsed.get("feedback", "")
        print(f"   ✅ Suficiente: {is_sufficient} | Feedback: {feedback[:80]}...")
        return {"is_sufficient": is_sufficient, "critic_feedback": feedback}
    except Exception as e:
        print(f"   ⚠️  Erro ao parsear JSON do Critic: {e}. Conteúdo: {res.content[:100]}")
        return {"is_sufficient": False, "critic_feedback": res.content}

async def writer_node(state: AgentState):
    print("✍️  Writer: Gerando relatório clínico...")

    llm_writer = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        max_tokens=4096,
        temperature=0.2
    )

    prompt = f"""You are a Senior Medical Writer. Write a complete clinical evidence report in Markdown.

Query: {state['query']}
Evidence collected: {state['research_data']}

Structure the report with:
# Executive Summary
# Evidence Analysis (with recommendation grades A/B/C)
# Comparative Table of Studies
# Limitations and Knowledge Gaps
# References"""

    res = llm_writer.invoke([
        SystemMessage(content="You are a Senior Medical Writer producing structured clinical reports."),
        HumanMessage(content=prompt)
    ])

    return {"final_report": res.content}