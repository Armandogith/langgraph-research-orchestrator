import os, asyncio
from dotenv import load_dotenv
from graph.workflow import clinical_app
load_dotenv()

async def main():
    query = "Latest evidence of semaglutide for obesity in CKD patients?"
    config = {"configurable": {"thread_id": "1"}}
    
    async for event in clinical_app.astream({"query": query}, config=config):
        print(f"--- Executing: {list(event.keys())[0]} ---")

    state = await clinical_app.aget_state(config)
    print(f"\nFeedback do Crítico: {state.values.get('critic_feedback')}")
    
    ans = input("\nAprovar pesquisa e gerar relatório? (y/n): ").lower()
    if ans == 'y':
        async for event in clinical_app.astream(None, config=config):
            if "writer" in event:
                report = event["writer"]["final_report"]
                with open("clinical_report.md", "w") as f: f.write(report)
                print("\n✅ Relatório gerado: clinical_report.md")

if __name__ == "__main__":
    asyncio.run(main())