# Architecture Decision Record

## Why LangGraph?

LangGraph was chosen over CrewAI and AutoGen because it gives explicit control 
over the graph edges, state transitions, and interrupts. In a clinical context, 
predictability matters more than convenience.

CrewAI abstracts too much. LangGraph lets you see exactly what runs, when, and why.

## Why does the Critic exist?

Without the Critic, the Writer would act on whatever the Researcher brought back 
— regardless of quality, completeness, or contradictions.

The Critic creates a quality gate. It asks:
- Is the evidence recent enough?
- Are there contradictions between sources?
- Is the sample size sufficient?
- Is the specific population covered?

Only when the Critic returns `is_sufficient: true` does the graph move forward.
This prevents hallucinated or thin reports from reaching the human reviewer.

## Why Human-in-the-Loop before the Writer?

In clinical settings, no AI should generate a medical report without a human 
checkpoint. The `interrupt_before=["writer"]` pattern in LangGraph pauses the 
graph, presents the collected evidence to the user, and only resumes on explicit 
approval.

This is not just a UX feature — it is a safety design decision.

## Why operator.add on research_data?

```python