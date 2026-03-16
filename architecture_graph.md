```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	Supervisor(Supervisor)
	Researcher(Researcher)
	Analyst(Analyst)
	Evaluator(Evaluator)
	__end__([<p>__end__</p>]):::last
	Analyst --> Supervisor;
	Evaluator --> Supervisor;
	Researcher --> Supervisor;
	Supervisor -. &nbsp;AgentName.ANALYST&nbsp; .-> Analyst;
	Supervisor -. &nbsp;AgentName.EVALUATOR&nbsp; .-> Evaluator;
	Supervisor -. &nbsp;AgentName.RESEARCHER&nbsp; .-> Researcher;
	Supervisor -. &nbsp;FINISH&nbsp; .-> __end__;
	__start__ --> Supervisor;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

```