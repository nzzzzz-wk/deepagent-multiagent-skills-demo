# DeepAgents Multi-Agent Demo

A demonstration of multi-agent collaboration using the [deepagents](https://github.com/langchain-ai/deepagents) framework, featuring:

- **Main Coordinator Agent** - Orchestrates task delegation
- **Subagent System** - Planner, Executor, Reviewer team
- **LangGraph Workflows** - Pre-built research and coding workflows
- **Context Isolation** - Each agent has its own context window

## Architecture

```
User Request → Coordinator Agent
    ├── Subagents:
    │   ├── Planner → Analyzes and plans
    │   ├── Executor → Carries out work
    │   └── Reviewer → Validates results
    │
    └── Workflows:
        ├── research-workflow → search → analyze → summarize
        └── coding-workflow → analyze → implement → test → review
```

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- Anthropic API Key (for Claude model)

## Installation

1. Clone or create the project:
```bash
cd deepagents-demo
```

2. Install dependencies:
```bash
uv sync
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

## Usage

### Interactive Demo

Run the interactive CLI:
```bash
uv run python main.py
```

Commands:
- `/research <topic>` - Run research workflow
- `/code <task>` - Run coding workflow
- `/ask <question>` - Ask the agent team
- `/quit` - Exit

### Simple Test

Run a quick test:
```bash
uv run python main.py --test
```

## Project Structure

```
deepagents-demo/
├── pyproject.toml           # Project dependencies
├── .env.example             # Environment variables template
├── README.md                # This file
├── main.py                  # Entry point
├── agents/
│   ├── __init__.py          # Agent exports
│   ├── coordinator.py       # Main coordinator agent
│   ├── planner.py           # Planning agent
│   ├── executor.py          # Execution agent
│   ├── reviewer.py          # Review agent
│   └── workflows/           # LangGraph workflows
│       ├── __init__.py
│       ├── research_workflow.py
│       └── coding_workflow.py
└── tools/
    └── __init__.py          # Custom tools
```

## Customization

### Adding Custom Tools

Edit `tools/__init__.py` to add new tools:

```python
from langchain_core.tools import tool

@tool
def my_tool(param: str) -> str:
    """Description of what my tool does."""
    return f"Result: {param}"
```

### Creating New Agents

Create a new agent file in `agents/`:

```python
from deepagents import create_deep_agent

def create_my_agent():
    return create_deep_agent(
        system_prompt="You are a specialized agent...",
        tools=[...],
    )
```

### Adding LangGraph Workflows

Create a new workflow file in `agents/workflows/`:

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class MyState(TypedDict):
    messages:: dict

def create_my_workflow():
    list
    data workflow = StateGraph(MyState)
    # Add nodes and edges
    workflow.add_node("step1", lambda s: {...s, "data": {...}})
    workflow.set_entry_point("step1")
    workflow.add_edge("step1", END)
    return workflow.compile()
```

## Resources

- [DeepAgents Documentation](https://docs.langchain.com/oss/python/deepagents)
- [LangGraph Documentation](https://docs.langchain.com/oss/python/langgraph)
- [DeepAgents GitHub](https://github.com/langchain-ai/deepagents)

## License

MIT
