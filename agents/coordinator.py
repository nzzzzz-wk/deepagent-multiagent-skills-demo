"""Coordinator agent - the main agent that orchestrates subagents."""

import os
from pathlib import Path
from typing import Sequence, Any, Dict
from langchain_core.tools import BaseTool, StructuredTool
from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from config import get_model_settings
from .planner import create_planner_agent, DEFAULT_SKILLS as PLANNER_SKILLS
from .executor import create_executor_agent
from .reviewer import create_reviewer_agent


# Get project root directory for filesystem backend
PROJECT_ROOT = Path(__file__).parent.parent.parent


def get_openai_model():
    """Get configured OpenAI-compatible model from environment variables."""
    settings = get_model_settings()
    return settings.create_model()


def get_filesystem_backend():
    """Get a FilesystemBackend instance configured with the project root."""
    return FilesystemBackend(root_dir=str(PROJECT_ROOT), virtual_mode=True)


def create_coordinator_agent(
    tools: Sequence[BaseTool] = None,
    system_prompt: str = None,
    skills: list[str] = None,
    checkpointer=None,
):
    """Create a coordinator agent that manages subagents.

    Args:
        tools: Additional tools for the coordinator.
        system_prompt: Custom system prompt.
        skills: List of skill directories for the planner subagent.
        checkpointer: LangGraph checkpointer for state persistence.

    Returns:
        A configured deep agent with subagent management capabilities.
    """
    from deepagents.middleware.subagents import CompiledSubAgent

    planner_skills = skills or PLANNER_SKILLS

    # Create actual agents with skills for subagents
    planner_agent = create_planner_agent(tools=tools, skills=planner_skills)
    executor_agent = create_executor_agent(tools=tools)
    reviewer_agent = create_reviewer_agent(tools=tools)

    # Define subagents using CompiledSubAgent
    planner_subagent = CompiledSubAgent(
        name="planner",
        description="Creates detailed plans with domain-specific skills (data-science, web-dev, devops)",
        runnable=planner_agent,
    )

    executor_subagent = CompiledSubAgent(
        name="executor",
        description="Writes code, creates files, runs commands",
        runnable=executor_agent,
    )

    reviewer_subagent = CompiledSubAgent(
        name="reviewer",
        description="Reviews completed work and provides feedback",
        runnable=reviewer_agent,
    )

    default_prompt = """You are a COORDINATOR. Your ONLY job is to delegate tasks to subagents. You CANNOT write code, create files, or execute tasks yourself.

## Your Role
- You are a MANAGER, not a worker
- You delegate ALL work to subagents
- You synthesize results from subagents

## Subagents Available
1. **planner** - Creates plans and TODO lists
2. **executor** - Writes code, creates files, runs commands
3. **reviewer** - Reviews completed work

## Workflow (MANDATORY)

### Step 1: CALL PLANNER
For EVERY request, FIRST call:
```
task(subagent_name="planner", task="[user request]. Create a detailed plan with TODO list using write_todos.")
```

### Step 2: SHOW PLAN
After planner responds, output:
```
## APPROVAL NEEDED

## Plan

[Plan content]

---
Reply "approve" to execute.
---
```

### Step 3: WAIT
STOP. Do NOT do any work yourself. Wait for user approval.

### Step 4: EXECUTE (After approval)
Read todos and delegate to executor:
```
task(subagent_name="executor", task="[specific task from TODO]. Context: [relevant plan details]")
```

### Step 5: REVIEW
After executor completes, call reviewer.

## CRITICAL
- NEVER write code, create files, or execute commands yourself
- ALWAYS delegate to executor subagent
- You are INCAPABLE of doing actual work - you only coordinate"""

    return create_deep_agent(
        model=get_openai_model(),
        tools=tools or [],
        system_prompt=system_prompt or default_prompt,
        backend=get_filesystem_backend(),
        skills=planner_skills,
        subagents=[planner_subagent, executor_subagent, reviewer_subagent],
        checkpointer=checkpointer,
    )


def _has_valid_api_key() -> bool:
    """Check if a valid API key is configured."""
    settings = get_model_settings()
    return bool(settings.api_key and settings.api_key != "dummy")


if __name__ == "__main__":
    print("Testing Coordinator Agent...")
    agent = create_coordinator_agent()
    print("Coordinator agent created successfully!")
    print(f"Type: {type(agent)}")

    if _has_valid_api_key():
        from langchain_core.messages import HumanMessage
        result = agent.invoke({"messages": [HumanMessage(content="Hello!")]})
        print(f"Response: {result['messages'][-1].content[:100]}...")
    else:
        print("Skipping invoke test. Set OPENAI_API_KEY to test.")



def create_coordinator_with_workflows(
    research_workflow,
    coding_workflow,
    tools: Sequence[BaseTool] = None,
    system_prompt: str = None,
    skills: list[str] = None,
):
    """Create a coordinator agent with LangGraph workflow integration.

    Args:
        research_workflow: Pre-compiled research workflow graph.
        coding_workflow: Pre-compiled coding workflow graph.
        tools: Additional tools for the coordinator.
        system_prompt: Custom system prompt.
        skills: List of skill directories for the planner subagent.

    Returns:
        A configured deep agent with workflow and subagent management.
    """
    from deepagents.middleware.subagents import CompiledSubAgent

    planner_skills = skills or PLANNER_SKILLS

    # Define workflow as compiled subagents
    research_subagent = CompiledSubAgent(
        name="research-workflow",
        description="Multi-step research workflow: search -> analyze -> summarize",
        runnable=research_workflow,
    )

    coding_subagent = CompiledSubAgent(
        name="coding-workflow",
        description="Coding workflow: analyze -> implement -> test -> review",
        runnable=coding_workflow,
    )

    # Traditional subagents with skills for planner
    planner_subagent = {
        "name": "planner",
        "description": "Used to analyze tasks and create detailed execution plans",
        "system_prompt": "You are an expert planner.",
        "tools": tools or [],
        "skills": planner_skills,
    }

    executor_subagent = {
        "name": "executor",
        "description": "Used to execute tasks according to the plan",
        "system_prompt": "You are a skilled executor.",
        "tools": tools or [],
    }

    reviewer_subagent = {
        "name": "reviewer",
        "description": "Used to review work products and provide feedback",
        "system_prompt": "You are a thorough reviewer.",
        "tools": tools or [],
    }

    default_prompt = """You are the main coordinator agent with access to both subagents and LangGraph workflows.

## Available Tools

### Workflows
- **research-workflow**: Multi-step research process (search -> analyze -> summarize)
- **coding-workflow**: Coding workflow (analyze -> implement -> test -> review)

### Subagents
- **planner**: Creates detailed execution plans (has domain-specific skills)
- **executor**: Carries out tasks
- **reviewer**: Validates and provides feedback

## Usage

For research tasks, use the research-workflow.
For coding tasks, use the coding-workflow.
For general tasks, use the subagent team (planner -> executor -> reviewer).

## Note
The planner has domain-specific skills (web-dev, data-science, devops) that activate based on the task type."""

    return create_deep_agent(
        model=get_openai_model(),
        tools=tools or [],
        system_prompt=system_prompt or default_prompt,
        backend=get_filesystem_backend(),
        skills=planner_skills,
        subagents=[research_subagent, coding_subagent, planner_subagent, executor_subagent, reviewer_subagent],
    )
