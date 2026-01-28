"""Planner agent definition."""

from pathlib import Path
from typing import Sequence
from langchain_core.tools import BaseTool
from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from config import get_model_settings


# Get project root directory for filesystem backend
PROJECT_ROOT = Path(__file__).parent.parent


def get_filesystem_backend():
    """Get a FilesystemBackend instance configured with the project root."""
    return FilesystemBackend(root_dir=str(PROJECT_ROOT), virtual_mode=True)

# Default skills for the planner - explicit paths for clarity
DEFAULT_SKILLS = [
    "skills/planning-core/",  # Core planning methodology
    "skills/web-dev/",        # Web development, APIs, databases
    "skills/data-science/",   # ML, data analysis, pipelines
    "skills/devops/",         # CI/CD, deployment, infrastructure
]


def get_openai_model():
    """Get configured OpenAI-compatible model from environment variables."""
    settings = get_model_settings()
    return settings.create_model()


def create_planner_agent(
    tools: Sequence[BaseTool] = None,
    system_prompt: str = None,
    skills: list[str] = None,
):
    """Create a planner agent.

    Args:
        tools: Additional tools for the planner.
        system_prompt: Custom system prompt.
        skills: List of skill directories to load. Each skill provides
                domain-specific planning patterns that activate based on
                the user query content.

    Returns:
        A configured deep agent for planning tasks.
    """
    default_prompt = """You are a pure TEXT-ONLY PLANNER. You have NO tools. You CANNOT write files, run code, or execute commands.

## Your Skills
You have access to domain-specific planning skills that activate based on the task:
- **planning-core**: Core planning methodology (always active)
- **web-dev**: Web development, APIs, databases
- **data-science**: ML, data analysis, pipelines
- **devops**: CI/CD, deployment, infrastructure

When planning, the appropriate skill activates based on your query content.

## Output Format:
Start your response with "## Plan: [Task Name]" followed by the detailed plan.
After the plan, include a TODO list using write_todos in this format:
```json
[
  {"id": "1", "content": "Task description", "status": "pending", "Active_Form": "Doing task"}
]
```

## CRITICAL - DO NOT:
- Do NOT use any tools (you have none)
- Do NOT write code or files
- Do NOT run tests or commands
- Do NOT create any files
- Do NOT call subagents
- ONLY output the plan text starting with "## Plan:"

If you're asked to create something, respond with ONLY a plan describing how to create it, nothing else."""

    # Use interrupt_on to require human approval before executing/running code
    return create_deep_agent(
        model=get_openai_model(),
        tools=tools or [],
        system_prompt=system_prompt or default_prompt,
        backend=get_filesystem_backend(),
        skills=skills or DEFAULT_SKILLS,
        interrupt_on={
            "execute": True,
            "write_file": True,
            "run_python": True,
        },
    )


def _has_valid_api_key() -> bool:
    """Check if a valid API key is configured."""
    settings = get_model_settings()
    return bool(settings.api_key and settings.api_key != "dummy")


if __name__ == "__main__":
    print("Testing Planner Agent...")
    agent = create_planner_agent()
    print("Planner agent created successfully!")
    print(f"Type: {type(agent)}")
    print(f"Skills: {DEFAULT_SKILLS}")

    if _has_valid_api_key():
        from langchain_core.messages import HumanMessage
        result = agent.invoke({"messages": [HumanMessage(content="Create a plan to build a web app.")]})
        print(f"Response: {result['messages'][-1].content[:200]}...")
    else:
        print("Skipping invoke test. Set OPENAI_API_KEY to test.")

