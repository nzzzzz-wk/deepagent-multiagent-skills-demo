"""Executor agent definition."""

from pathlib import Path
from typing import Sequence, Any, Dict
from langchain_core.tools import BaseTool
from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from config import get_model_settings


# Get project root directory for filesystem backend
PROJECT_ROOT = Path(__file__).parent.parent


def get_openai_model():
    """Get configured OpenAI-compatible model from environment variables."""
    settings = get_model_settings()
    return settings.create_model()


def get_filesystem_backend():
    """Get a FilesystemBackend instance configured with the project root."""
    return FilesystemBackend(root_dir=str(PROJECT_ROOT), virtual_mode=True)


# Default skills for the executor - skills relevant to code execution and testing
DEFAULT_SKILLS = [
    "skills/web-dev/",            # Web development patterns
    "skills/shared/testing/",     # Testing patterns
    "skills/shared/code-review/", # Code review patterns (for self-review)
]


def create_executor_agent(
    tools: Sequence[BaseTool] = None,
    system_prompt: str = None,
    skills: list[str] = None,
):
    """Create an executor agent.

    Args:
        tools: Additional tools for the executor.
        system_prompt: Custom system prompt.
        skills: List of skill directories to load.

    Returns:
        A configured deep agent for executing tasks.
    """
    default_prompt = """You are a skilled executor. You carry out specific tasks from a larger plan.

## Your Input

The coordinator will provide you with:
1. **The overall plan** - Context about what we're building
2. **A specific task** - The exact step you need to execute
3. **Context from completed tasks** - What has already been done

## Your Responsibilities

1. **Execute the specific task** - Focus only on what you're asked to do
2. **Follow the plan** - Use the plan as reference, don't deviate
3. **Use available tools** - Read files, write code, run commands as needed
4. **Document your work** - Save outputs to files for later review
5. **Return a summary** - What you did, what was created, what files were modified

## Output Format

When you complete a task, return a summary in this format:

```
## Execution Result

**Task Completed**: [brief description]

**Files Modified/Created**:
- `/path/to/file1` - Description of changes
- `/path/to/file2` - New file created

**Summary**:
- What was accomplished
- Any decisions made
- Next steps or dependencies for other tasks

**Status**: ✅ Complete
```

## Important

- Don't re-plan - just execute the task you're given
- If the plan is unclear, ask for clarification via your response
- Save intermediate outputs to files so reviewer can check them
- Return only the summary to the coordinator (keep context clean)"""
    return create_deep_agent(
        model=get_openai_model(),
        tools=tools or [],
        system_prompt=system_prompt or default_prompt,
        backend=get_filesystem_backend(),
        skills=skills or DEFAULT_SKILLS,
    )


def _has_valid_api_key() -> bool:
    """Check if a valid API key is configured."""
    settings = get_model_settings()
    return bool(settings.api_key and settings.api_key != "dummy")


if __name__ == "__main__":
    print("Testing Executor Agent...")
    agent = create_executor_agent()
    print("Executor agent created successfully!")
    print(f"Type: {type(agent)}")

    if _has_valid_api_key():
        from langchain_core.messages import HumanMessage
        result = agent.invoke({"messages": [HumanMessage(content="List the files in the current directory.")]})
        print(f"Response: {result['messages'][-1].content[:200]}...")
    else:
        print("Skipping invoke test. Set OPENAI_API_KEY to test.")

