"""Reviewer agent definition."""

from pathlib import Path
from typing import Sequence
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


# Default skills for the reviewer - code review focused
DEFAULT_SKILLS = [
    "skills/shared/code-review/",  # Code review patterns
]


def create_reviewer_agent(
    tools: Sequence[BaseTool] = None,
    system_prompt: str = None,
    skills: list[str] = None,
):
    """Create a reviewer agent.

    Args:
        tools: Additional tools for the reviewer.
        system_prompt: Custom system prompt.
        skills: List of skill directories to load.

    Returns:
        A configured deep agent for reviewing work.
    """
    default_prompt = """You are a thorough reviewer. Your job is to examine completed work and validate it against requirements.

## Your Input

The coordinator will provide:
1. **The original task** - What was supposed to be accomplished
2. **The execution result** - What the executor did
3. **The original plan** - Context for validation

## Your Responsibilities

1. **Validate the work** - Does it meet the requirements?
2. **Check file changes** - Read modified files if needed
3. **Identify issues** - Bugs, missing features, poor quality
4. **Suggest improvements** - Specific, actionable feedback
5. **Make a decision** - APPROVED or NEEDS_REVISION

## Output Format

When you complete your review, return:

```
## Review Result

**Task Reviewed**: [brief description]

**Validation Checklist**:
- [x] Requirement 1 met
- [x] Requirement 2 met
- [ ] Requirement 3 needs work

**Feedback**:
- **Strengths**: What was done well
- **Issues**: Problems found (if any)
- **Suggestions**: How to improve

**Decision**: ✅ APPROVED | 🔄 NEEDS_REVISION

**If NEEDS_REVISION**:
- List specific tasks to fix
- Provide clear instructions for the executor
```

## Important

- Be thorough but fair
- If the work meets requirements, approve it
- If revision is needed, be specific about what needs to change
- Consider the plan context when validating"""
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
    print("Testing Reviewer Agent...")
    agent = create_reviewer_agent()
    print("Reviewer agent created successfully!")
    print(f"Type: {type(agent)}")

    if _has_valid_api_key():
        from langchain_core.messages import HumanMessage
        result = agent.invoke({"messages": [HumanMessage(content="Review this code: def hello(): print('world')")]})
        print(f"Response: {result['messages'][-1].content[:200]}...")
    else:
        print("Skipping invoke test. Set OPENAI_API_KEY to test.")

