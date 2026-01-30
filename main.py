"""Multi-Agent Demo - Working version with qwen2.5/Ollama compatibility."""

import os
import sys
import uuid
from dotenv import load_dotenv
load_dotenv()


def multiline_input(prompt="You: ") -> str:
    """Read multi-line input, ending with empty line."""
    print(prompt, end="")
    lines = []
    for line in sys.stdin:
        line = line.rstrip('\n')
        if line == "":
            break
        lines.append(line)
    return "\n".join(lines)

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from config import get_model_settings
from agents.coordinator import create_coordinator_agent
from agents.workflows.research_workflow import create_research_workflow
from agents.workflows.coding_workflow import create_coding_workflow


def handle_interrupt(result, coordinator, config):
    """Handle human-in-the-loop interrupt."""
    if not result.get("__interrupt__"):
        return result

    interrupts = result["__interrupt__"][0].value
    action_requests = interrupts["action_requests"]
    review_configs = interrupts["review_configs"]

    # Create config map
    config_map = {cfg["action_name"]: cfg for cfg in review_configs}

    print("\n" + "=" * 50)
    print("  HUMAN-IN-THE-LOOP APPROVAL NEEDED")
    print("=" * 50)

    # Display pending actions
    for i, action in enumerate(action_requests, 1):
        review_config = config_map.get(action["name"], {})
        allowed = review_config.get("allowed_decisions", ["approve"])
        print(f"\n{i}. Tool: {action['name']}")
        print(f"   Args: {action.get('args', {})}")

    print("\n" + "-" * 50)
    print("Available decisions: approve, edit, reject")
    print("-" * 50)

    # Get user decisions
    decisions = []
    for action in action_requests:
        review_config = config_map.get(action["name"], {})
        allowed = review_config.get("allowed_decisions", ["approve"])

        while True:
            decision = input(f"\nDecision for {action['name']} [{', '.join(allowed)}]: ").strip().lower()

            if decision == "edit" and "edit" in allowed:
                # Edit arguments
                print(f"  Original args: {action.get('args', {})}")
                new_args = {}
                for key in action.get("args", {}).keys():
                    val = input(f"  New value for {key} (leave empty to keep): ").strip()
                    if val:
                        new_args[key] = val
                decisions.append({
                    "type": "edit",
                    "edited_action": {
                        "name": action["name"],
                        "args": new_args if new_args else action.get("args", {})
                    }
                })
                break
            elif decision in allowed:
                decisions.append({"type": decision})
                break
            else:
                print(f"Invalid decision. Choose from: {allowed}")

    # Resume execution
    print("\nResuming execution...")
    result = coordinator.invoke(
        Command(resume={"decisions": decisions}),
        config=config
    )

    return result


# ===== Main =====
def run_demo():
    print("=" * 50)
    print("  Multi-Agent Demo (qwen2.5)")
    print("=" * 50)

    settings = get_model_settings()
    if settings.api_key == "dummy" and not os.environ.get("OPENAI_API_KEY"):
        print("Set: OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL")
        return

    print(f"Model: {settings.model}")
    print()

    # Create checkpointer for state persistence
    checkpointer = MemorySaver()

    # Create coordinator with checkpointer
    coordinator = create_coordinator_agent(checkpointer=checkpointer)

    # Also create workflows for direct use
    research_wf = create_research_workflow()
    coding_wf = create_coding_workflow()

    print("Commands: /ask <query>, /research <topic>, /code <task>, /quit")
    print()

    # Reuse thread_id for multi-turn conversations
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    print(f"[Thread: {thread_id[:8]}...]")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["/quit", "/exit"]:
                print("Bye!")
                break

            if user_input.startswith("/research "):
                result = research_wf.invoke({"messages": [], "query": user_input[10:], "findings": ""})
                print(f"\n{result.get('findings', '')}\n")
            elif user_input.startswith("/code "):
                result = coding_wf.invoke({"messages": [], "task": user_input[6:], "code": ""})
                print(f"\n{result.get('code', '')}\n")
            else:
                # Use coordinator for general queries
                result = coordinator.invoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config
                )

                # Handle interrupts (write_file, execute, etc.)
                result = handle_interrupt(result, coordinator, config)

                # Check for plan approval
                last_msg = result['messages'][-1].content

                if "## APPROVAL NEEDED" in last_msg:
                    print("\n" + last_msg)
                    approval = input("\nYour decision (approve/edit/reject): ").strip().lower()

                    # Resume with user's decision
                    result = coordinator.invoke(
                        {"messages": [AIMessage(content=f"User decision: {approval}")]},
                        config=config
                    )

                    # Handle any new interrupts after decision
                    result = handle_interrupt(result, coordinator, config)

                print(f"\n{result['messages'][-1].content}\n")

        except KeyboardInterrupt:
            print("\nBye!")
            break


# ===== Stream Demo =====
def run_stream_demo():
    """Test streaming with planner agent."""
    from agents.planner import create_planner_agent

    settings = get_model_settings()
    if settings.api_key == "dummy" and not os.environ.get("OPENAI_API_KEY"):
        print("Set: OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL")
        return

    print("=" * 50)
    print("  Planner Agent - Stream Mode Demo")
    print("=" * 50)
    print()

    agent = create_planner_agent()
    query = input("Enter your request: ").strip()

    print("\n--- Stream Output ---")
    sys.stdout.flush()
    for chunk in agent.stream({"messages": [HumanMessage(content=query)]}):
        # Content is in chunk['model']['messages'][-1]
        if "model" in chunk and "messages" in chunk["model"]:
            msg = chunk["model"]["messages"][-1]
            # msg can be dict or AIMessage object
            if isinstance(msg, dict):
                content = msg.get("content", "")
            else:
                content = msg.content if hasattr(msg, "content") else str(msg)
            if content:
                print(content, end="", flush=True)
                sys.stdout.flush()
    print("\n END-------")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--stream":
        run_stream_demo()
    else:
        run_demo()
