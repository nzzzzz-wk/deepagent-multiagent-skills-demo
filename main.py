"""Multi-Agent Demo - Working version with qwen2.5/Ollama compatibility."""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from config import get_model_settings
from agents.coordinator import create_coordinator_agent
from agents.workflows.research_workflow import create_research_workflow
from agents.workflows.coding_workflow import create_coding_workflow


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
                thread_id = str(id(user_input))
                config = {"configurable": {"thread_id": thread_id}}

                result = coordinator.invoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config
                )

                # Check for approval request marker
                last_msg = result['messages'][-1].content

                if "## APPROVAL NEEDED" in last_msg:
                    print("\n" + last_msg)
                    approval = input("\nYour decision (approve/edit/reject): ").strip().lower()

                    # Resume with user's decision
                    result = coordinator.invoke(
                        {"messages": [AIMessage(content=f"User decision: {approval}")]},
                        config=config
                    )

                print(f"\n{result['messages'][-1].content}\n")

        except KeyboardInterrupt:
            print("\nBye!")
            break


if __name__ == "__main__":
    run_demo()
