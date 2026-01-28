"""Coding workflow using LangGraph StateGraph."""

from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage


class CodingState(TypedDict):
    """State for the coding workflow."""
    messages: Annotated[List[BaseMessage], "All messages in the conversation"]
    task: str
    analysis: str
    code: str
    test_result: str
    review: str


def create_coding_workflow():
    """Create the coding workflow graph."""

    # Define the nodes
    def analyze_node(state: CodingState) -> CodingState:
        """Analyze the coding task."""
        return {
            **state,
            "analysis": f"Analysis of task: {state['task']}",
            "messages": state["messages"] + [HumanMessage(content="Task analyzed")],
        }

    def implement_node(state: CodingState) -> CodingState:
        """Implement the code."""
        task = state["task"]
        return {
            **state,
            "code": f"# Implementation for: {task}\nprint('Hello, World!')",
            "messages": state["messages"] + [HumanMessage(content="Code implemented")],
        }

    def test_node(state: CodingState) -> CodingState:
        """Test the code."""
        return {
            **state,
            "test_result": "Tests passed",
            "messages": state["messages"] + [HumanMessage(content="Tests passed")],
        }

    def review_node(state: CodingState) -> CodingState:
        """Review the code."""
        return {
            **state,
            "review": f"Code review:\n- Implementation: {state['code']}\n- Tests: {state['test_result']}",
            "messages": state["messages"] + [HumanMessage(content="Code reviewed")],
        }

    # Create the workflow
    workflow = StateGraph(CodingState)

    # Add nodes
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("implement", implement_node)
    workflow.add_node("test", test_node)
    workflow.add_node("review", review_node)

    # Set up edges
    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "implement")
    workflow.add_edge("implement", "test")
    workflow.add_edge("test", "review")
    workflow.add_edge("review", END)

    # Compile the graph
    return workflow.compile()


if __name__ == "__main__":
    print("Testing Coding Workflow...")
    wf = create_coding_workflow()
    print("Coding workflow created successfully!")
    print(f"Type: {type(wf)}")
    result = wf.invoke({"messages": [], "task": "test task", "analysis": "", "code": "", "test_result": "", "review": ""})
    print(f"Result keys: {result.keys()}")

