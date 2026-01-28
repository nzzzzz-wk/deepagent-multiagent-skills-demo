"""Research workflow using LangGraph StateGraph."""

from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool


class ResearchState(TypedDict):
    """State for the research workflow."""
    messages: Annotated[List[BaseMessage], "All messages in the conversation"]
    query: str
    research_data: Annotated[dict, "Research findings and data"]
    summary: str


@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    # Placeholder for actual web search
    return f"Search results for: {query}"


@tool
def analyze_findings(findings: str) -> str:
    """Analyze research findings."""
    return f"Analysis of findings: {findings}"


@tool
def summarize_research(data: dict) -> str:
    """Summarize the research results."""
    return f"Summary of research: {data.get('findings', 'No findings')}"


def create_research_workflow():
    """Create the research workflow graph."""

    # Define the nodes
    def search_node(state: ResearchState) -> ResearchState:
        """Search for information."""
        query = state["query"]
        result = search_web.invoke({"query": query})
        return {
            **state,
            "research_data": {"findings": result},
            "messages": state["messages"] + [HumanMessage(content=f"Searched: {query}")],
        }

    def analyze_node(state: ResearchState) -> ResearchState:
        """Analyze findings."""
        findings = state["research_data"].get("findings", "")
        result = analyze_findings.invoke({"findings": findings})
        return {
            **state,
            "research_data": {"findings": findings, "analysis": result},
            "messages": state["messages"] + [HumanMessage(content="Analysis complete")],
        }

    def summarize_node(state: ResearchState) -> ResearchState:
        """Summarize research."""
        summary = summarize_research.invoke({"data": state["research_data"]})
        return {
            **state,
            "summary": summary,
            "messages": state["messages"] + [HumanMessage(content="Research complete")],
        }

    # Create the workflow
    workflow = StateGraph(ResearchState)

    # Add nodes
    workflow.add_node("search", search_node)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("summarize", summarize_node)

    # Set up edges
    workflow.set_entry_point("search")
    workflow.add_edge("search", "analyze")
    workflow.add_edge("analyze", "summarize")
    workflow.add_edge("summarize", END)

    # Compile the graph
    return workflow.compile()


if __name__ == "__main__":
    print("Testing Research Workflow...")
    wf = create_research_workflow()
    print("Research workflow created successfully!")
    print(f"Type: {type(wf)}")
    result = wf.invoke({"messages": [], "query": "test query", "research_data": {}, "summary": ""})
    print(f"Result keys: {result.keys()}")

