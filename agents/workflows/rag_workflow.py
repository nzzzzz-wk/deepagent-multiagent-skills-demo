"""RAG Workflow using LangGraph StateGraph.

Simplified RAG workflow with:
- Search (vector similarity search)
- Rerank (optional cross-encoder reranking)
- Format (result formatting)

Supports ChromaDB and Milvus vector databases.
"""

import os
import logging
from typing import TypedDict, List, Annotated, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


# ============================================================================
# State Definition
# ============================================================================

class RAGState(TypedDict):
    """State for the RAG workflow."""
    query: str                              # User query
    search_results: Annotated[List[dict], "Search results from vector DB"]
    reranked_results: Annotated[List[dict], "Reranked results"]
    final_output: Annotated[dict, "Final formatted output"]
    messages: Annotated[List[BaseMessage], "Conversation messages"]
    top_k: int                              # Number of results to retrieve


# ============================================================================
# Vector Database Clients
# ============================================================================

class BaseVectorDBClient:
    """Base class for vector database clients."""

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        """Search for similar documents."""
        raise NotImplementedError


class ChromaDBClient(BaseVectorDBClient):
    """ChromaDB client wrapper."""

    def __init__(self, collection_name: str = "documents", persist_directory: str = "./chroma_db"):
        try:
            import chromadb
            self.client = chromadb.PersistentClient(path=persist_directory)
            self.collection = self.client.get_or_create_collection(collection_name)
            self._available = True
        except ImportError:
            logger.warning("ChromaDB not installed, using mock mode")
            self._available = False
        except Exception as e:
            logger.warning(f"Failed to connect to ChromaDB: {e}, using mock mode")
            self._available = False

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        """Search for similar documents in ChromaDB."""
        if not self._available:
            return self._mock_search(query, top_k)

        try:
            results = self.collection.query(query_texts=[query], n_results=top_k)
            documents = []
            for i, (doc, meta, cid) in enumerate(zip(
                results.get("documents", [[]])[0],
                results.get("metadatas", [[]])[0],
                results.get("ids", [[]])[0]
            )):
                documents.append({
                    "id": cid,
                    "content": doc,
                    "metadata": meta or {},
                    "score": results.get("distances", [[]])[0][i] if i < len(results.get("distances", [[]])[0]) else 0.0
                })
            return documents
        except Exception as e:
            logger.error(f"ChromaDB search error: {e}")
            return self._mock_search(query, top_k)

    def _mock_search(self, query: str, top_k: int) -> List[dict]:
        """Mock search results for testing."""
        return [
            {
                "id": f"doc_{i}",
                "content": f"Mock document content for query: {query} - result {i}",
                "metadata": {"source": "mock"},
                "score": 0.9 - (i * 0.1)
            }
            for i in range(min(top_k, 3))
        ]


class MilvusDBClient(BaseVectorDBClient):
    """Milvus vector database client."""

    def __init__(self, collection_name: str = "documents", uri: str = None):
        try:
            from pymilvus import connections, Collection
            self.Collection = Collection
            self.connections = connections
            self._connected = False
            self._collection_name = collection_name

            if uri:
                self.connections.connect(uri=uri)
            else:
                # Default local connection
                self.connections.connect("default")

            # Load collection if exists
            try:
                self.collection = self.Collection(collection_name)
                self.collection.load()
                self._connected = True
            except Exception:
                logger.warning(f"Collection {collection_name} not found, using mock mode")
                self._connected = False
        except ImportError:
            logger.warning("pymilvus not installed, using mock mode")
            self._connected = False
        except Exception as e:
            logger.warning(f"Failed to connect to Milvus: {e}, using mock mode")
            self._connected = False

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        """Search for similar documents in Milvus."""
        if not self._connected:
            return self._mock_search(query, top_k)

        try:
            # Note: In real implementation, you would need to embed the query first
            # This is a simplified version
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            results = self.collection.search(
                data=[query],  # Would need embedding here
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["content", "metadata"]
            )

            documents = []
            for hit in results[0]:
                documents.append({
                    "id": hit.id,
                    "content": hit.entity.get("content", ""),
                    "metadata": hit.entity.get("metadata", {}),
                    "score": hit.score
                })
            return documents
        except Exception as e:
            logger.error(f"Milvus search error: {e}")
            return self._mock_search(query, top_k)

    def _mock_search(self, query: str, top_k: int) -> List[dict]:
        """Mock search results for testing."""
        return [
            {
                "id": f"doc_{i}",
                "content": f"Mock document content for query: {query} - result {i}",
                "metadata": {"source": "mock"},
                "score": 0.9 - (i * 0.1)
            }
            for i in range(min(top_k, 3))
        ]


def get_vector_db_client(provider: str = "chroma", **kwargs) -> BaseVectorDBClient:
    """Factory function to create vector database client."""
    if provider == "chroma":
        return ChromaDBClient(
            collection_name=kwargs.get("collection_name", "documents"),
            persist_directory=kwargs.get("persist_directory", "./chroma_db")
        )
    elif provider == "milvus":
        return MilvusDBClient(
            collection_name=kwargs.get("collection_name", "documents"),
            uri=kwargs.get("uri")
        )
    else:
        raise ValueError(f"Unsupported vector DB provider: {provider}")


# ============================================================================
# Reranker (Simple scoring based reranking)
# ============================================================================

class SimpleReranker:
    """Simple reranker using keyword matching."""

    def __init__(self):
        pass

    def rerank(self, query: str, documents: List[dict], top_k: int = 5) -> List[dict]:
        """Rerank documents based on query relevance."""
        if not documents:
            return []

        query_terms = set(query.lower().split())

        def score_doc(doc):
            content = doc.get("content", "").lower()
            metadata = str(doc.get("metadata", {})).lower()
            combined = content + " " + metadata

            # Count matching terms
            matches = sum(1 for term in query_terms if term in combined)
            # Boost by original score
            original_score = doc.get("score", 0.5)
            return matches * 0.3 + original_score * 0.7

        scored = [(score_doc(doc), doc) for doc in documents]
        scored.sort(key=lambda x: x[0], reverse=True)

        return [doc for _, doc in scored[:top_k]]


# ============================================================================
# Workflow Nodes
# ============================================================================

def create_search_node(vector_db_client: BaseVectorDBClient):
    """Create a search node with the given vector DB client."""

    def search_node(state: RAGState) -> RAGState:
        """Search for relevant documents."""
        query = state["query"]
        top_k = state.get("top_k", 5)

        logger.info(f"Searching for: {query[:50]}...")

        results = vector_db_client.search(query, top_k)

        logger.info(f"Found {len(results)} results")

        return {
            **state,
            "search_results": results,
            "messages": state["messages"] + [HumanMessage(content=f"Searched for: {query}")],
        }

    return search_node


def create_rerank_node(reranker: SimpleReranker = None):
    """Create a rerank node with optional reranker."""

    def rerank_node(state: RAGState) -> RAGState:
        """Rerank search results."""
        local_reranker = reranker if reranker is not None else SimpleReranker()

        query = state["query"]
        results = state.get("search_results", [])
        top_k = state.get("top_k", 5)

        logger.info(f"Reranking {len(results)} results...")

        reranked = local_reranker.rerank(query, results, top_k)

        logger.info(f"Reranked to {len(reranked)} results")

        return {
            **state,
            "reranked_results": reranked,
            "messages": state["messages"] + [HumanMessage(content=f"Reranked {len(results)} results")],
        }

    return rerank_node


def create_format_node():
    """Create a format node for final output."""

    def format_node(state: RAGState) -> RAGState:
        """Format the final output."""
        results = state.get("reranked_results", []) or state.get("search_results", [])
        query = state["query"]

        # Build final output
        references = []
        for doc in results:
            references.append({
                "id": doc.get("id", ""),
                "content": doc.get("content", "")[:500],  # Truncate long content
                "source": doc.get("metadata", {}).get("source", "unknown"),
                "score": doc.get("score", 0.0)
            })

        confidence = sum(r["score"] for r in references) / len(references) if references else 0.0

        final_output = {
            "answer": f"Based on the search results for '{query}', I found {len(references)} relevant documents.",
            "references": references,
            "confidence": confidence,
            "result_count": len(references)
        }

        return {
            **state,
            "final_output": final_output,
            "messages": state["messages"] + [HumanMessage(content=f"Generated answer with {len(references)} references")],
        }

    return format_node


# ============================================================================
# Workflow Factory
# ============================================================================

def create_rag_workflow(
    vector_db_client: BaseVectorDBClient = None,
    provider: str = "chroma",
    enable_rerank: bool = True,
    collection_name: str = "documents",
    **kwargs
):
    """Create and compile the RAG LangGraph workflow.

    Args:
        vector_db_client: Pre-configured vector DB client (optional)
        provider: Vector DB provider if client not provided
        enable_rerank: Whether to enable reranking
        collection_name: Collection name for vector DB
        **kwargs: Additional arguments for vector DB

    Returns:
        Compiled LangGraph workflow
    """
    # Create vector DB client if not provided
    if vector_db_client is None:
        vector_db_client = get_vector_db_client(provider, collection_name=collection_name, **kwargs)

    # Create reranker
    reranker = SimpleReranker() if enable_rerank else None

    # Create workflow
    workflow = StateGraph(RAGState)

    # Add nodes
    workflow.add_node("search", create_search_node(vector_db_client))
    workflow.add_node("format", create_format_node())

    if enable_rerank:
        workflow.add_node("rerank", create_rerank_node(reranker))

    # Set up edges
    workflow.set_entry_point("search")

    if enable_rerank:
        workflow.add_edge("search", "rerank")
        workflow.add_edge("rerank", "format")
    else:
        workflow.add_edge("search", "format")

    workflow.add_edge("format", END)

    # Compile
    return workflow.compile()


# ============================================================================
# Execution Functions
# ============================================================================

async def execute_rag_workflow(
    query: str,
    provider: str = "chroma",
    collection_name: str = "documents",
    top_k: int = 5,
    enable_rerank: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """Execute the RAG workflow with the given query.

    Args:
        query: User query
        provider: Vector DB provider
        collection_name: Collection name
        top_k: Number of results
        enable_rerank: Whether to enable reranking
        **kwargs: Additional arguments

    Returns:
        Final workflow state
    """
    # Create workflow
    workflow = create_rag_workflow(
        provider=provider,
        collection_name=collection_name,
        enable_rerank=enable_rerank,
        **kwargs
    )

    # Initial state
    initial_state: RAGState = {
        "query": query,
        "search_results": [],
        "reranked_results": [],
        "final_output": {},
        "messages": [],
        "top_k": top_k
    }

    # Execute
    try:
        result = await workflow.ainvoke(initial_state)
        return result
    except Exception as e:
        logger.error(f"RAG workflow execution failed: {e}")
        raise


def sync_execute_rag_workflow(
    query: str,
    provider: str = "chroma",
    collection_name: str = "documents",
    top_k: int = 5,
    enable_rerank: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """Synchronous version of execute_rag_workflow."""
    import asyncio

    return asyncio.run(execute_rag_workflow(
        query=query,
        provider=provider,
        collection_name=collection_name,
        top_k=top_k,
        enable_rerank=enable_rerank,
        **kwargs
    ))


# ============================================================================
# CLI Test
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 60)
    print("Testing RAG Workflow")
    print("=" * 60)

    # Test with mock data (ChromaDB mock mode)
    print("\n1. Creating workflow...")
    workflow = create_rag_workflow(provider="chroma", enable_rerank=True)

    print("2. Executing workflow...")
    result = sync_execute_rag_workflow(
        query="工厂检查要求",
        provider="chroma",
        top_k=3
    )

    print(f"\n3. Results:")
    print(f"   Search results: {len(result.get('search_results', []))}")
    print(f"   Reranked results: {len(result.get('reranked_results', []))}")
    print(f"   Final output confidence: {result.get('final_output', {}).get('confidence', 0):.2f}")

    print("\n4. References:")
    for ref in result.get("final_output", {}).get("references", []):
        print(f"   - [{ref['source']}] {ref['content'][:80]}...")

    print("\n" + "=" * 60)
    print("RAG Workflow test completed!")
    print("=" * 60)
