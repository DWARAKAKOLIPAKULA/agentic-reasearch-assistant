"""
tools.py
--------
Defines the two tools available to the agent:
1. search_local_documents - queries the FAISS vector store (RAG)
2. search_web - queries DuckDuckGo for live information

Each tool is wrapped with the @tool decorator so LangGraph's agent node
can bind them and the LLM can decide, on its own, which one to call.
"""

from langchain_core.tools import tool
from ddgs import DDGS
from vector_store import get_retriever

# Build the retriever once at import time (caches the FAISS index on disk)
_retriever = get_retriever(k=3)


@tool
def search_local_documents(query: str) -> str:
    """
    Search the internal company knowledge base (policy handbook) for
    information relevant to the query. Use this tool for questions about
    company policies, internal guidelines, remote work rules, reimbursement,
    or anything that sounds like it would be documented internally.
    """
    docs = _retriever.invoke(query)
    if not docs:
        return "No relevant internal documents found."

    formatted = "\n\n".join(
        f"[Source chunk {i+1}]\n{doc.page_content}" for i, doc in enumerate(docs)
    )
    return formatted


@tool
def search_web(query: str) -> str:
    """
    Search the live web for current, general-knowledge, or external
    information that would NOT be found in internal company documents.
    Use this for questions about current events, public facts, general
    knowledge, or anything outside the internal policy handbook's scope.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=4))
    except Exception as e:
        return f"Web search failed: {e}"

    if not results:
        return "No web results found."

    formatted = "\n\n".join(
        f"[{r.get('title', 'Untitled')}]\n{r.get('body', '')}\nSource: {r.get('href', '')}"
        for r in results
    )
    return formatted


TOOLS = [search_local_documents, search_web]


if __name__ == "__main__":
    # Quick manual test: python tools.py
    print("=== Testing local document search ===")
    print(search_local_documents.invoke("equipment allowance"))

    print("\n=== Testing web search ===")
    print(search_web.invoke("latest LangGraph release"))
