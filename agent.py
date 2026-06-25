"""
agent.py
--------
Defines the agentic graph using LangGraph.

Architecture
------------
This is a classic "ReAct-style" tool-using agent expressed as an explicit
state graph (rather than a hidden chain), which makes the reasoning loop
fully inspectable and testable:

    START -> agent_node -> (conditional) -> tools_node -> agent_node -> ...
                                |
                                +--> END (when no more tool calls needed)

The agent_node calls the LLM with the conversation history and bound tools.
If the LLM responds with one or more tool calls, control routes to
tools_node, which executes them and appends the results back into the
message history. The graph loops until the LLM produces a final answer
with no further tool calls.

This is the same pattern used in production multi-agent systems: the LLM
decides ON ITS OWN whether to query local documents, search the web, do
both, or answer directly from its own knowledge -- nothing is hardcoded.
"""

import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools import TOOLS

load_dotenv()

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

SYSTEM_PROMPT = """You are a research assistant with access to two tools:

1. search_local_documents - for questions about internal company policy
   (remote work rules, reimbursements, equipment allowance, etc.)
2. search_web - for current events, general knowledge, or anything outside
   the internal policy handbook.

Reasoning rules:
- Decide which tool(s) to use based on the nature of the question.
- If a question could be answered by either source, prefer the local
  documents first, then supplement with web search only if needed.
- If neither tool is relevant, answer directly from your own knowledge.
- Always cite which source(s) you used in your final answer.
- Be concise and accurate. Do not fabricate information not present in
  the tool results.
"""


class AgentState(TypedDict):
    """The shared state passed between graph nodes."""
    messages: Annotated[list[AnyMessage], add_messages]


def build_agent():
    """Constructs and compiles the LangGraph agent."""

    llm = ChatGroq(model=GROQ_MODEL, temperature=0)
    llm_with_tools = llm.bind_tools(TOOLS)

    def agent_node(state: AgentState) -> dict:
        """Calls the LLM with the current conversation + system prompt."""
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        """Routes to tools if the last AI message requested tool calls."""
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return END

    tool_node = ToolNode(TOOLS)

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")  # loop back after tool execution

    return graph.compile()


def run_query(question: str) -> str:
    """Convenience wrapper: run a single question through the agent."""
    app = build_agent()
    result = app.invoke({"messages": [("user", question)]})
    return result["messages"][-1].content


if __name__ == "__main__":
    # Quick manual test: python agent.py
    test_questions = [
        "What is the equipment allowance for remote employees?",
        "What is LangGraph used for?",
    ]
    for q in test_questions:
        print(f"\n{'='*60}\nQ: {q}\n{'='*60}")
        print(run_query(q))
