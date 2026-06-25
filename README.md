# Agentic Research Assistant

A tool-using AI agent built with **LangGraph** that autonomously decides whether
to query an internal knowledge base (RAG over a local vector store), search the
live web, or answer directly — based on the nature of the question, not hardcoded
rules.

This was built to demonstrate practical agentic AI patterns: explicit state
graphs, tool binding, conditional routing, and retrieval-augmented generation,
using a fast open-source LLM via Groq.

## Why this project

Most "RAG demos" hardcode a single retrieval path. This project instead gives
the LLM **two distinct tools** and lets it reason about which one (or both) a
given question actually needs — the same decision-making pattern used in
production multi-agent systems for tool use and delegation.

## Architecture

```
                ┌─────────────┐
   START ──────▶│  agent node │◀────────────┐
                └──────┬──────┘              │
                       │                     │
              tool call requested?           │
                 ┌─────┴─────┐               │
                 │           │               │
                YES          NO              │
                 │           │               │
                 ▼           ▼               │
         ┌─────────────┐   END        ┌──────┴──────┐
         │ tools node  │──────────────▶│  results    │
         │ (executes)  │               │  appended   │
         └─────────────┘               └─────────────┘
```

The graph loops between the **agent node** (calls the LLM with bound tools)
and the **tools node** (executes whichever tool the LLM requested) until the
LLM produces a final answer with no further tool calls. This is the classic
ReAct pattern expressed as an explicit, inspectable state machine rather than
a hidden chain.

### Tools available to the agent

| Tool | Purpose |
|---|---|
| `search_local_documents` | RAG retrieval over a FAISS vector store built from internal `.txt` documents (chunked with `RecursiveCharacterTextSplitter`, embedded with `all-MiniLM-L6-v2`) |
| `search_web` | Live web search via DuckDuckGo for current events / general knowledge outside the local knowledge base |

The LLM decides which tool(s) to call based on the system prompt's reasoning
rules — nothing is routed by keyword matching or regex.

## Tech Stack

- **LangGraph** — explicit agent state graph + conditional routing
- **LangChain** — tool abstraction, document loaders, text splitting
- **Groq (Llama 3.3 70B)** — fast, free-tier LLM inference
- **FAISS** — local vector store for semantic search
- **HuggingFace Sentence Transformers** — embedding generation (`all-MiniLM-L6-v2`)
- **DuckDuckGo Search** — live web retrieval, no API key required

## Setup

```bash
# 1. Clone and enter the project
git clone https://github.com/DWARAKAKOLIPAKULA/agentic-research-assistant.git
cd agentic-research-assistant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up your Groq API key (free at https://console.groq.com/keys)
cp .env.example .env
# then edit .env and paste your key

# 4. Run interactively
python main.py

# or run a single one-off query
python main.py --query "What is the equipment allowance for remote employees?"
```

## Example interaction

```
You: What is the equipment allowance for remote employees?

Assistant: Remote employees are eligible for a one-time equipment allowance
of up to $500 for home office setup (monitors, chairs, keyboards), plus a
$30/month internet reimbursement upon submitting a utility bill.
[Source: internal company policy handbook]

You: What is LangGraph used for?

Assistant: LangGraph is a framework for building stateful, multi-step AI
agents as explicit graphs, supporting cyclic reasoning loops, conditional
routing, and tool use...
[Source: web search]
```

Notice the agent chose **different tools** for each question without being
told which one to use — that decision happens entirely inside the `agent_node`.

## Project Structure

```
agentic-research-assistant/
├── agent.py            # LangGraph state graph, system prompt, agent/tool nodes
├── tools.py            # Tool definitions: search_local_documents, search_web
├── vector_store.py      # FAISS index builder + retriever (RAG pipeline)
├── main.py             # CLI entry point (interactive + single-query mode)
├── data/
│   └── company_policy.txt   # Sample document for the local knowledge base
├── requirements.txt
└── .env.example
```

## Design notes

- **State is explicit.** `AgentState` is a typed dict carrying the full
  message history; LangGraph's `add_messages` reducer appends rather than
  overwrites, so the agent retains context across tool calls.
- **Routing is conditional, not sequential.** `should_continue` inspects the
  last message for `tool_calls` and routes to either the tools node or `END`
  — the graph can loop through multiple tool calls before finishing.
- **Tools are model-agnostic.** Swapping `ChatGroq` for `ChatOpenAI` or
  `ChatAnthropic` requires changing one line; the graph and tools are
  unaffected.
- **The vector index is cached** to disk after first build, so repeated runs
  don't re-embed the documents.

