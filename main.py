"""
main.py
-------
Interactive command-line entry point for the agentic research assistant.

Usage:
    python main.py
    python main.py --query "What is the remote work equipment allowance?"

Set your Groq API key first:
    export GROQ_API_KEY="your-key-here"
    (or put it in a .env file -- see .env.example)
"""

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def check_api_key() -> bool:
    if not os.environ.get("GROQ_API_KEY"):
        print(
            "ERROR: GROQ_API_KEY not found.\n"
            "Set it via: export GROQ_API_KEY='your-key-here'\n"
            "Or copy .env.example to .env and fill it in.\n"
            "Get a free key at https://console.groq.com/keys"
        )
        return False
    return True


def run_single_query(question: str):
    from agent import run_query

    print(f"\nQuestion: {question}\n")
    print("Thinking...\n")
    answer = run_query(question)
    print(f"Answer:\n{answer}\n")


def run_interactive():
    from agent import build_agent

    print("=" * 60)
    print("  Agentic Research Assistant  (LangGraph + Groq)")
    print("  Type 'exit' or 'quit' to stop.")
    print("=" * 60)

    app = build_agent()
    conversation = []

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if question.lower() in {"exit", "quit", ""}:
            print("Goodbye!")
            break

        conversation.append(("user", question))
        result = app.invoke({"messages": conversation})
        answer = result["messages"][-1].content
        conversation = result["messages"]  # carry full history forward

        print(f"\nAssistant: {answer}")


def main():
    parser = argparse.ArgumentParser(description="Agentic Research Assistant")
    parser.add_argument(
        "--query", "-q", type=str, default=None,
        help="Run a single one-off query instead of interactive mode."
    )
    args = parser.parse_args()

    if not check_api_key():
        sys.exit(1)

    if args.query:
        run_single_query(args.query)
    else:
        run_interactive()


if __name__ == "__main__":
    main()
