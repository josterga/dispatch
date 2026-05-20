import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv(override=True)

import db
from agent_loader import build_registry
from pipeline import MODEL, OLLAMA_TIMEOUT, create_pipeline, AgentPipeline
from providers import LLMClient


def _check_api_key():
    from tools import OLLAMA_API_KEY
    if not OLLAMA_API_KEY:
        print("Error: OLLAMA_API_KEY is not set.")
        sys.exit(1)


def _run_once(client, session_id, prompt, config):
    answer = create_pipeline(client, session_id, config).run(prompt)
    if answer is not None:
        db.log_message(session_id, "assistant", answer)
        print(answer)
    else:
        print("[pipeline did not produce a result]")
    db.close()


def _interactive_loop(client, session_id, config, label, default_prompt=None):
    print(f"{label} — Pipeline Mode (model: {MODEL})")
    print("-" * 40)
    first_run = default_prompt is not None
    while True:
        try:
            if first_run:
                user_input = default_prompt
                first_run = False
                print(f"\nYou: {user_input}")
            else:
                user_input = input("\nYou: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit"):
                    print("Goodbye!")
                    break
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        db.log_message(session_id, "user", user_input)
        try:
            answer = create_pipeline(client, session_id, config).run(user_input)
            if answer is not None:
                db.log_message(session_id, "assistant", answer)
                print(f"\nAssistant: {answer}\n")
            else:
                print("\nAssistant: [pipeline did not produce a result]\n")
        except Exception as e:
            print(f"\nUnexpected error: {e}")
    db.close()


if __name__ == "__main__":
    # Seed DB from YAML files before building the registry
    db.seed_agents_from_yaml()
    db.seed_default_dispatches()
    registry = build_registry()
    agent_names = sorted(registry)

    parser = argparse.ArgumentParser(description="Local web search assistant")
    parser.add_argument("--serve", action="store_true", help="Start the HTTP server")
    parser.add_argument("--agent", type=str, default=None,
                        choices=agent_names, metavar="AGENT",
                        help=f"Agent to run. Choices: {', '.join(agent_names)}")
    parser.add_argument("--once", action="store_true",
                        help="Run once headlessly and exit (for dispatch use)")
    parser.add_argument("--query", type=str, default=None,
                        help="Query to use with --once")
    args = parser.parse_args()

    if args.serve:
        import uvicorn
        from server import app
        host = os.getenv("SERVER_HOST", "0.0.0.0")
        port = int(os.getenv("SERVER_PORT", "8000"))
        print(f"Starting server on {host}:{port}")
        uvicorn.run(app, host=host, port=port, timeout_keep_alive=600)
        sys.exit(0)

    _check_api_key()
    agent_name = args.agent or "general"
    if agent_name not in registry:
        print(f"Unknown agent '{agent_name}'. Available: {', '.join(agent_names)}")
        sys.exit(1)

    config = registry[agent_name]
    client = LLMClient(timeout=OLLAMA_TIMEOUT)
    session_id = db.start_session(MODEL, agent=agent_name, source="dispatch" if args.once else "cli")

    if args.once:
        # --once requires a query for agents whose goal_template is just "{query}"
        prompt = args.query or config.goal_template
        if not prompt or prompt == "{query}":
            print(f"Error: --query is required for agent '{agent_name}' with --once.")
            sys.exit(1)
        db.log_message(session_id, "user", prompt)
        _run_once(client, session_id, prompt, config)
    else:
        default_prompt = None if config.goal_template == "{query}" else config.goal_template
        _interactive_loop(client, session_id, config, config.agent_name.title(), default_prompt)
