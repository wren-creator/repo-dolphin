"""Interactive session: load the index once, then ask as many questions as
you want without re-paying the load cost or re-activating anything each time.

    python repl.py

Type a question and press enter. Ctrl-D, Ctrl-C, or "exit"/"quit" to leave.
"""

from __future__ import annotations

from ask import answer, load_config, load_store


def main() -> None:
    config = load_config()
    print(f"Loading index ({config['index_path']})...")
    store = load_store(config)
    print(f"Ready — {len(store.chunks)} chunks indexed. Ask a question, or 'exit' to quit.\n")

    while True:
        try:
            question = input("ask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break

        try:
            print()
            print(answer(question, store, config))
            print()
        except Exception as e:
            print(f"error: {e}\n")


if __name__ == "__main__":
    main()
