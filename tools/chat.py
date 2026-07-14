#!/usr/bin/env python3
"""Minimal terminal chat client for a running Alveare server.

Talks to the OpenAI-compatible endpoint (`/v1/chat/completions`) over HTTP, so it
works against `alveare serve ...` (or any OpenAI-compatible server). It streams
tokens as they arrive. Quick and dirty — for eyeballing a model, not production.

Usage:
    alveare chat [--host H] [--port P] [--model NAME] [--no-stream]

Type your message and press Enter. Ctrl-C or an empty line at the prompt exits.
"""
import argparse
import json
import sys

import requests


def list_default_model(base):
    """Ask the server which model it's serving; return the first id, or None."""
    try:
        r = requests.get(f"{base}/models", timeout=5)
        r.raise_for_status()
        data = r.json().get("data", [])
        if data:
            return data[0].get("id")
    except Exception:
        pass
    return None


def stream_reply(base, model, messages):
    """POST a chat completion with stream=True and print deltas as they arrive.

    Returns the full assistant text (to append to the conversation), or None on error.
    """
    try:
        with requests.post(
            f"{base}/chat/completions",
            json={"model": model, "messages": messages, "stream": True, "temperature": 0.0},
            stream=True,
            timeout=None,
        ) as r:
            r.raise_for_status()
            parts = []
            for line in r.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8", "replace")
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                try:
                    delta = json.loads(payload)["choices"][0]["delta"].get("content", "")
                except (KeyError, IndexError, json.JSONDecodeError):
                    continue
                if delta:
                    parts.append(delta)
                    print(delta, end="", flush=True)
            print()
            return "".join(parts)
    except requests.RequestException as e:
        print(f"\n[error talking to server: {e}]", file=sys.stderr)
        return None


def blocking_reply(base, model, messages):
    """Non-streaming variant: one request, print the whole reply."""
    try:
        r = requests.post(
            f"{base}/chat/completions",
            json={"model": model, "messages": messages, "temperature": 0.0},
            timeout=None,
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        print(text)
        return text
    except (requests.RequestException, KeyError, IndexError) as e:
        print(f"[error talking to server: {e}]", file=sys.stderr)
        return None


def main():
    ap = argparse.ArgumentParser(description="Minimal terminal chat against a running Alveare server.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default="8000")
    ap.add_argument("--model", default=None, help="model id (default: whatever the server reports)")
    ap.add_argument("--no-stream", action="store_true", help="wait for the full reply instead of streaming")
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}/v1"
    model = args.model or list_default_model(base)
    if model is None:
        print(f"Could not reach an Alveare server at {base}.", file=sys.stderr)
        print("Start one first, e.g.:  alveare serve gemma4", file=sys.stderr)
        sys.exit(1)

    print(f"Alveare chat — model '{model}' @ {base}")
    print("Type a message and press Enter. Empty line or Ctrl-C to quit.\n")

    messages = []
    while True:
        try:
            user = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            break
        messages.append({"role": "user", "content": user})
        reply = (blocking_reply if args.no_stream else stream_reply)(base, model, messages)
        if reply is None:
            messages.pop()  # drop the turn we couldn't answer
            continue
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
