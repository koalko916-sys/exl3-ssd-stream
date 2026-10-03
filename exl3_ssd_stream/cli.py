"""Command-line entry point; --help works without importing PyTorch."""

import argparse
import json
from pathlib import Path

from .validation import cache_budget, validate_generation


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Experimental GLM EXL3 SSD streaming")
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--prompt", default="What is 2 + 2? Answer briefly.")
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--context", type=int, default=512)
    p.add_argument(
        "--cache-gib",
        type=cache_budget,
        default=-1,
        help="GPU weight cache GiB; -1 chooses from free VRAM",
    )
    p.add_argument(
        "--host-cache-gib",
        type=cache_budget,
        default=-1,
        help="Pinned RAM weight cache GiB; -1 automatic, 0 disables",
    )
    p.add_argument("--report", type=Path, default=Path("benchmark-exl3.json"))
    p.add_argument("--interactive", action="store_true")
    return p


def main() -> None:
    p = parser()
    args = p.parse_args()
    try:
        validate_generation(args.context, args.tokens)
        if not args.model.is_dir():
            raise ValueError("Model directory does not exist")
        if not args.interactive and not args.prompt.strip():
            raise ValueError("Prompt must not be empty")
    except ValueError as error:
        p.error(str(error))

    from .runtime import StreamingEngine

    engine = StreamingEngine(args.model, args.context, args.cache_gib, args.host_cache_gib)
    try:
        history = []
        while True:
            try:
                prompt = input("\nGLM > ").strip() if args.interactive else args.prompt
            except EOFError:
                break
            if args.interactive and prompt.lower() in ("exit", "quit", "выход"):
                break
            if args.interactive and prompt.lower() in ("/new", "/новый"):
                history = []
                continue
            if not prompt:
                continue
            try:
                result = engine.generate(prompt, args.tokens, history=history)
            except ValueError as error:
                if not args.interactive:
                    raise
                print(str(error), flush=True)
                continue
            next_history = result.pop("history")
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                f"\n{result['tokens_per_second']:.3f} tokens/s; "
                f"{result['generated_tokens']} tokens; {result['finish_reason']}"
            )
            if not result["answer"]:
                if not args.interactive:
                    raise RuntimeError("Model produced no final answer; inspect report")
                print(
                    "No final answer: reasoning may have exhausted --tokens. "
                    "The incomplete turn was not added to history.",
                    flush=True,
                )
            else:
                history = next_history
            if not args.interactive:
                print(json.dumps(result, ensure_ascii=False, indent=2))
                break
    finally:
        engine.close()
