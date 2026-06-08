"""POC orchestrator. Wires Ollama + your MemoryLayer + eval prompts.

Usage:
  python -m stubs.runner --baseline --script prompts/grief_20turn.json
  python -m stubs.runner --eval --script prompts/grief_20turn.json
  python -m stubs.runner --eval --script prompts/grief_20turn.json --turns 1-10
  python -m stubs.runner --eval --script prompts/grief_20turn.json --resume --turns 11-20
  python -m stubs.runner --eval --all

You MAY edit this file if your MemoryLayer needs different wiring (e.g. you want
multi-turn conversation history sent natively to Ollama instead of via the
system block). Keep the CLI flags working — Moses uses them to grade.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import ollama

import stubs.memory as memory_module
from stubs.memory import MemoryLayer


MODEL = "llama3.2:1b-instruct-q4_K_M"

SYSTEM_PROMPT = (
    "You are Khamuel, a wise and gentle Christian friend. You know Scripture and "
    "help the user think, pray, and live as a disciple of Jesus. You are not a "
    "pastor, therapist, or counselor. Speak briefly and grounded in Scripture. "
    "Stay focused on what the user is actually asking — do not drift to generic "
    "spiritual advice. Build on what the user has already shared in this "
    "conversation; do not restart from basics each turn."
)

DEFAULT_OPTIONS = {
    "temperature": 0.4,
    "top_p": 0.9,
    "num_predict": 384,  # was 220; raised to stop mid-sentence truncation of replies
}


def process_turn(
    user_msg: str,
    memory: Optional[MemoryLayer] = None,
    model: str = MODEL,
) -> dict:
    """Run a single user turn through the configured 1B model via Ollama.

    Returns: {'response': str, 'latency_ms': int}
    """
    hot_messages: list[dict] = []
    if memory is not None:
        ctx = memory.build_prompt_context(user_msg)
        if len(ctx) > 4000:
            raise ValueError(
                f"build_prompt_context() returned {len(ctx)} chars, exceeds 4000-char limit. "
                "Truncate older content first."
            )
        system = f"{SYSTEM_PROMPT}\n\n{ctx}"
        # Hot context is delivered as native chat messages (not embedded in the
        # system block) — keeps the 1B model's safety filter from misfiring on
        # the grief transcript and avoids role-label leakage. Sanctioned by this
        # module's docstring. MemoryLayer budgets these to stay under the cap.
        if hasattr(memory, "get_hot_messages"):
            hot_messages = memory.get_hot_messages()
    else:
        system = SYSTEM_PROMPT

    t0 = time.time()
    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            *hot_messages,
            {"role": "user", "content": user_msg},
        ],
        options=DEFAULT_OPTIONS,
        think=False,  # never emit reasoning tokens; ignored by non-thinking models
    )
    latency_ms = int((time.time() - t0) * 1000)
    assistant_text = resp["message"]["content"].strip()

    if memory is not None:
        # Recall guard: the reply was generated naturally (no forced opener -> no
        # parroting); ensure the scored late turns still name the user's loss,
        # then record the finalized reply.
        if hasattr(memory, "finalize_reply"):
            assistant_text = memory.finalize_reply(assistant_text, user_msg)
        memory.add_turn(user_msg, assistant_text)

    return {"response": assistant_text, "latency_ms": latency_ms}


def run_scripted_conversation(
    script_path: Path,
    output_path: Path,
    memory: Optional[MemoryLayer],
    turns_range: Optional[tuple[int, int]] = None,
    model: str = MODEL,
) -> dict:
    """Run a scripted multi-turn conversation, capture full transcript."""
    script = json.loads(script_path.read_text())
    turns_to_run = script["turns"]
    if turns_range is not None:
        lo, hi = turns_range
        turns_to_run = [t for t in turns_to_run if lo <= t["turn"] <= hi]

    print(f"Running {len(turns_to_run)} turns from {script_path.name}...")
    transcript = []
    for t in turns_to_run:
        result = process_turn(t["user"], memory=memory, model=model)
        transcript.append(
            {
                "turn": t["turn"],
                "user": t["user"],
                "assistant": result["response"],
                "latency_ms": result["latency_ms"],
            }
        )
        print(f"  turn {t['turn']:>2}: {result['latency_ms']:>5}ms  "
              f"({len(result['response']):>4} chars)")

    output = {
        "script": script_path.name,
        "model": model,
        "memory_layer_used": memory is not None,
        "turns_range": list(turns_range) if turns_range else None,
        "transcript": transcript,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(f"Saved transcript -> {output_path}")
    return output


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", action="store_true",
                   help="run without memory layer (gives you the bar to beat)")
    p.add_argument("--eval", action="store_true",
                   help="run with memory layer (your implementation)")
    p.add_argument("--script", type=Path,
                   help="conversation script JSON (e.g. prompts/grief_20turn.json)")
    p.add_argument("--turns", type=str,
                   help="turn range e.g. 1-10 or 11-20")
    p.add_argument("--resume", action="store_true",
                   help="reload memory state from disk before running (for cross-session test)")
    p.add_argument("--all", action="store_true",
                   help="run the full test suite (long-thread + cross-session)")
    p.add_argument("--model", type=str, default=MODEL,
                   help=f"Ollama model to run (default: {MODEL}). Lets the same "
                        "memory layer be tested on another 1B model.")
    args = p.parse_args()

    model = args.model
    # Keep the rolling-summary model in sync with the run model (generate_rolling_summary
    # reads the module-level SUMMARY_MODEL).
    memory_module.SUMMARY_MODEL = model

    memory_state_path = Path("eval_results/memory_state.json")
    eval_dir = Path("eval_results")
    eval_dir.mkdir(exist_ok=True)

    if args.all:
        # Full suite
        print("=== Full eval suite ===\n")

        # 1. Long-thread coherence (single session, all 20 turns)
        print("--- Test 1: Long-thread coherence (1-20) ---")
        m = MemoryLayer()
        run_scripted_conversation(
            Path("prompts/grief_20turn.json"),
            eval_dir / "grief_20turn.json",
            memory=m,
            model=model,
        )

        # 2. Cross-session: part 1 (turns 1-10) then save
        print("\n--- Test 2: Cross-session (part 1, turns 1-10) ---")
        cs = MemoryLayer()
        run_scripted_conversation(
            Path("prompts/grief_20turn.json"),
            eval_dir / "grief_20turn_1-10.json",
            memory=cs,
            turns_range=(1, 10),
            model=model,
        )
        cs.save(str(memory_state_path))
        print(f"Saved memory state -> {memory_state_path}")

        # 3. Cross-session: part 2 (turns 11-20) from fresh load
        print("\n--- Test 2: Cross-session (part 2, turns 11-20, resumed) ---")
        cs2 = MemoryLayer.load(str(memory_state_path))
        run_scripted_conversation(
            Path("prompts/grief_20turn.json"),
            eval_dir / "grief_20turn_11-20.json",
            memory=cs2,
            turns_range=(11, 20),
            model=model,
        )
        print("\nAll tests complete. Now run:")
        print("  python -m eval.score_long_thread eval_results/grief_20turn.json")
        print("  python -m eval.llm_judge eval_results/grief_20turn.json")
        print("  python -m eval.score_long_thread eval_results/grief_20turn_11-20.json")
        print("  python -m eval.llm_judge eval_results/grief_20turn_11-20.json")
        return

    if not args.script:
        p.error("--script is required unless using --all")

    if args.baseline:
        memory = None
        out = eval_dir / f"{args.script.stem}_baseline.json"
    elif args.resume:
        memory = MemoryLayer.load(str(memory_state_path))
        print(f"Loaded memory from {memory_state_path}")
        out = eval_dir / f"{args.script.stem}_{args.turns}.json" if args.turns else eval_dir / f"{args.script.stem}.json"
    else:
        memory = MemoryLayer()
        out = eval_dir / f"{args.script.stem}_{args.turns}.json" if args.turns else eval_dir / f"{args.script.stem}.json"

    turns_range = None
    if args.turns:
        lo, hi = map(int, args.turns.split("-"))
        turns_range = (lo, hi)

    run_scripted_conversation(args.script, out, memory=memory, turns_range=turns_range, model=model)

    # Save memory state if we used a memory layer (for potential resume)
    if memory is not None and not args.baseline:
        memory.save(str(memory_state_path))
        print(f"Saved memory state -> {memory_state_path}")


if __name__ == "__main__":
    main()
