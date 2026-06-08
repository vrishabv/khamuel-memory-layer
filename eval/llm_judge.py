"""LLM-as-judge for long-thread rubric metrics.

Scores three dimensions on the transcript:
  1. topic_adherence (1-5)
  2. progressive_depth (1-5)
  3. cross_session_continuity (1-5; only meaningful on the 11-20 transcript)

Defaults to OpenAI gpt-4o-mini for reproducibility across candidates.
Falls back to local Ollama (llama3.2:3b) if OPENAI_API_KEY not set —
slightly noisier but still useful for the candidate's self-grading.

Usage:
  python -m eval.llm_judge eval_results/grief_20turn.json
  OPENAI_API_KEY=sk-... python -m eval.llm_judge eval_results/grief_20turn.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


JUDGE_PROMPT = """You are an evaluator scoring the quality of a Christian-counseling chatbot's responses in a multi-turn conversation about grief.

CONTEXT (the user is a fictional persona being scripted):
- User is a 40-year-old working parent of two elementary-aged children.
- User's mother (named Sarah) died one month ago from cancer, age 71, six weeks from diagnosis to passing.
- Turns 1-3 establish loss + name + role.
- Turns 4-12 explore doubt, anger at God, guilt, isolation.
- Turns 13-20 ask for practical next steps (daily practice, scripture choice, talking to kids, returning to church).

YOUR JOB: read the full transcript below and score it on THREE dimensions, 1-5 each.

DIMENSION 1 — topic_adherence (1-5):
  Does each response stay anchored in the user's SPECIFIC situation (Sarah's death, the user's role as a working parent, the user's specific questions), or does the bot drift to generic faith/grief advice that ignores the user's stated context?
    5 = every response anchored in Sarah / user's grief
    4 = ~1-2 turns drift to generic advice
    3 = roughly half of responses drift
    2 = most responses generic
    1 = no awareness of Sarah / specific situation

DIMENSION 2 — progressive_depth (1-5):
  Do turns 13-20 build on what was established in turns 1-12, or does the bot keep restarting with basics (e.g., "have you considered praying", "remember God loves you", "lean on community") that already came up earlier?
    5 = late turns clearly build on earlier ones; no restarts; depth visibly increases
    4 = mostly progressive with ~1 minor restart
    3 = some basics re-explained
    2 = bot restarts frequently
    1 = every turn reads like a fresh conversation

DIMENSION 3 — cross_session_continuity (1-5):
  ONLY applicable when scoring a transcript that resumes from saved state (turns 11-20 only).
  Does turn 11 (first turn after the reload) reference at least one specific fact from earlier turns (Sarah's name, the user's role, the cause of death, an earlier emotion)?
    5 = turn 11 explicitly references a named fact from earlier turns
    4 = turn 11 references "what you shared" with some specificity
    3 = vague continuity ("what we discussed before")
    2 = generic opening, no continuity
    1 = turn 11 reads like a brand new conversation, no awareness of prior turns
  If the transcript starts at turn 1 (not a resume), set this field to null.

Output STRICT JSON only, no prose:
{
  "topic_adherence": <1-5>,
  "progressive_depth": <1-5>,
  "cross_session_continuity": <1-5 or null>,
  "rationale": "<2-3 sentence justification>"
}
"""


def _format_transcript(transcript: list[dict]) -> str:
    parts = []
    for t in transcript:
        parts.append(f"TURN {t['turn']}\nUSER: {t['user']}\nASSISTANT: {t['assistant']}")
    return "\n\n".join(parts)


def judge_with_openai(transcript: list[dict]) -> dict:
    from openai import OpenAI

    client = OpenAI()
    formatted = _format_transcript(transcript)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": formatted},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def judge_with_ollama(transcript: list[dict]) -> dict:
    import ollama

    formatted = _format_transcript(transcript)
    resp = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": formatted},
        ],
        options={"temperature": 0.0},
        format="json",
    )
    return json.loads(resp["message"]["content"])


def summarise(scores: dict) -> dict:
    """Add convenience pass/fail booleans against the README targets."""
    targets = {
        "topic_adherence": 4.0,
        "progressive_depth": 4.0,
        "cross_session_continuity": 4.0,
    }
    summary = {}
    for k, target in targets.items():
        v = scores.get(k)
        if v is None:
            summary[k] = {"score": None, "target": target, "pass": None, "n/a": True}
        else:
            summary[k] = {"score": v, "target": target, "pass": v >= target}
    return summary


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m eval.llm_judge eval_results/grief_20turn.json")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"Not found: {path}")
        sys.exit(1)

    data = json.loads(path.read_text())
    transcript = data.get("transcript", [])

    if os.environ.get("OPENAI_API_KEY"):
        print("Judging via OpenAI gpt-4o-mini (reproducible)...")
        scores = judge_with_openai(transcript)
    else:
        print("No OPENAI_API_KEY — falling back to Ollama llama3.2:3b.")
        print("For reproducibility across candidates, set OPENAI_API_KEY.")
        scores = judge_with_ollama(transcript)

    result = {
        "raw_scores": scores,
        "summary": summarise(scores),
    }
    print(json.dumps(result, indent=2))

    out = path.parent / f"{path.stem}_judge.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
