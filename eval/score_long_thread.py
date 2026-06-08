"""Deterministic scoring for the long-thread test.

Reads a transcript JSON from eval_results/, computes three automated metrics:

  1. Fact recall — how many late turns (>=13) reference 'Sarah' or 'my mother'?
  2. Repetition — avg near-duplicate sentence pairs per response (cosine > 0.85).
  3. Restart-marker reuse — do very late turns (>=17) reuse basics-restart phrases
     that already appeared in early turns (1-6)?

Writes <stem>_scores.json next to the input.

Usage:
  python -m eval.score_long_thread eval_results/grief_20turn.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import nltk
import numpy as np
from sentence_transformers import SentenceTransformer

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# Load once at module import so the script is fast on subsequent runs.
EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")

# Phrases that indicate a "restart at basics" response.
# If these appear in BOTH early (turns 1-6) AND very-late (turns 17+) responses,
# the bot is restarting the conversation instead of progressing.
RESTART_MARKERS = [
    r"have you (considered|tried|thought about) prayi(ng|ng)",
    r"it might (help|be helpful) to talk to (someone|a pastor|a counselor|a therapist)",
    r"remember that god (loves you|is with you|cares for you|has a plan)",
    r"god has a plan",
    r"trust in (jesus|god|the lord)",
    r"lean on your (faith|community|church|family)",
    r"have you (read|tried reading) the bible",
    r"god understands your (pain|grief|sorrow|loss)",
    r"\bhe is always with you\b",
    r"turn to (god|jesus|prayer) for comfort",
    r"prayer can be a powerful",
    r"i'm here to listen",  # generic chatbot open
]

# Patterns indicating recall of the user's setup facts (Sarah / mother).
FACT_RECALL_PATTERNS = [
    r"\bsarah\b",
    r"\bmy mother\b",
    r"\byour (mom|mother)\b",
    r"\bher (death|passing|cancer)\b",
    r"\b(your|her) (mom|mother)'s (death|passing)\b",
]


def sentences(text: str) -> list[str]:
    """Split into sentences; drop very short ones."""
    return [s.strip() for s in nltk.sent_tokenize(text) if len(s.strip()) > 10]


def near_dup_count(text: str, threshold: float = 0.85) -> int:
    """Count near-duplicate sentence pairs (cosine > threshold)."""
    sents = sentences(text)
    if len(sents) < 2:
        return 0
    embs = EMBEDDER.encode(sents, show_progress_bar=False)
    n = len(sents)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            denom = np.linalg.norm(embs[i]) * np.linalg.norm(embs[j])
            if denom == 0:
                continue
            sim = float(np.dot(embs[i], embs[j]) / denom)
            if sim > threshold:
                count += 1
    return count


def fact_recall_hits(text: str) -> list[str]:
    """Return matched fact-recall patterns."""
    lower = text.lower()
    return [p for p in FACT_RECALL_PATTERNS if re.search(p, lower)]


def restart_marker_hits(text: str) -> list[str]:
    """Return restart-at-basics markers found in text."""
    lower = text.lower()
    return [m for m in RESTART_MARKERS if re.search(m, lower)]


def score_transcript(transcript: list[dict]) -> dict:
    """Compute all deterministic metrics for a transcript."""
    if not transcript:
        return {"error": "empty transcript"}

    # Bucket turns
    early = [t for t in transcript if t["turn"] <= 6]
    late = [t for t in transcript if t["turn"] >= 13]
    very_late = [t for t in transcript if t["turn"] >= 17]

    # Fact recall
    fact_recall_count = sum(1 for t in late if fact_recall_hits(t["assistant"]))

    # Repetition
    dup_counts = [near_dup_count(t["assistant"]) for t in transcript]
    avg_dups = sum(dup_counts) / len(dup_counts) if dup_counts else 0.0
    per_turn_dups = [
        {"turn": t["turn"], "near_dups": d}
        for t, d in zip(transcript, dup_counts)
    ]

    # Restart-marker reuse
    early_markers: set[str] = set()
    for t in early:
        early_markers.update(restart_marker_hits(t["assistant"]))

    late_marker_reuse_turns: list[int] = []
    for t in very_late:
        late_hits = set(restart_marker_hits(t["assistant"]))
        if late_hits & early_markers:
            late_marker_reuse_turns.append(t["turn"])

    fact_recall_pass = fact_recall_count >= 3
    repetition_pass = avg_dups <= 0.5
    restart_pass = len(late_marker_reuse_turns) == 0

    return {
        "total_turns": len(transcript),
        "fact_recall": {
            "late_turns_with_recall": fact_recall_count,
            "total_late_turns": len(late),
            "target_min": 3,
            "pass": fact_recall_pass,
        },
        "repetition": {
            "avg_near_dup_pairs_per_response": round(avg_dups, 3),
            "per_turn": per_turn_dups,
            "target_max": 0.5,
            "pass": repetition_pass,
        },
        "restart_markers": {
            "early_markers_seen": sorted(early_markers),
            "late_turn_reuses": late_marker_reuse_turns,
            "target_max": 0,
            "pass": restart_pass,
        },
        "all_deterministic_pass": (
            fact_recall_pass and repetition_pass and restart_pass
        ),
    }


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m eval.score_long_thread eval_results/grief_20turn.json")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"Not found: {path}")
        sys.exit(1)

    data = json.loads(path.read_text())
    transcript = data.get("transcript", [])
    scores = score_transcript(transcript)

    print(json.dumps(scores, indent=2))

    out = path.parent / f"{path.stem}_scores.json"
    out.write_text(json.dumps(scores, indent=2))
    print(f"\nSaved -> {out}")

    if scores.get("all_deterministic_pass"):
        print("\n✓ ALL deterministic checks PASS")
    else:
        print("\n✗ One or more deterministic checks FAIL — see details above")


if __name__ == "__main__":
    main()
