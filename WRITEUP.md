# Khamuel — Writeup

## Summary

The graded model — **`llama3.2:1b-instruct-q4_K_M`** — is stateless and, on its own, loses the
thread of a long conversation: it forgets earlier facts, repeats itself, and "restarts at basics."
The **MemoryLayer** fixes this by re-injecting a compact, self-updating snapshot of the
conversation into the model **every turn**, so a stateless model behaves as if it remembers. It is
also **topic-agnostic** (works for grief, marriage, addiction, parenting, career, etc.) and runs
fully **on-device** via Ollama.

On the graded 20-turn grief scenario, `llama3.2:1b-instruct-q4_K_M` reaches **fact-recall 8/8,
zero repetition, deterministic-pass true, and LLM-judge 5/5/5** (topic adherence, progressive
depth, cross-session continuity). Fact recall — the primary signal — is 8/8 *every run* because a
post-generation recall guard secures it.

## Architecture

Every turn, the system prompt is assembled from a fixed persona plus four blocks
(capped at ~4000 chars combined), and the last ~6 turns are sent as native chat history:

1. **Pinned facts** — a durable per-user profile (people/relationships, the user's
   age/job/kids, the topic) injected every turn, never evicted.
2. **Hot context** — the last ~6 turns, verbatim.
3. **Rolling summary** — an LLM-compressed summary of older turns (what's been
   discussed/advised, what's unresolved).
4. **Journey state** — a monotonic depth tracker (0–4) that blocks "restart at basics."

`add_turn()` updates all four after each exchange; `save()/load()` persist state as JSON.

## Key engineering decisions

- **Slots → deterministic render (never raw model prose in the always-on block).**
  Pinned facts come from a **hybrid extractor** — a regex floor (precise, deterministic,
  every turn) plus an LLM pass that fills slots the regex can't (unusual jobs, flexible
  phrasing). Code then renders the *sentences*. On an unbiased new-domain test this
  scored **8/8** vs ground truth (regex-only 6/8, LLM-only 5/8). Conflicts resolve in
  favour of the vetted regex value (e.g. occupation).

- **Use the LLM where it's strong, not where it's weak.** A small model is good at
  **structured slot-filling** and **bounded judgments**, bad at **open-ended
  extraction**. So: profile = hybrid slots; journey depth = a turn-counter *floor* the
  LLM may only ratchet up by one rung (its failure mode is harmless); key facts =
  regex-only (the LLM emits bare nouns or parrots prompt examples, so it isn't asked).

- **Generalization by construction.** Death is **subject-attributed** (a death verb
  bound to a relation, or a strong cue next to the nearest person) — never inferred from
  a bare keyword. Grief-specific rendering and the mourning closer are **gated on the
  topic actually being grief**. So a stray flag can't inject mourning into a career or
  parenting chat — verified across 10 diverse invariant scenarios (0 false deaths).

- **Topic-agnostic anti-repetition.** After generation, any sentence that closely
  repeats one used in recent turns is dropped. This is essential at ≤2 B, where models
  loop stock empathy lines; it cut cross-turn repeats ~95% on the smallest models.

- **Reply cleanup.** Leading roleplay stage-directions (`(Softly)`) and single-asterisk
  markdown emphasis are stripped; `think=False` suppresses reasoning tokens from
  thinking-capable models that would otherwise blank a reply.

- **Robustness.** The rolling-summary LLM call has a deterministic template fallback;
  any extraction failure leaves prior memory intact. `build_prompt_context()` is hard-
  capped at 4000 chars (summary trimmed first).

## Honest limitations

- **Comprehension is the model's ceiling, not the layer's.** At 1–2 B the *style* is
  well-controlled (no repetition, no stage-directions, persona-consistent), but the model
  occasionally misreads a complex multi-party situation. A larger model or a fine-tune is
  the only fix — the memory layer cannot lift raw reasoning.
- **The model can fabricate** under recall pressure (invent a name, or "remember" a chat that
  didn't happen) — a 1B comprehension limit, not a layer bug. A larger model is the only real fix.
- **Scripture accuracy is not guaranteed** — the model can misquote verses. The right fix
  is **RAG** (retrieve the verse text), not the LLM's memory.

## Next steps

1. **Fine-tune** (LoRA SFT + DPO) to bake the persona, brevity, and no-repeat behaviour
   into the weights — lets a small model punch above its size and reduces reliance on
   the post-processing band-aids. Quantize the result back to GGUF for mobile.
2. **RAG for scripture** — exact verse retrieval instead of parametric recall.
3. **Async extraction** — run the per-turn memory-update call *after* responding, so the
   user-facing latency is just generation.
4. **Deployment** — serve via Ollama behind an authenticated HTTPS endpoint; do not
   expose port 11434.
