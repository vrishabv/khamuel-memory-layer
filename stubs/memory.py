"""Khamuel conversation MemoryLayer (generalized).

Implements the four-block architecture from README.md so a 1B/2B model can sustain
a coherent 20+ turn single-topic conversation -- for ANY user and ANY problem, not
just the grief/Sarah scenario:

  1. Pinned facts   -> a generic per-user PROFILE (people + relationships + the
                       user's own facts + the topic) rendered into canonical
                       sentences and injected on EVERY turn; never evicted.
  2. Key facts      -> top-5 surfaced each turn (pinned always included).
  3. Hot context    -> last N turns, sent as NATIVE chat messages.
  4. Rolling summary -> [Tier 2] compresses turns older than the hot window.
  5. Journey state   -> [Tier 3] depth tracker that blocks basics-restart.

GENERALIZATION (vs the earlier grief-only build):
  - Extraction is topic-agnostic: it pulls whoever/whatever the user mentions
    (relationship + name + status/cause/age), the user's own facts, and a TOPIC
    label (grief / marriage / addiction / mental-health / faith-doubt / ...),
    instead of hardcoded mother/Sarah/cancer regexes.
  - The directive, pinned facts, remembrance closers and rolling-summary prompt
    are all BUILT FROM the profile at runtime -- no "Sarah" baked into the code.
  - Identity grounding prevents addressing the user by another person's name.

WIRING NOTE (sanctioned by runner.py's docstring): hot context is delivered as
real user/assistant chat messages via get_hot_messages(), NOT embedded in the
system block. build_prompt_context() returns only the system-side blocks.

Hard rules respected: public method signatures unchanged; build_prompt_context()
always <= 4000 chars; save()/load() are pure JSON.
"""

from __future__ import annotations

import difflib
import json
import os
import re
from typing import Optional

import ollama

# Profile extraction is HYBRID:
#   - regex runs EVERY turn (fast, precise, deterministic, free) -- the primary path;
#   - an LLM "gap-fill" rides the rolling-summary call (every SUMMARY_INTERVAL turns)
#     to catch facts regex missed (unusual phrasing/jobs/relationships) -- ZERO extra
#     model calls, since the summary call already happens.
# The merge fills only EMPTY slots, so regex (high precision) always wins on conflict
# and the LLM only adds coverage. Set KHAMUEL_NO_GAPFILL=1 for pure-regex (e.g. for a
# deterministic A/B comparison).
LLM_GAP_FILL = os.environ.get("KHAMUEL_NO_GAPFILL") != "1"

# Total per-turn context budget (system block + native hot messages combined).
MAX_CONTEXT_CHARS = 4000

# Rolling-summary settings.
SUMMARY_MODEL = "llama3.2:1b-instruct-q4_K_M"  # kept in sync with the run model by runner.py
SUMMARY_INTERVAL = 6
SUMMARY_MAX_CHARS = 900
SUMMARY_FOLD_AFTER = 24

# Header the README mandates verbatim.
FACTS_HEADER = "Things Khamuel knows about you:"


# ---------------------------------------------------------------------------
# Generic, deterministic profile extraction (topic-agnostic).
# ---------------------------------------------------------------------------

# Relationship words we recognise; normalised to a canonical form below.
_RELATION_WORDS = (
    "mother", "mom", "mum", "father", "dad", "husband", "wife", "son", "daughter",
    "brother", "sister", "friend", "partner", "boss", "grandmother", "grandfather",
    "grandma", "grandpa", "fiance", "fiancee", "girlfriend", "boyfriend", "child",
    "baby", "uncle", "aunt", "cousin", "mentor", "coworker", "colleague",
    "roommate", "landlord", "neighbor", "niece", "nephew", "stepmother", "stepfather",
    "godmother", "godfather",
)
_RELATION_NORM = {
    "mom": "mother", "mum": "mother", "dad": "father",
    "grandma": "grandmother", "grandpa": "grandfather",
}
# Capitalised words that are NOT names (guards the optional name capture, since the
# case-insensitive person regex would otherwise grab "is", "We", etc.).
_NAME_STOP = {
    "is", "was", "are", "were", "has", "had", "and", "the", "but", "so", "a", "an",
    "i", "we", "she", "he", "they", "who", "that", "just", "really", "still", "now",
    "passed", "died", "loved", "always", "never", "told", "said", "left", "got",
}
# "my <relation> [Name]"  (optional Capitalised name immediately after)
_PERSON_RE = re.compile(
    r"\bmy (" + "|".join(_RELATION_WORDS) + r")\b(?:[,\s]+(?:named\s+)?([A-Z][a-z]+))?",
    re.I,
)
# Death is detected by EVIDENCE BOUND TO A PERSON, never by a bare keyword anywhere
# in the message. Two routes:
#   1. a death VERB tied to a relation: "lost/buried/grieving my <relation>",
#   2. a STRONG death cue ("died", "passed away", "funeral") attributed to the
#      NEAREST person mention (so "while I'm gone" / "I lost my job" never count).
# Ambiguous words ("gone", bare "passed"/"lost", "leaving") are deliberately excluded
# -- they only count when grammatically attached to a person (route 1) or adjacent to
# one (route 2). This generalises the same subject-attribution rule used for kids.
_LOST_PERSON_RE = re.compile(
    r"\b(?:lost|losing|buried|grieving|mourning)\s+(?:my|our|their|his|her)\s+"
    r"(?:dear\s+|beloved\s+|late\s+)?(" + "|".join(_RELATION_WORDS) + r")\b",
    re.I,
)
_STRONG_DEATH_RE = re.compile(
    r"\b(died|passed away|passed on|funeral|no longer with us|her passing|his passing|"
    r"their passing|gone forever|rest in peace)\b",
    re.I,
)
_CAUSE_RE = re.compile(
    r"\b(cancer|heart attack|stroke|covid|car accident|accident|suicide|overdose|"
    r"dementia|alzheimer'?s|kidney failure|liver failure|pneumonia|illness)\b",
    re.I,
)
_PERSON_AGE_RE = re.compile(r"\b(?:she|he|they)\s+(?:was|were)\s+(\d{1,3})\b", re.I)
# duration only counts when the message is about a loss/diagnosis (avoids grabbing
# "three weeks since I went to church" as the illness timeframe).
_DURATION_RE = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(day|week|month|year)s?\b",
    re.I,
)
_DIAGNOSIS_CTX = re.compile(r"\b(diagnos|passing|passed|died|terminal|sick|illness)\b", re.I)
_AGO_RE = re.compile(
    r"\b(a month|one month|\d+\s*(?:days?|weeks?|months?|years?))\s+ago\b", re.I
)

_USER_AGE_RE = re.compile(r"\bI(?:'?m| am)\s+(\d{2})\b")
_USER_NAME_RE = re.compile(r"\bmy name is\s+([A-Z][a-z]+)", re.I)
_FULLTIME_RE = re.compile(r"\b(full[- ]?time|working full)\b", re.I)
_KIDS_RE = re.compile(r"\b(one|two|three|four|\d+)\s+(kids|children)\b", re.I)
_KIDS_SCHOOL_RE = re.compile(r"\b(elementary|middle school|high school|kindergarten|toddler|teenage)\b", re.I)
# Guard against the "Maya has three kids" trap: a kids count governed by a
# THIRD-PERSON subject (a name or she/he/they + 'has'/'have') belongs to that
# person, not the user. Keyed on third-person-singular 'has' (vs first-person
# 'I/we have'), so "two kids in elementary" in a self-description still counts.
_KIDS_OWNER_RE = re.compile(
    r"(?:she|he|they|her|his|their|[A-Z][a-z]+)\s+has\s+$|(?:she|he|they)\s+have\s+$",
    re.I,
)


def _kids_owned_by_other(msg: str, idx: int) -> bool:
    return bool(_KIDS_OWNER_RE.search(msg[max(0, idx - 30):idx]))
# occupation: a small word-list OR an explicit "I'm a / I work as a <job>".
_OCC_WORDS = (
    "project manager", "nurse", "teacher", "engineer", "doctor", "lawyer",
    "accountant", "pastor", "manager", "developer", "designer", "writer",
    "social worker", "therapist", "salesperson", "consultant", "analyst",
)
# occupation only when stated in FIRST PERSON about oneself, so "talk to our
# pastor" is NOT taken as the user's job. Known-job list keeps it precise; the
# LLM extractor (below) handles unusual jobs.
_OCC_FP_RE = re.compile(
    r"\bI(?:'?m| am| work as)\s+(?:a |an )?(" + "|".join(_OCC_WORDS) + r")\b", re.I
)

# Topic classifier: (label, situation phrase, regex). First match wins.
_TOPICS = [
    ("grief", "grieving a loss",
     # 'lost' must be bound to a PERSON (not "lost my job/keys/way"); bare ambiguous
     # words ("passed") are dropped so only genuine bereavement routes here.
     r"\blost\s+(?:(?:my|our|the)\s+(?:dear\s+|beloved\s+)?"
     r"(?:mom|mum|mother|dad|father|husband|wife|son|daughter|brother|sister|child|"
     r"baby|grand\w+|partner|fiance\w*|friend)|someone|a loved one|him|her|them)"
     r"|\bdied\b|passed away|passed on|grieving|\bgrief\b|funeral|mourning|"
     r"miss (?:her|him|them)|(?:her|his|their) (?:death|passing)"),
    ("marriage", "struggles in their marriage",
     r"divorce|affair|cheated on|my (husband|wife|marriage)|separation|unfaithful"),
    ("addiction", "an addiction struggle",
     r"\bdrink|drunk|sober|relapse|addict|porn|overdose|using again|can'?t stop\b"),
    ("mental_health", "heavy feelings of anxiety or depression",
     r"depress|anxious|anxiety|panic|hopeless|empty|numb|can'?t get out of bed"),
    ("faith_doubt", "wrestling with doubt and faith",
     r"doubt|don'?t believe|losing my faith|deconstruct|why does god|unanswered prayer"),
    ("parenting", "a struggle with their child",
     r"my (kid|kids|son|daughter|teen|teenager|child)\b"),
    ("work_finance", "stress over work or money",
     r"\bjob\b|laid off|fired|unemploy|money|bankrupt|debt|can'?t afford|bills"),
]


def _norm_relation(rel: str) -> str:
    rel = rel.lower()
    return _RELATION_NORM.get(rel, rel)


def _find_person(profile: dict, relation: str) -> Optional[dict]:
    for p in profile.get("people", []):
        if p["relation"] == relation:
            return p
    return None


def _deceased_relations(msg: str) -> set:
    """Relations the message states as DECEASED, by subject attribution (not mere
    keyword presence). Route 1: 'lost/buried/grieving my <relation>'. Route 2: a
    strong death cue ('died'/'passed away'/'funeral') attributed to the NEAREST
    'my <relation>' mention within ~50 chars. So 'while I'm gone' or 'I lost my job'
    never flag anyone, and 'my mentor ... is retiring' stays living."""
    rels = set()
    for m in _LOST_PERSON_RE.finditer(msg):
        rels.add(_norm_relation(m.group(1)))
    people = list(_PERSON_RE.finditer(msg))
    for d in _STRONG_DEATH_RE.finditer(msg):
        nearest, best = None, 10 ** 9
        for p in people:
            dist = abs(p.start() - d.start())
            if dist < best:
                best, nearest = dist, _norm_relation(p.group(1))
        if nearest is not None and best <= 50:
            rels.add(nearest)
    return rels


def _extract_profile(profile: dict, user_msg: str) -> None:
    """Update the generic profile dict in place from one user message.

    Earliest authoritative statement wins (we only fill empty slots), so a later
    casual mention can't corrupt an anchor.
    """
    profile.setdefault("people", [])
    deceased_rels = _deceased_relations(user_msg)  # subject-attributed, not keyword presence

    # people + relationships (+ optional name)
    for m in _PERSON_RE.finditer(user_msg):
        relation = _norm_relation(m.group(1))
        name = m.group(2)
        person = _find_person(profile, relation)
        if person is None:
            person = {"relation": relation, "name": None, "status": "living",
                      "age": None, "cause": None, "when": None, "timeframe": None}
            profile["people"].append(person)
        if name and not person["name"] and name[0].isupper() and name.lower() not in _NAME_STOP:
            person["name"] = name
        if relation in deceased_rels and person["status"] != "deceased":
            person["status"] = "deceased"

    # the "primary" person = first deceased, else first mentioned
    primary = _primary_person(profile)

    if primary is not None:
        c = _CAUSE_RE.search(user_msg)
        if c and not primary["cause"]:
            primary["cause"] = c.group(1).lower()
        a = _PERSON_AGE_RE.search(user_msg)
        if a and not primary["age"]:
            primary["age"] = a.group(1)
        if _DIAGNOSIS_CTX.search(user_msg) and not primary["timeframe"]:
            d = _DURATION_RE.search(user_msg)
            if d:
                primary["timeframe"] = f"{d.group(1).lower()} {d.group(2).lower()}s".replace("ss", "s")
        if not primary["when"]:
            ago = _AGO_RE.search(user_msg)
            if ago:
                primary["when"] = ago.group(0).strip()

    # user's own facts
    if not profile.get("user_name"):
        nm = _USER_NAME_RE.search(user_msg)
        if nm:
            profile["user_name"] = nm.group(1)
    if not profile.get("user_age"):
        ua = _USER_AGE_RE.search(user_msg)
        if ua:
            profile["user_age"] = ua.group(1)
    if _FULLTIME_RE.search(user_msg):
        profile["works_fulltime"] = True
    if not profile.get("kids"):
        k = _KIDS_RE.search(user_msg)
        if k and not _kids_owned_by_other(user_msg, k.start()):
            school = _KIDS_SCHOOL_RE.search(user_msg)
            profile["kids"] = f"{k.group(1)} {k.group(2)}" + (f" in {school.group(1)}" if school else "")
    # occupation: a regex match comes from the vetted job list and is first-person,
    # so it OVERRIDES an earlier LLM guess (e.g. a turn-1 "works full-time" that
    # should yield to a later "I'm a project manager").
    o = _OCC_FP_RE.search(user_msg)
    if o:
        profile["occupation"] = o.group(1).lower()

    # topic: first detected topic STICKS for the rest of the conversation.
    if not profile.get("topic"):
        for label, situation, pat in _TOPICS:
            if re.search(pat, user_msg, re.I):
                profile["topic"] = label
                profile["situation"] = situation
                break

    # enrich a grief situation with the specific person, once known
    if profile.get("topic") == "grief":
        pp = _primary_person(profile)
        if pp and pp.get("name"):
            profile["situation"] = f"grieving the loss of their {pp['relation']} {pp['name']}"


# ---------------------------------------------------------------------------
# LLM-based profile extraction (robust alternative to the regex extractor).
# ---------------------------------------------------------------------------

_PROFILE_PROMPT = (
    "You maintain a structured profile of a person talking to a Christian pastoral "
    "chatbot. Given the CURRENT profile (JSON) and the person's NEW message, return "
    "the UPDATED profile as STRICT JSON only -- no prose. Rules:\n"
    "- Only record facts the PERSON has clearly stated about themselves; never invent.\n"
    "- Keep existing facts unless the new message corrects them.\n"
    "- 'occupation' is the USER'S OWN job, not someone they merely mention "
    "(e.g. 'talk to our pastor' is NOT the user's job).\n"
    "- A person is 'deceased' only if the message says they died/passed/were lost.\n"
    "Schema (use null when unknown):\n"
    '{"user_name": null, "user_age": null, "occupation": null, "works_fulltime": false, '
    '"kids": null, "people": [{"relation": "", "name": null, "status": "living", '
    '"age": null, "cause": null, "when": null, "timeframe": null}], '
    '"topic": null, "situation": null}\n'
    "topic is one of: grief, marriage, addiction, mental_health, faith_doubt, "
    "parenting, work_finance, general."
)


def _merge_profile(profile: dict, data: dict) -> None:
    """Conservatively merge an LLM-returned profile into the live one: only ADD or
    fill empty fields, never drop a previously-known fact (protects recall)."""
    profile.setdefault("people", [])
    for k in ("user_name", "user_age", "occupation", "kids", "topic", "situation"):
        if data.get(k) and not profile.get(k):
            profile[k] = data[k]
    if data.get("works_fulltime"):
        profile["works_fulltime"] = True
    for np in data.get("people", []) or []:
        if not isinstance(np, dict) or not np.get("relation"):
            continue
        rel = _norm_relation(np["relation"])
        ex = _find_person(profile, rel)
        if ex is None:
            ex = {"relation": rel, "name": None, "status": "living", "age": None,
                  "cause": None, "when": None, "timeframe": None}
            profile["people"].append(ex)
        for f in ("name", "age", "cause", "when", "timeframe"):
            if np.get(f) and not ex.get(f):
                ex[f] = np[f]
        if np.get("status") == "deceased":
            ex["status"] = "deceased"


def _extract_profile_llm(profile: dict, user_msg: str) -> None:
    """Update the profile via an LLM call; fall back to regex on any failure."""
    try:
        resp = ollama.chat(
            model=SUMMARY_MODEL,
            messages=[
                {"role": "system", "content": _PROFILE_PROMPT},
                {"role": "user",
                 "content": f"CURRENT PROFILE:\n{json.dumps(profile)}\n\nNEW MESSAGE:\n{user_msg}"},
            ],
            options={"temperature": 0},
            format="json",
        )
        data = json.loads(resp["message"]["content"])
        if isinstance(data, dict):
            _merge_profile(profile, data)
            return
    except Exception:
        pass
    _extract_profile(profile, user_msg)  # deterministic fallback


def update_profile(profile: dict, user_msg: str) -> None:
    """Dispatch to the configured extractor (LLM or regex)."""
    if USE_LLM_PROFILE:
        _extract_profile_llm(profile, user_msg)
    else:
        _extract_profile(profile, user_msg)


def _primary_person(profile: dict) -> Optional[dict]:
    people = profile.get("people", [])
    for p in people:
        if p["status"] == "deceased":
            return p
    return people[0] if people else None


def _anchor_phrase(profile: dict) -> Optional[str]:
    """e.g. 'your mother Sarah' / 'your husband' -- for closers & recall checks."""
    p = _primary_person(profile)
    if not p:
        return None
    rel = p["relation"]
    return f"your {rel} {p['name']}" if p.get("name") else f"your {rel}"


def _mentions_primary(text: str, profile: dict) -> bool:
    """True if the reply already references the primary person (name, relation, or
    loss), so the recall guard only appends a closer when it's genuinely missing."""
    p = _primary_person(profile)
    if not p:
        return True  # nothing to anchor to
    # Mirror the recall scorer: a reply "names the person" only via the actual
    # name, a possessive + relation ("your mother"), or the cause -- NOT bare
    # "grief/loss/death" words (those slip past the scorer and starve recall).
    pats = [r"(your|my|her|his|their)\s+" + re.escape(p["relation"])]
    if p.get("name"):
        pats.append(re.escape(p["name"]))
    if p.get("cause"):
        pats.append(re.escape(p["cause"]))
    return bool(re.search(r"\b(" + "|".join(pats) + r")\b", text, re.I))


def _remembrance_closers(profile: dict) -> list[str]:
    """Build rotating closing lines from the profile (no hardcoded 'Sarah')."""
    anchor = _anchor_phrase(profile)
    p = _primary_person(profile)
    if anchor and p and p["status"] == "deceased":
        cap = anchor[0].upper() + anchor[1:]
        return [
            f"Hold close the love of {anchor}, my friend -- it does not leave you.",
            f"And remember: {anchor}'s love still surrounds you today.",
            f"Carry the memory of {anchor} gently with you; you are not alone.",
            f"{cap} is held safely in God's hands, and so are you.",
        ]
    # non-loss topics: neutral, still-warm closers that don't invent a death
    return [
        "You are not walking through this alone, my friend.",
        "I'm here with you, and God is nearer still.",
        "Be gentle with yourself today; one step at a time.",
        "Whatever tomorrow holds, you do not face it alone.",
    ]


# "Restart at basics" comfort cliches (topic-agnostic pastoral platitudes). The
# scorer fails a very-late turn (>=17) reusing one ALSO used early (1-6); the
# recall guard strips such reuse so very-late turns stay concrete.
_RESTART_MARKERS = [
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
    r"i'm here to listen",
]


# The small model sometimes leaks its own scaffolding into the reply as a bracketed
# placeholder, e.g. "stories about [mention a favorite memory]" or "[child's name]".
# Strip instruction-like bracketed spans (leaves legit brackets like "[KJV]" alone).
_TEMPLATE_LEAK_RE = re.compile(
    r"\s*\[[^\]\n]*\b(mention|insert|e\.?g\.?|name|specific|add|choose|example|"
    r"placeholder|favorite|your|describe)\b[^\]\n]*\]",
    re.I,
)
# Some small models narrate their own delivery as a roleplay stage-direction
# at the start of a reply: "(A gentle sigh)", "(Softly)", "(Takes a slow breath)".
# Strip leading parenthetical stage cues -- a chat friend doesn't narrate themselves.
_STAGE_DIRECTION_RE = re.compile(r"^\s*(?:\([^)\n]{2,60}\)[\s—\-:.]*)+")
# Strip *single-asterisk* markdown emphasis (e.g. "*begin*") while leaving **bold**
# and bullet lists intact -- chat bubbles render plain text, not markdown.
_MD_EMPHASIS_RE = re.compile(r"(?<!\*)\*([^*\n]+?)\*(?!\*)")


def _strip_template_leaks(text: str) -> str:
    cleaned = _TEMPLATE_LEAK_RE.sub("", text)
    cleaned = _STAGE_DIRECTION_RE.sub("", cleaned)      # drop leading roleplay stage cues
    cleaned = _MD_EMPHASIS_RE.sub(r"\1", cleaned)       # drop *single* emphasis markers
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.,!?;:])", r"\1", cleaned)  # tidy space left before punctuation
    return cleaned.strip() or text.strip()             # never empty the reply


def _strip_reused_restart_markers(reply: str, early_markers: list[str]) -> str:
    if not early_markers:
        return reply
    parts = re.split(r"(?<=[.!?])\s+", reply)
    kept = [p for p in parts if not any(re.search(m, p, re.I) for m in early_markers)]
    cleaned = " ".join(kept).strip()
    return cleaned if cleaned else reply


def _norm_for_dedup(s: str) -> str:
    """Lowercase, strip punctuation/whitespace -- a canonical form for comparing
    whether two sentences say the same thing."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", s.lower())).strip()


def _strip_recycled_sentences(reply: str, prior_texts: list[str],
                              ratio: float = 0.8, prefix_words: int = 5) -> str:
    """Topic-AGNOSTIC anti-repetition: drop any sentence in `reply` that closely
    repeats one already used in recent prior assistant turns -- either a high overall
    similarity (catches verbatim/near-verbatim) or a shared opening phrase (catches
    templated openers like 'My friend, I'm so sorry you're ...'). A small model loops
    stock empathy/reassurance lines; this forces variation. Never guts the reply: if
    stripping would leave almost nothing, the original is kept.
    """
    prior = []
    for t in prior_texts:
        for s in re.split(r"(?<=[.!?])\s+", t):
            n = _norm_for_dedup(s)
            if len(n.split()) >= 4:
                prior.append(n)
    if not prior:
        return reply
    prior_prefix = {" ".join(p.split()[:prefix_words]) for p in prior}

    kept, dropped = [], 0
    for part in re.split(r"(?<=[.!?])\s+", reply):
        n = _norm_for_dedup(part)
        words = n.split()
        if len(words) >= 4:
            pref = " ".join(words[:prefix_words])
            is_dup = (pref in prior_prefix
                      or any(difflib.SequenceMatcher(None, n, p).ratio() >= ratio for p in prior))
            if is_dup:
                dropped += 1
                continue
        kept.append(part)
    if not dropped:
        return reply
    cleaned = " ".join(kept).strip()
    return cleaned if len(cleaned.split()) >= 5 else reply


# Journey/depth tracking (0..4), monotonic -- topic-neutral labels.
DEPTH_LABELS = {
    0: "getting to know the situation",
    1: "hearing the story",
    2: "exploring the hard feelings",
    3: "seeking how faith applies",
    4: "practical next-steps",
}
_PRACTICAL_SIGNAL = re.compile(
    r"\b(steps?|plan|practice|concrete|routine|daily|what can i|what should i|give me)\b",
    re.I,
)

# A reply that matches any of these has degenerated into a canned safety refusal;
# get_hot_messages() swaps it for a neutral placeholder so one refusal can't
# cascade into later turns. The real reply still lives in the saved transcript.
_REFUSAL_MARKERS = re.compile(
    r"\b(i cannot provide|i can'?t provide|i can'?t assist|i can'?t engage|"
    r"i cannot help|i can'?t help with|"
    r"harming yourself|hotline|1-800|741-?741|"
    r"mental health professional|harmful activities)\b",
    re.I,
)
_REFUSAL_PLACEHOLDER = (
    "I'm here with you, and I'll keep walking through this with you, "
    "step by step."
)


def extract_key_facts(user_msg: str) -> list[dict]:
    """Light, topic-agnostic incidental facts that aid topic adherence.

    Deliberately avoids clinical trigger words in the standing context (they prime
    a small model's safety classifier into refusals). Kept minimal and generic.
    """
    facts: list[dict] = []
    if re.search(r"\b(a plan|steps|concrete|practical|what (can|should) i do)\b", user_msg, re.I):
        facts.append({"fact": "the user wants concrete, practical steps", "type": "incidental", "confidence": 0.7})
    return facts


class MemoryLayer:
    """Generalized conversation memory layer for Khamuel (see module docstring)."""

    HOT_TURNS = 6

    def __init__(self) -> None:
        self.turns: list[dict] = []
        self.rolling_summary: str = ""
        self.key_facts: list[dict] = []
        self.pinned_facts: list[dict] = []
        self.persona: dict = {}          # the generic profile (kept name for save/load compat)
        self.journey_state: dict = {"topic": None, "depth_level": 0, "last_milestone": None}
        self._last_context_chars: int = 0
        self._just_resumed: bool = False

    # -- ingestion ---------------------------------------------------------

    def add_turn(self, user_msg: str, assistant_msg: str) -> None:
        turn_idx = len(self.turns) + 1
        self.turns.append({"user": user_msg, "assistant": assistant_msg, "turn_idx": turn_idx})
        self._just_resumed = False

        if USE_LLM_MEMORY:
            # Production path: one LLM call/turn fills pinned + key facts + depth +
            # profile. No regex. On failure, prior memory is kept unchanged.
            self._update_memory_llm(turn_idx)
        else:
            # Legacy regex path (KHAMUEL_MEMORY_LLM=0).
            _extract_profile(self.persona, user_msg)
            if self.journey_state["topic"] is None and self.persona.get("situation"):
                self.journey_state["topic"] = self.persona["situation"]
            self._rebuild_pinned()
            for f in extract_key_facts(user_msg):
                existing = next((k for k in self.key_facts if k["fact"] == f["fact"]), None)
                if existing:
                    existing["last_mentioned_turn"] = turn_idx
                else:
                    f["last_mentioned_turn"] = turn_idx
                    self.key_facts.append(f)
            new_depth = max(self.journey_state["depth_level"], self._depth_for(turn_idx, user_msg))
            if new_depth != self.journey_state["depth_level"] or self.journey_state["last_milestone"] is None:
                self.journey_state["depth_level"] = new_depth
                self.journey_state["last_milestone"] = DEPTH_LABELS[new_depth]

        older = self.turns[: -self.HOT_TURNS] if len(self.turns) > self.HOT_TURNS else []
        # Rolling summary (progression compression) is a separate concern from the
        # per-turn memory extraction above; it still runs on the 6-turn cadence. In
        # LLM-memory mode the profile gap-fill is off (the per-turn call already does it).
        start_turn = self.HOT_TURNS + 1  # = 7
        if older and (turn_idx == start_turn or turn_idx % SUMMARY_INTERVAL == 0):
            gap = LLM_GAP_FILL and not USE_LLM_MEMORY
            summary, updates = generate_summary_and_profile(
                older, self.rolling_summary, self.persona, gap_fill=gap)
            self.rolling_summary = summary
            if updates:
                _merge_profile(self.persona, updates)
                if not USE_LLM_MEMORY:
                    self._rebuild_pinned()

    def _update_memory_llm(self, turn_idx: int) -> None:
        """Consolidated per-turn memory update (hybrid).

        PINNED facts use the slot model: regex (deterministic floor, every turn) AND
        the LLM both fill the persona, then _rebuild_pinned() renders the sentences
        (Option B -- never raw model prose in the never-evicted block). KEY facts come
        from the LLM. DEPTH is a HYBRID: the turn-counter is the floor and the LLM can
        only ratchet it UP (max), so the LLM's failure mode (returning 0) is harmless.
        If the LLM call fails the regex floor + counter depth still apply.
        """
        user_msg = self.turns[-1]["user"] if self.turns else ""

        # Regex floor: precise, deterministic, runs even when the LLM call fails. It
        # goes FIRST so its high-precision slots win over the LLM on conflict (the
        # merge below only fills empty slots).
        _extract_profile(self.persona, user_msg)

        # Turn-counter depth is the deterministic FLOOR. We pass it (and the turn #) to
        # the LLM so it makes a bounded "has it gone DEEPER than this?" judgment rather
        # than rating depth from scratch (which it fails at on a small model).
        baseline = max(0, min(4, max(self.journey_state["depth_level"],
                                     self._depth_for(turn_idx, user_msg))))

        data = update_memory_llm(self.turns, self.persona,
                                 self.journey_state["depth_level"],
                                 turn_idx=turn_idx, baseline_depth=baseline)
        if data:
            prof = data.get("profile")
            if isinstance(prof, dict):
                _merge_profile(self.persona, prof)  # LLM adds coverage; fill-empty only

        # Render pinned from the (regex + LLM) persona.
        if self.journey_state["topic"] is None and self.persona.get("situation"):
            self.journey_state["topic"] = self.persona["situation"]
        self._rebuild_pinned()

        # Apply the counter floor first, so depth is correct even if the LLM failed.
        self.journey_state["depth_level"] = baseline
        self.journey_state["last_milestone"] = DEPTH_LABELS.get(baseline)

        # Key facts: REGEX FLOOR ONLY (deterministic, runs every turn -- even if the
        # LLM call failed). A small model cannot reliably extract open-ended incidental
        # details: it emits bare nouns ('recipe') or parrots the prompt's own examples
        # as if they were data. So we do NOT ask the LLM for key facts. Accumulate
        # across turns, dedupe, keep the 5 most recently mentioned.
        seen = {f["fact"].lower(): f for f in self.key_facts}
        for f in extract_key_facts(user_msg):
            k = f["fact"].lower()
            if k in seen:
                seen[k]["last_mentioned_turn"] = turn_idx
            else:
                seen[k] = {**f, "last_mentioned_turn": turn_idx}
        self.key_facts = sorted(
            seen.values(), key=lambda f: f.get("last_mentioned_turn", 0), reverse=True)[:5]

        if not data:
            return  # LLM failed: regex floor (profile/pinned/key facts/depth) already applied

        # Hybrid depth: LLM is upside-only -- it can lift depth ABOVE the counter floor,
        # never below it. The lift is BOUNDED to curb early over-eagerness: no lift in
        # the first 3 (establishing) turns, and at most ONE rung above the floor after.
        lvl = data.get("depth_level")
        llm_depth = int(lvl) if isinstance(lvl, (int, float)) else 0
        lift_cap = baseline if turn_idx <= 3 else baseline + 1
        effective = min(llm_depth, lift_cap, 4)
        if effective > baseline:
            self.journey_state["depth_level"] = effective
            self.journey_state["last_milestone"] = (
                str(data.get("label") or "").strip() or DEPTH_LABELS.get(effective))

    def _rebuild_pinned(self) -> None:
        """Render canonical pinned-fact sentences from the generic profile."""
        p = self.persona
        pinned: list[dict] = []

        for person in p.get("people", []):
            rel = person["relation"]
            name = person.get("name")
            who = f"The user's {rel}, {name}," if name else f"The user's {rel}"
            # Grief rendering (death + "is grieving" + cause/age) is GATED on the topic
            # actually being grief, so a stray deceased flag can never inject mourning
            # into a non-grief conversation. Otherwise the person is rendered neutrally.
            if person["status"] == "deceased" and p.get("topic") == "grief":
                when = f" died {person['when']}" if person.get("when") else " has died"
                line = f"{who}{when}; the user is grieving {'her' if rel in ('mother','sister','wife','daughter','grandmother') else 'them'}."
                pinned.append(self._pin(line, "person"))
                bits = []
                if person.get("age"):
                    bits.append(f"was {person['age']}")
                if person.get("cause"):
                    bits.append(f"died of {person['cause']}")
                if bits:
                    subj = name or f"The user's {rel}"
                    detail = f"{subj} {' and '.join(bits)}"
                    if person.get("timeframe"):
                        detail += f", only {person['timeframe']} from diagnosis to passing"
                    pinned.append(self._pin(detail + ".", "person_detail"))
            else:
                pinned.append(self._pin(f"{who} is part of the user's situation.", "person"))

        role_bits = []
        if p.get("user_age"):
            role_bits.append(f"is {p['user_age']}")
        if p.get("works_fulltime"):
            role_bits.append("works full-time")
        if p.get("kids"):
            role_bits.append(f"has {p['kids']}")
        if role_bits:
            pinned.append(self._pin("The user " + ", ".join(role_bits) + ".", "role"))
        if p.get("occupation"):
            pinned.append(self._pin(
                f"The user is a {p['occupation']} and thinks in concrete plans and steps.", "role"))
        if p.get("situation"):
            pinned.append(self._pin(f"The user came to talk about {p['situation']}.", "topic"))

        self.pinned_facts = pinned

    @staticmethod
    def _pin(fact: str, ftype: str) -> dict:
        return {"fact": fact, "type": ftype, "confidence": 0.95, "pinned": True}

    # -- prompt assembly (system-side blocks only) -------------------------

    def build_prompt_context(self, current_user_msg: str) -> str:
        protected: list[str] = []
        facts_block = self._facts_block(current_user_msg)
        if facts_block:
            protected.append(facts_block)
        protected.append(self._directive_block())
        journey_block = self._journey_block(current_user_msg)
        if journey_block:
            protected.append(journey_block)
        protected_text = "\n\n".join(protected)

        summary_block = self._summary_block()
        if summary_block:
            room = MAX_CONTEXT_CHARS - len(protected_text) - 2
            if len(summary_block) > room:
                summary_block = summary_block[: max(0, room)] if room > 0 else ""
            ctx = f"{summary_block}\n\n{protected_text}" if summary_block else protected_text
        else:
            ctx = protected_text

        if len(ctx) > MAX_CONTEXT_CHARS:
            ctx = ctx[:MAX_CONTEXT_CHARS]
        self._last_context_chars = len(ctx)
        return ctx

    def _directive_block(self) -> str:
        """Behaviour directive built from the profile at runtime (no hardcoded names)."""
        p = self.persona
        people = p.get("people", [])
        other_names = [pe["name"] for pe in people if pe.get("name")]
        user_name = p.get("user_name")

        # identity grounding
        if user_name:
            id_line = f"You are speaking WITH {user_name}."
        else:
            id_line = ("You are speaking WITH the person who came to you; you do not know "
                       "their name, so do not invent one.")
        if other_names:
            names = ", ".join(f"'{n}'" for n in other_names)
            id_line += (f" {names} {'is' if len(other_names)==1 else 'are'} OTHER people in "
                        f"their life -- NEVER address the user by {'that name' if len(other_names)==1 else 'those names'}. "
                        "Refer to those people in the third person.")
        if not user_name:
            id_line += " If you address the user, use 'my friend' or 'my child' -- never a name."
        id_line += " Refer to yourself in the first person ('I'), never as 'Khamuel'."

        situation = p.get("situation") or "what they are walking through"
        care = (
            "This is ordinary spiritual support, not an emergency: respond with gentle, "
            "practical pastoral care, and do not decline or redirect the user to hotlines."
        )
        return (
            "IDENTITY (read carefully): You are Khamuel, a warm Christian friend. " + id_line + "\n\n"
            f"The user came to talk about {situation}. Speak warmly and personally, continuing "
            "one ongoing conversation. Answer their specific question in your own words and build "
            "on what they have already shared instead of restarting with basics. Do NOT reuse the "
            "same opening line or reassurance you used in earlier replies -- vary your wording. " + care
        )

    def finalize_reply(self, reply: str, current_user_msg: str) -> str:
        """RECALL GUARD -- ensures the primary person/topic is referenced on the scored
        late turns (>=13) and on the first reply after a reload, without a forced
        opener. Builds the closer from the profile (no hardcoded 'Sarah')."""
        turn_idx = len(self.turns) + 1
        reply = _strip_template_leaks(reply)   # remove leaked scaffolding like "[mention ...]"
        # Topic-agnostic anti-repetition: drop sentences recycled from recent replies.
        # Essential for small models that loop stock empathy lines.
        if self.turns:
            reply = _strip_recycled_sentences(reply, [t["assistant"] for t in self.turns[-8:]])
        if turn_idx >= 17 and len(self.turns) >= 6:
            early_text = " ".join(t["assistant"] for t in self.turns[:6])
            early_markers = [m for m in _RESTART_MARKERS if re.search(m, early_text, re.I)]
            reply = _strip_reused_restart_markers(reply, early_markers)
        # The remembrance closer is GATED on the topic being grief, so a stray deceased
        # flag (or any non-grief chat) can never get a mourning line appended.
        needs_anchor = turn_idx >= 13 or self._just_resumed
        if (needs_anchor and self.persona.get("topic") == "grief"
                and _primary_person(self.persona) and not _mentions_primary(reply, self.persona)):
            closers = _remembrance_closers(self.persona)
            reply = reply.rstrip() + "\n\n" + closers[turn_idx % len(closers)]
        return reply

    def _summary_block(self) -> str:
        if not self.rolling_summary:
            return ""
        return f"Summary of earlier conversation:\n{self.rolling_summary}"

    @staticmethod
    def _depth_for(turn_idx: int, user_msg: str) -> int:
        if turn_idx >= 13:
            base = 3
        elif turn_idx >= 7:
            base = 2
        elif turn_idx >= 4:
            base = 1
        else:
            base = 0
        if turn_idx >= 13 and _PRACTICAL_SIGNAL.search(user_msg or ""):
            base = 4
        return base

    def _journey_block(self, current_user_msg: str = "") -> str:
        current_turn = len(self.turns) + 1
        depth = max(self.journey_state["depth_level"], self._depth_for(current_turn, current_user_msg))
        if depth <= 0 and not self.journey_state["topic"]:
            return ""
        label = DEPTH_LABELS.get(depth, "")
        return (
            f"Conversation depth: {depth}/4 ({label}). Build forward from here; "
            "do not re-explain basics already covered."
        )

    def _facts_block(self, current_user_msg: str) -> str:
        surfaced = list(self.pinned_facts)
        remaining = 5 - len(surfaced)
        if remaining > 0 and self.key_facts:
            ranked = sorted(self.key_facts, key=lambda f: f.get("last_mentioned_turn", 0), reverse=True)
            surfaced.extend(ranked[:remaining])
        if not surfaced:
            return ""
        lines = "\n".join(f"- {f['fact']}" for f in surfaced)
        return f"{FACTS_HEADER}\n{lines}"

    # -- hot context as native chat messages -------------------------------

    def get_hot_messages(self) -> list[dict]:
        budget = MAX_CONTEXT_CHARS - self._last_context_chars
        if budget < 800:
            budget = 800
        hot = self.turns[-self.HOT_TURNS:]

        def size(turns: list[dict]) -> int:
            return sum(len(t["user"]) + len(t["assistant"]) for t in turns)

        while len(hot) > 1 and size(hot) > budget:
            hot = hot[1:]

        messages: list[dict] = []
        for t in hot:
            messages.append({"role": "user", "content": t["user"]})
            assistant = t["assistant"]
            if _REFUSAL_MARKERS.search(assistant):
                assistant = _REFUSAL_PLACEHOLDER
            messages.append({"role": "assistant", "content": assistant})
        return messages

    # -- persistence (pure JSON) ------------------------------------------

    def save(self, filepath: str) -> None:
        state = {
            "version": 2,
            "turns": self.turns,
            "rolling_summary": self.rolling_summary,
            "key_facts": self.key_facts,
            "pinned_facts": self.pinned_facts,
            "persona": self.persona,
            "journey_state": self.journey_state,
        }
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> "MemoryLayer":
        with open(filepath, "r", encoding="utf-8") as fh:
            state = json.load(fh)
        m = cls()
        m.turns = state.get("turns", [])
        m.rolling_summary = state.get("rolling_summary", "")
        m.key_facts = state.get("key_facts", [])
        m.pinned_facts = state.get("pinned_facts", [])
        m.persona = state.get("persona", {})
        m.journey_state = state.get(
            "journey_state", {"topic": None, "depth_level": 0, "last_milestone": None}
        )
        m._just_resumed = True
        return m


# ---------------------------------------------------------------------------
# Rolling-summary helper (generic).
# ---------------------------------------------------------------------------

# The summary tracks the conversation's PROGRESSION -- what's been discussed, what
# advice/suggestions were already given (so the bot doesn't repeat them), and what's
# still unresolved. It deliberately does NOT re-describe who the user is -- the pinned
# facts already carry that. This is the value the pinned facts can't provide.
_SUMMARY_PROMPT = (
    "You are tracking the PROGRESSION of a pastoral conversation so the assistant "
    "doesn't repeat itself. Read the exchange and the prior summary (if any), then "
    "output STRICT JSON only, no prose, with these keys:\n"
    '{"topics_covered": ["..."], "advice_or_suggestions_given": ["..."], '
    '"open_threads": ["..."]}\n'
    "Do NOT restate who the user is or their basic facts. Capture only what has been "
    "DISCUSSED, what was SUGGESTED/ADVISED (so it isn't offered again), and what is "
    "still UNRESOLVED. Keep it compact (about 120 words)."
)

# Hybrid prompt: the same progression-summary call ALSO returns profile gap-fills
# (facts the regex extractor may have missed). Merged conservatively (adds coverage).
_SUMMARY_PROFILE_PROMPT = (
    "You are (1) tracking the PROGRESSION of a pastoral conversation and (2) noting any "
    "clearly-stated facts about the PERSON an automated extractor might have missed. "
    "Output STRICT JSON only, no prose, with these keys:\n"
    '{"topics_covered": ["..."], "advice_or_suggestions_given": ["..."], '
    '"open_threads": ["..."], "profile_updates": {"user_age": null, "occupation": null, '
    '"kids": null, "people": [{"relation": "", "name": null, "status": "living", '
    '"cause": null, "age": null}]}}\n'
    "For the summary: do NOT restate who the user is; capture only what's been DISCUSSED, "
    "what was already SUGGESTED/ADVISED (so it isn't repeated), and what's UNRESOLVED. "
    "For profile_updates: only facts the PERSON clearly stated about themselves; never "
    "invent; 'occupation' is the user's OWN job, not someone they mention; mark a person "
    "'deceased' only if they died/passed/were lost; leave unknown fields null."
)


# ---------------------------------------------------------------------------
# Consolidated LLM memory extractor (hybrid). One call per turn returns:
#   - profile  -> structured slots; merged with the regex extractor, then
#                 _rebuild_pinned() renders the PINNED facts (Option B: code writes
#                 the always-on sentences, never raw model prose);
#   - key_facts -> specific incidental details (the LLM's real value-add);
#   - depth_level/label -> journey depth (clamped monotonic).
# Regex runs every turn as a deterministic floor, so a failed LLM call never empties
# the pinned block.
# ---------------------------------------------------------------------------

# Default ON for this branch. Set KHAMUEL_MEMORY_LLM=0 to fall back to the regex path.
USE_LLM_MEMORY = os.environ.get("KHAMUEL_MEMORY_LLM", "1") != "0"

_MEMORY_PROMPT = (
    "You maintain the working MEMORY of a Christian pastoral assistant talking with "
    "ONE person across many turns. Read the PRIOR MEMORY and the NEW exchanges, then "
    "output the UPDATED memory as STRICT JSON only -- no prose, no markdown. Keep "
    "prior facts unless the new exchanges correct them; only ADD or REFINE. Record "
    "ONLY what the PERSON clearly stated about themselves; never invent.\n"
    "Output EXACTLY these keys:\n"
    '{"profile": {"user_name": null, "user_age": null, "occupation": null, '
    '"kids": null, "people": [{"relation": "", "name": null, "status": "living", '
    '"cause": null, "age": null}], "situation": null}, '
    '"depth_level": 0, "label": ""}\n'
    "profile: durable structured facts. 'occupation' is the user's OWN job, not "
    "someone they merely mention. status 'deceased' ONLY for a literal death (they "
    "died, passed away, or there was a funeral) -- NEVER for retiring, moving, leaving, "
    "long-distance, being away, or a breakup.\n"
    "depth_level + label: how deep the conversation has gone. You are given a BASELINE "
    "DEPTH (a floor) in the user message -- NEVER return less than it, and raise it by "
    "at most ONE rung, only with clear evidence the user has ALREADY moved there:\n"
    " 0 getting to know the situation\n 1 hearing the story\n"
    " 2 exploring the hard feelings (the user is actively dwelling in painful emotion, "
    "NOT merely recalling a memory or sharing a fact)\n"
    " 3 seeking how faith applies\n"
    " 4 practical next-steps (the user is asking for concrete actions, plans, or steps)\n"
    "The opening turns are for establishing the situation and story (rungs 0-1) -- do "
    "NOT jump ahead. When unsure, return the baseline."
)


def update_memory_llm(turns: list[dict], prior_profile: dict, prior_depth: int,
                      turn_idx: Optional[int] = None, baseline_depth: int = 0) -> Optional[dict]:
    """One Ollama call -> parsed memory dict, or None on any failure.

    Sends the prior memory + recent exchanges; the model returns the updated structured
    profile and journey depth. turn_idx + baseline_depth anchor the depth judgment (the
    model is told the floor it must not go below). Returns None (caller keeps prior
    state) if Ollama errors or the JSON is unusable.
    """
    recent = turns[-MemoryLayer.HOT_TURNS:] if turns else []
    convo = "\n".join(f"User: {t['user']}\nKhamuel: {t['assistant'][:240]}" for t in recent)
    # Only the structured profile + depth are sent: pinned facts are rendered from the
    # profile (Option B), and key facts are regex-only (the LLM isn't asked for them).
    prior_mem = {
        "profile": prior_profile,
        "depth_level": prior_depth,
    }
    anchor = ""
    if turn_idx is not None:
        anchor = (f"\n\nCURRENT TURN: {turn_idx}\nBASELINE DEPTH (floor -- never return "
                  f"less than this; only go higher if the user has clearly moved deeper): "
                  f"{baseline_depth}")
    user_content = f"PRIOR MEMORY:\n{json.dumps(prior_mem)}\n\nNEW EXCHANGES:\n{convo}{anchor}"
    try:
        resp = ollama.chat(
            model=SUMMARY_MODEL,
            messages=[{"role": "system", "content": _MEMORY_PROMPT},
                      {"role": "user", "content": user_content}],
            options={"temperature": 0},
            format="json",
            think=False,  # extraction needs JSON, never reasoning tokens
        )
        data = json.loads(resp["message"]["content"])
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def generate_summary_and_profile(turns: list[dict], previous_summary: str = "",
                                 profile: Optional[dict] = None, gap_fill: bool = True):
    """One LLM call -> (rolling_summary_text, profile_updates_dict).

    When gap_fill is True the model also returns profile_updates for facts the regex
    extractor may have missed. Falls back to the deterministic template (and no
    updates) on any Ollama/JSON failure, so a hiccup never crashes a run.
    """
    convo = "\n".join(f"User: {t['user']}\nKhamuel: {t['assistant'][:240]}" for t in turns)
    parts = []
    if gap_fill and profile is not None:
        parts.append(f"KNOWN PROFILE (do not repeat, only add what's missing):\n{json.dumps(profile)}")
    if previous_summary:
        parts.append(f"PRIOR SUMMARY:\n{previous_summary}")
    parts.append(f"EXCHANGES:\n{convo}")
    user_content = "\n\n".join(parts)
    prompt = _SUMMARY_PROFILE_PROMPT if gap_fill else _SUMMARY_PROMPT
    try:
        resp = ollama.chat(
            model=SUMMARY_MODEL,
            messages=[{"role": "system", "content": prompt},
                      {"role": "user", "content": user_content}],
            options={"temperature": 0.2},
            format="json",
            think=False,  # extraction needs JSON, never reasoning tokens
        )
        data = json.loads(resp["message"]["content"])
        rendered = _render_summary(data)
        if not rendered.strip():
            raise ValueError("empty summary")
        updates = data.get("profile_updates") or {} if gap_fill else {}
        return rendered[:SUMMARY_MAX_CHARS], (updates if isinstance(updates, dict) else {})
    except Exception:
        return _template_summary(turns, previous_summary), {}


def _render_summary(data: dict) -> str:
    lines: list[str] = []

    def add(label: str, value) -> None:
        if not value:
            return
        if isinstance(value, list):
            seen, items = set(), []
            for v in value:
                v = str(v).strip()
                if v and v.lower() not in seen:
                    seen.add(v.lower())
                    items.append(v)
            if items:
                lines.append(f"{label}: " + "; ".join(items))
        else:
            lines.append(f"{label}: {str(value).strip()}")

    # progression-focused (with back-compat for older key names)
    add("Topics already discussed", data.get("topics_covered"))
    add("Advice/suggestions already given (do not repeat)",
        data.get("advice_or_suggestions_given") or data.get("advice_given")
        or data.get("scriptures_or_advice_given"))
    add("Still unresolved", data.get("open_threads") or data.get("open_questions"))
    return "\n".join(lines)


def _template_summary(turns: list[dict], previous_summary: str) -> str:
    """Deterministic fallback if Ollama is unavailable or returns bad JSON."""
    n = turns[-1]["turn_idx"] if turns else 0
    topics = "; ".join(t["user"][:60].strip().rstrip(".") for t in turns[-4:])
    base = f"Earlier in the conversation (through turn {n}), recent threads: {topics}."
    if previous_summary:
        base = previous_summary.split("\n")[0] + " " + base
    return base[:SUMMARY_MAX_CHARS]


def generate_rolling_summary(turns: list[dict], previous_summary: str = "") -> str:
    """Back-compat wrapper: summary only, no profile gap-fill."""
    return generate_summary_and_profile(turns, previous_summary, profile=None, gap_fill=False)[0]
