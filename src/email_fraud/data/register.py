"""Lightweight heuristic register classifier — shared by data + generation code.

A single email is bucketed into one of three registers:

    "terse"   very short (< ~20 words); too little signal to call formal/casual
    "casual"  more casual-signal words than formal-signal words
    "formal"  otherwise (the Enron default)

This is deliberately cheap (set membership, no model) because it runs over the
whole training corpus to drive register-stratified episode sampling
(``samplers.py``) and over style-context examples during synthetic generation
(``scripts/generate_synthetic_emails.py``).  Both callers must agree on the
definition, so it lives here as the single source of truth.

Register-stratified sampling is the LLM-free replacement for the old
``cross_register`` LLM positives: instead of asking an LLM to fake a register
shift and storing the forgery under the real sender_id, we co-sample a sender's
*real* formal and casual emails into the same episode, so the contrastive loss
pulls genuine cross-register same-author pairs together.
"""

from __future__ import annotations

REGISTERS: tuple[str, ...] = ("formal", "casual", "terse")

# Below this word count an email is "terse" regardless of vocabulary — there is
# not enough text to reliably separate formal from casual.
_TERSE_MAX_WORDS = 20

_FORMAL_SIGNALS = frozenset({
    "pursuant", "attached", "regarding", "please", "review", "confirm",
    "schedule", "meeting", "contract", "agreement", "proposal", "transaction",
    "invoice", "deadline", "budget", "report", "analysis", "approved",
    "request", "forward", "update", "action", "item", "committee",
    "management", "department", "compliance", "policy", "procedure",
})

_CASUAL_SIGNALS = frozenset({
    "hey", "hi", "yeah", "yep", "nope", "lol", "haha", "btw", "fyi",
    "gonna", "wanna", "gotta", "kinda", "sorta", "ok", "okay",
    "awesome", "great", "cool", "fun", "nice", "congrats", "thanks",
    "dinner", "lunch", "weekend", "vacation", "holiday", "party",
    "friend", "family", "kids", "baby", "dog", "hope",
})


def detect_register(text: str) -> str:
    """Classify a single email as 'formal', 'casual', or 'terse'."""
    words = text.lower().split()
    if len(words) < _TERSE_MAX_WORDS:
        return "terse"
    word_set = set(words)
    formal_hits = len(word_set & _FORMAL_SIGNALS)
    casual_hits = len(word_set & _CASUAL_SIGNALS)
    if casual_hits > formal_hits:
        return "casual"
    return "formal"


def partition_by_register(texts: list[str]) -> dict[str, list[str]]:
    """Return a dict mapping register → list of texts (terse/casual/formal)."""
    buckets: dict[str, list[str]] = {r: [] for r in REGISTERS}
    for t in texts:
        buckets[detect_register(t)].append(t)
    return buckets
