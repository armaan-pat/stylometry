"""Generate synthetic emails for contrastive training.

Three generation modes are interleaved per sender:

  hard_neg  (stored as ``sender__syn``)
    Classic hard-negative: LLM mimics the sender's style on a business topic.
    Used by SyntheticBalancedSampler to guarantee a real/synthetic contrast pair
    in every batch.  Training objective: separate real-Alice from synthetic-Alice.

  cross_register  (stored under the real ``sender_id``)
    Cross-register positive: style examples are drawn from one register (formal
    or casual); the LLM is asked to write as the same person in the *opposite*
    register.  Stored under the real sender ID so SupConLoss treats them as
    genuine positives.  Training objective: same-author embeddings should cluster
    even when register shifts hard — directly addressing the #1 FN failure mode.

  cross_length  (stored under the real ``sender_id``)
    Cross-length positive: show the sender's (typically longer) emails and ask
    for a *terse* note in the same voice.  Stored under the real sender ID as a
    genuine positive.  Training objective: same-author embeddings should cluster
    even when length collapses — directly addressing the short-email failure
    mode (the model currently scores short same-author pairs as DIFFERENT and
    short different-author pairs as SAME; see results/error_analysis.txt).

Length conditioning (all modes)
    Every job samples a target length bucket from a short-heavy distribution
    (~60 % short, 25 % medium, 15 % long) and asks the LLM for roughly that many
    words.  Acceptance floors/ceilings are per-bucket so short outputs are KEPT
    (the old global --min-words 15 silently discarded every short email, leaving
    training with no short synthetics at all).  cross_length jobs are pinned to
    the short bucket.

LLM positives vs. real positives  (--llm-positives)
    cross_register and cross_length store LLM-generated text under the *real*
    sender_id, so SupConLoss treats them as genuine same-author positives.  That
    is a double-edged sword: it teaches register/length invariance, but it also
    teaches the encoder that a convincing LLM impersonation belongs in the real
    author's cluster — directly opposing the fraud signal.  Two variants:

      --llm-positives exclude   (real-positives dataset)
          Emit ONLY hard_neg rows (all stored as sid__syn).  No LLM text is ever
          stored under a real sender_id — enforced by an assertion at save time.
          Positives during training come solely from real emails; the synthetic
          set contributes negatives only.  cross_*_fraction are forced to 0.

      --llm-positives include   (mixed dataset, default)
          Current behaviour: LLM positives (cross_register + cross_length) AND
          LLM negatives (hard_neg), per the fractions below.

    Produce both from one corpus by running the script twice with different
    --output and --llm-positives values.

Splits and defaults  (only apply when --llm-positives include)
    --cross-register-fraction 0.4  → 40 % of n-per-sender are cross-register
    --cross-length-fraction   0.2  → 20 % are cross-length positives
                                      (remaining 40 % are hard negatives)

Quality gates (applied after preprocessing)
    per-bucket floor/ceiling   reject outputs outside the bucket's word range
                               (short bucket floor is 4 words, not 15)
    --min-words N              absolute floor applied on top of the bucket floor
    --max-overlap 0.40         reject outputs that are > 40 % Jaccard-overlap
                               with any style-context example (catches near-copies)

Usage (single local model, back-compat):
    python scripts/generate_synthetic_emails.py \\
        --config configs/base.yaml \\
        --model mistralai/Mistral-7B-Instruct-v0.3 \\
        --n-per-sender 15 --n-examples 5 \\
        --output data/synthetic/enron_synthetic_negonly \\
        --load-in-4bit

Usage (MIX of generators — recommended for diverse hard negatives):
    GROQ_API_KEY=... GEMINI_API_KEY=... \\
    python scripts/generate_synthetic_emails.py \\
        --config configs/base.yaml \\
        --generators groq:llama-3.3-70b-versatile \\
                     gemini:gemini-1.5-flash \\
                     hf:mistralai/Mistral-7B-Instruct-v0.3 \\
        --n-per-sender 15 --n-examples 5 \\
        --request-workers 6 \\
        --output data/synthetic/enron_synthetic_multigen

    Jobs are round-robin assigned across the generators and each row is tagged in
    the ``generator`` column, so one dataset spans several impersonator models.
    This (a) stops the detector from overfitting to one model's fingerprint and
    (b) lets you hold one generator out entirely as an OOD test set.

LLM positives are OFF by default (--llm-positives exclude): every row is a hard
negative (sid__syn). LLM text is NEVER stored under a real sender_id.

Backends (BACKEND:MODEL specs for --generators):
    hf:<repo>          local transformers (GPU; --load-in-4bit), e.g.
                       hf:mistralai/Mistral-7B-Instruct-v0.3 (strong, ~7 GB 4-bit)
                       hf:meta-llama/Llama-3.1-8B-Instruct
    groq:<model>       Groq API — FREE tier, fast (llama-3.3-70b-versatile, gemma2-9b-it)
    gemini:<model>     Google Gemini — FREE tier (gemini-1.5-flash, gemini-2.0-flash)
    openrouter:<model> OpenRouter — has a free model pool
    together:<model>   Together AI — cheap
    deepseek:<model>   DeepSeek — very cheap (deepseek-chat)
    openai:<model>     OpenAI — cheap tier (gpt-4o-mini)
    ollama:<model>     local Ollama daemon — free, no GPU deps in this script
    API backends read <BACKEND>_API_KEY from the environment (GEMINI_API_KEY or
    GOOGLE_API_KEY for gemini).

Output Arrow dataset columns:
    text                  preprocessed generated email body
    sender_id             real sid (cross_*) or sid__syn (hard_neg)
    source_sender_id      always the real sid
    generation_mode       "hard_neg" | "cross_register" | "cross_length"
    topic                 topic string passed to the LLM
    context_register      register of style-context examples: "formal" | "casual" | "mixed"
    target_len_bucket     requested length bucket: "short" | "medium" | "long"
    generator             "<backend>:<model>" that produced the row (diversity audits / OOD splits)
    source_split          real split the senders came from: "train" | "validation" | "test"
                          (test/validation = eval-only, unseen-sender impersonation)

Requirements:
    pip install transformers accelerate bitsandbytes datasets
    (bitsandbytes only needed with --load-in-4bit)
"""

from __future__ import annotations

import argparse
import random
import time
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, load_from_disk
from tqdm import tqdm

from email_fraud.config import load_config
from email_fraud.data.preprocessing import preprocess
from email_fraud.data.register import detect_register, partition_by_register

# Generation backends (HF + API). torch / transformers / requests are imported
# lazily inside the backends, so an API-only run needs no GPU stack and an
# HF-only run needs no `requests`.
from llm_backends import build_generators, parse_spec

# ---------------------------------------------------------------------------
# Topic pools — sampled per generation to prevent topic leakage from examples.
# ---------------------------------------------------------------------------

# Professional / business register — used for hard_neg and casual→formal cross.
_BUSINESS_TOPICS = [
    "a scheduling conflict for next week",
    "a quarterly budget update",
    "an equipment or software request",
    "following up on an overdue invoice",
    "requesting feedback on a document",
    "a data access or permissions issue",
    "travel arrangements for a conference",
    "a vendor contract renewal",
    "a new hire starting on the team",
    "a client escalation that needs handling",
    "a project milestone being delayed",
    "a policy change or HR announcement",
    "an IT outage or system maintenance window",
    "clarifying action items from a meeting",
    "asking about the status of a pending report",
    "a last-minute change to a deliverable",
    "requesting an extension on a deadline",
    "flagging a discrepancy in a spreadsheet",
    "scheduling a performance review",
    "a compliance question about a recent transaction",
    "following up after a client call",
    "announcing a team reorganization",
    "requesting approval for a purchase",
    "a contract clause that needs legal review",
]

# Personal / casual register — used for formal→casual cross-register generation.
_PERSONAL_TOPICS = [
    "weekend plans with a friend",
    "congratulating a colleague on personal news",
    "catching up after not being in touch for a while",
    "recommending a restaurant or book",
    "a funny thing that happened at work",
    "planning a group get-together or celebration",
    "asking a friend for advice on a personal decision",
    "venting about a frustrating situation",
    "a short thank-you for a favour",
    "sharing excitement about an upcoming trip",
    "checking in on someone who has been unwell",
    "a quick update on how things are going",
    "asking whether someone is free over the holidays",
    "a lighthearted complaint about the weather or commute",
    "following up after running into someone unexpectedly",
]

# Terse / minimal-form — a small pool that produces short, signal-sparse emails,
# which are over-represented in the test set and over-contribute to FP errors.
_TERSE_TOPICS = [
    "a one-sentence confirmation that something was received",
    "a two-sentence reply agreeing to a meeting time",
    "a brief note that you are out of office",
    "forwarding something with a single line of context",
    "a quick heads-up that a file was sent",
]

# ---------------------------------------------------------------------------
# Length conditioning.
#
# Production has to score short emails, but the old pipeline rejected every
# output under 15 words, so training saw zero short synthetics.  We now sample a
# target length per job from a short-heavy distribution and accept/reject
# against per-bucket floors and ceilings (in *words*, measured post-preprocess).
#
#   bucket   word range   floor   ceil    weight
#   short    5–25         4       45      0.60
#   medium   25–80        15      140     0.25
#   long     80–250       40      400     0.15
#
# floor/ceil are deliberately looser than the prompt range: LLMs only roughly
# honour a word target, and we'd rather keep a 30-word "short" email than reject
# it.  The ceil still guarantees the short bucket stays genuinely short.
# ---------------------------------------------------------------------------

_LENGTH_BUCKETS: dict[str, tuple[int, int]] = {
    "short":  (5, 25),
    "medium": (25, 80),
    "long":   (80, 250),
}
_LENGTH_WEIGHTS: dict[str, float] = {"short": 0.60, "medium": 0.25, "long": 0.15}
_LENGTH_FLOOR:   dict[str, int]   = {"short": 4,  "medium": 15,  "long": 40}
_LENGTH_CEIL:    dict[str, int]   = {"short": 45, "medium": 140, "long": 400}


def _sample_length(rng: random.Random, force: str | None = None) -> tuple[str, int]:
    """Pick a length bucket (weighted, or forced) and a target word count in it."""
    if force is not None:
        bucket = force
    else:
        buckets = list(_LENGTH_WEIGHTS.keys())
        bucket = rng.choices(buckets, weights=[_LENGTH_WEIGHTS[b] for b in buckets])[0]
    lo, hi = _LENGTH_BUCKETS[bucket]
    return bucket, rng.randint(lo, hi)


def _length_instruction(target_words: int) -> str:
    """One-line length directive injected into every prompt."""
    return (
        f"Length: aim for about {target_words} words. Match the email to that "
        "length naturally — a terse one- or two-line note is perfectly fine if "
        "short. Do not pad it out to seem complete."
    )


# Register detection (formal / casual / terse) lives in
# email_fraud.data.register so the generator and the register-stratified episode
# sampler share one definition; imported above as detect_register /
# partition_by_register.


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_hard_neg_prompt(examples: list[str], topic: str, target_words: int) -> str:
    """Classic hard-negative prompt: mimic style, write on a given topic."""
    joined = "\n---\n".join(examples)
    return (
        "You are imitating the writing style of one person based on their emails.\n\n"
        f"Here are examples of their writing:\n---\n{joined}\n---\n\n"
        "Write a new email they might send. Faithfully reproduce:\n"
        "- Sentence length and rhythm\n"
        "- Punctuation habits (comma splices, ellipses, run-ons, capitalisation quirks)\n"
        "- Greeting and sign-off style\n"
        "- Level of formality and hedging language\n"
        "- Any characteristic vocabulary, filler phrases, or personal quirks\n\n"
        f"Write about: {topic}\n"
        f"{_length_instruction(target_words)}\n"
        "Do NOT copy any sentences from the examples above.\n"
        "Output only the email body. No subject line, no metadata."
    )


def _build_cross_length_prompt(examples: list[str], topic: str, target_words: int) -> str:
    """Cross-length positive prompt: same voice, collapsed to a terse note.

    Shows the sender's (usually longer) emails and asks for a brief message in
    the same voice.  Stored under the real sender_id so SupConLoss treats it as a
    positive — teaching the encoder that an author's short notes belong in the
    same cluster as their long ones.  This is the length analogue of the
    cross-register trick and directly targets the short-email failure mode.
    """
    joined = "\n---\n".join(examples)
    return (
        "You are imitating the writing style of a specific person.\n\n"
        f"Here are examples of their writing:\n---\n{joined}\n---\n\n"
        "This same person is now dashing off a SHORT, quick email — the kind of "
        "terse one- or two-line note people fire off without much thought.\n"
        "The message is brief, but their underlying VOICE stays the same: keep "
        "their punctuation habits, greeting/sign-off style, capitalisation "
        "quirks, and characteristic phrasing even in a tiny message.\n\n"
        f"Write about: {topic}\n"
        f"{_length_instruction(target_words)}\n"
        "Do NOT copy any sentences from the examples above.\n"
        "Output only the email body. No subject line, no metadata."
    )


def _build_cross_register_prompt(
    examples: list[str],
    context_register: str,
    topic: str,
    target_words: int,
) -> str:
    """Cross-register prompt: show examples from one register, write in the other.

    The key instruction is that the author's *voice* (rhythm, punctuation,
    characteristic phrases) must survive the register shift.  This is exactly
    the invariance we want the encoder to learn.
    """
    joined = "\n---\n".join(examples)

    if context_register == "formal":
        register_instruction = (
            "This same person is now writing a casual, informal email — to a friend, "
            "family member, or close colleague outside a work context.\n"
            "The REGISTER shifts (informal, relaxed, personal) but their underlying "
            "VOICE stays the same: preserve their sentence rhythm, punctuation habits, "
            "characteristic phrases, and personal quirks."
        )
    elif context_register == "casual":
        register_instruction = (
            "This same person is now writing a professional business email — to a "
            "colleague, manager, or external contact.\n"
            "The REGISTER shifts (formal, structured, professional) but their underlying "
            "VOICE stays the same: preserve their sentence rhythm, punctuation habits, "
            "characteristic phrases, and personal quirks."
        )
    else:  # mixed
        register_instruction = (
            "This same person is now writing a casual, personal email to a friend.\n"
            "Preserve their characteristic voice — sentence rhythm, punctuation style, "
            "typical phrases — while shifting to an informal register."
        )

    return (
        "You are imitating the writing style of a specific person.\n\n"
        f"Here are examples of their writing:\n---\n{joined}\n---\n\n"
        f"{register_instruction}\n\n"
        f"Write about: {topic}\n"
        f"{_length_instruction(target_words)}\n"
        "Do NOT copy any sentences from the examples above.\n"
        "Output only the email body. No subject line, no metadata."
    )


# ---------------------------------------------------------------------------
# Quality filter
# ---------------------------------------------------------------------------

def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _quality_ok(
    text: str,
    examples: list[str],
    min_words: int,
    max_overlap: float,
    max_words: int | None = None,
) -> bool:
    """Return True if the generated text passes basic quality gates.

    Rejects:
      - Outputs shorter than min_words words.
      - Outputs longer than max_words words (per-bucket ceiling; keeps the short
        bucket genuinely short when the LLM overshoots the requested length).
      - Outputs that are close paraphrases / near-copies of any style-context
        example (Jaccard overlap > max_overlap on word sets).
    """
    words = text.split()
    if len(words) < min_words:
        return False
    if max_words is not None and len(words) > max_words:
        return False

    if max_overlap < 1.0:
        text_words = set(w.lower() for w in words)
        for ex in examples:
            ex_words = set(w.lower() for w in ex.split())
            if _jaccard(text_words, ex_words) > max_overlap:
                return False

    return True


# Generation backends (HF + API) live in scripts/llm_backends.py; built via
# build_generators() in main() and round-robin-assigned to jobs.


# ---------------------------------------------------------------------------
# Per-sender job planning
# ---------------------------------------------------------------------------

def _plan_sender_jobs(
    sid: str,
    texts: list[str],
    n_per_sender: int,
    cross_register_fraction: float,
    cross_length_fraction: float,
    n_examples: int,
    min_words: int,
    rng: random.Random,
) -> list[dict]:
    """Return a list of generation job descriptors for one sender.

    Each job is a dict with keys:
        prompt          str    full LLM prompt
        sender_id       str    where to store the result (real sid or sid__syn)
        topic           str
        context_register str
        mode            str    "hard_neg" | "cross_register" | "cross_length"
        target_len_bucket str  "short" | "medium" | "long"
        min_words       int    per-job acceptance floor (bucket floor ∨ global)
        max_words       int    per-job acceptance ceiling (bucket ceil)
        examples        list[str]   the style-context texts used (for quality filter)
    """
    n_cross_len = round(cross_length_fraction * n_per_sender)
    n_cross_reg = round(cross_register_fraction * n_per_sender)
    n_hard = n_per_sender - n_cross_reg - n_cross_len

    by_register = partition_by_register(texts)
    formal_pool = by_register["formal"]
    casual_pool = by_register["casual"]

    jobs: list[dict] = []

    def _length_fields(bucket: str) -> dict:
        return {
            "target_len_bucket": bucket,
            "min_words": max(min_words, _LENGTH_FLOOR[bucket]),
            "max_words": _LENGTH_CEIL[bucket],
        }

    # --- Hard-negative jobs (length sampled short-heavy) ---
    for _ in range(n_hard):
        n_ex = min(n_examples, len(texts))
        ex = rng.sample(texts, n_ex)
        ex_trunc = [e[:600] for e in ex]
        topic = rng.choice(_BUSINESS_TOPICS)
        bucket, target_words = _sample_length(rng)
        jobs.append({
            "prompt": _build_hard_neg_prompt(ex_trunc, topic, target_words),
            "sender_id": f"{sid}__syn",
            "topic": topic,
            "context_register": "mixed",
            "mode": "hard_neg",
            "examples": ex_trunc,
            **_length_fields(bucket),
        })

    # --- Cross-register jobs (length sampled short-heavy) ---
    for _ in range(n_cross_reg):
        # Prefer to show examples from whichever register is more represented,
        # then ask the LLM to write in the opposite one.
        if len(formal_pool) >= max(2, n_examples // 2):
            # Sender mostly writes formally → ask for a casual email.
            n_ex = min(n_examples, len(formal_pool))
            ex = rng.sample(formal_pool, n_ex)
            context_reg = "formal"
            topic = rng.choice(_PERSONAL_TOPICS)
        elif len(casual_pool) >= max(2, n_examples // 2):
            # Sender mostly writes casually → ask for a formal email.
            n_ex = min(n_examples, len(casual_pool))
            ex = rng.sample(casual_pool, n_ex)
            context_reg = "casual"
            topic = rng.choice(_BUSINESS_TOPICS)
        else:
            # Sender has only terse emails or a mix with no clear majority —
            # fall back to any available texts and target a personal topic.
            n_ex = min(n_examples, len(texts))
            ex = rng.sample(texts, n_ex)
            context_reg = "mixed"
            topic = rng.choice(_PERSONAL_TOPICS)

        ex_trunc = [e[:400] for e in ex]  # slightly shorter to leave room for register instruction
        bucket, target_words = _sample_length(rng)
        jobs.append({
            "prompt": _build_cross_register_prompt(ex_trunc, context_reg, topic, target_words),
            "sender_id": sid,          # real ID → treated as positive in contrastive loss
            "topic": topic,
            "context_register": context_reg,
            "mode": "cross_register",
            "examples": ex_trunc,
            **_length_fields(bucket),
        })

    # --- Cross-length jobs (pinned to the short bucket) ---
    for _ in range(n_cross_len):
        n_ex = min(n_examples, len(texts))
        ex = rng.sample(texts, n_ex)
        ex_trunc = [e[:600] for e in ex]
        topic = rng.choice(_TERSE_TOPICS)
        bucket, target_words = _sample_length(rng, force="short")
        jobs.append({
            "prompt": _build_cross_length_prompt(ex_trunc, topic, target_words),
            "sender_id": sid,          # real ID → positive (short note in same voice)
            "topic": topic,
            "context_register": "mixed",
            "mode": "cross_length",
            "examples": ex_trunc,
            **_length_fields(bucket),
        })

    return jobs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic emails (hard negatives + cross-register / cross-length positives)"
    )
    parser.add_argument("--config",     required=True)
    parser.add_argument(
        "--generators",
        nargs="+",
        default=None,
        metavar="BACKEND:MODEL",
        help="One or more generation backends, e.g. "
             "'groq:llama-3.3-70b-versatile gemini:gemini-1.5-flash "
             "hf:mistralai/Mistral-7B-Instruct-v0.3'. Jobs are round-robin "
             "assigned across them so a single dataset contains a MIX of "
             "generators (each row tagged in the 'generator' column) — the point "
             "is to keep the detector from overfitting to one model's fingerprint "
             "and to enable held-out-generator OOD evaluation. Backends: hf, groq, "
             "openrouter, together, deepseek, openai, gemini, ollama. API backends "
             "read <BACKEND>_API_KEY from the environment. If omitted, falls back "
             "to a single hf:<--model> generator.",
    )
    parser.add_argument(
        "--model",
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="Back-compat shorthand: HuggingFace model ID for a single local "
             "generator, used only when --generators is not given "
             "(default: Mistral-7B-Instruct-v0.3).",
    )
    parser.add_argument("--request-workers", type=int, default=4,
                        help="Concurrent in-flight requests per API backend (default: 4). "
                             "Ignored by the hf backend, which batches on-device.")
    parser.add_argument("--from-split", choices=["train", "validation", "test"], default="train",
                        help="Which processed split to draw real senders/style examples from "
                             "(default: train). Use 'test' to build UNSEEN-SENDER impersonation "
                             "negatives for OOD eval (eval-only — see warning at runtime; do not "
                             "use such a dataset as a training augmentation path).")
    parser.add_argument("--n-per-sender",  type=int,   default=15,
                        help="Total synthetic emails to generate per sender (default: 15)")
    parser.add_argument("--n-examples",    type=int,   default=5,
                        help="Real emails to use as style context per generation (default: 5)")
    parser.add_argument("--llm-positives", choices=["include", "exclude"], default="exclude",
                        help="exclude (DEFAULT): emit ONLY hard_neg negatives (sid__syn); no LLM "
                             "text is stored under a real sender_id (real-positives dataset). This "
                             "is the project invariant — LLM text is a hard negative, never a "
                             "positive. include: ALSO emit cross_register + cross_length LLM "
                             "positives (mixed dataset). DEPRECATED ablation only: storing an LLM "
                             "impersonation under a real sender_id teaches the encoder that a "
                             "convincing forgery belongs in the author's cluster, opposing the "
                             "fraud signal. SyntheticAugmentedDataset drops these rows at train "
                             "time unless llm_negatives_only=False.")
    parser.add_argument("--cross-register-fraction", type=float, default=0.4,
                        help="Fraction of n-per-sender to generate as cross-register positives "
                             "(stored under real sender_id). Default: 0.4. "
                             "Forced to 0 when --llm-positives exclude.")
    parser.add_argument("--cross-length-fraction", type=float, default=0.2,
                        help="Fraction of n-per-sender to generate as cross-length (short, "
                             "same-voice) positives, stored under real sender_id. Default: 0.2. "
                             "cross_register + cross_length fractions must sum to < 1.0. "
                             "Forced to 0 when --llm-positives exclude.")
    parser.add_argument("--min-words",     type=int,   default=4,
                        help="Absolute minimum word count to accept a generated email "
                             "(default: 4). Applied on top of the per-bucket floor; the short "
                             "bucket floor is 4, so the old global 15 is no longer the default — "
                             "short emails are now kept.")
    parser.add_argument("--max-overlap",   type=float, default=0.40,
                        help="Maximum Jaccard overlap with any style-context example (default: 0.40)")
    parser.add_argument("--output",        required=True)
    parser.add_argument("--load-in-4bit",  action="store_true")
    parser.add_argument("--batch-size",    type=int,   default=4)
    parser.add_argument("--max-senders",   type=int,   default=None)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--wandb",         action="store_true",
                        help="Log acceptance stats + dataset composition to W&B. "
                             "Honors WANDB_RUN_GROUP / WANDB_JOB_TYPE / WANDB_NAME.")
    args = parser.parse_args()

    # Resolve LLM-positive policy into the *effective* cross-* fractions used
    # everywhere below. In `exclude` mode no LLM text may land under a real
    # sender_id, so both cross-* modes are zeroed (enforced again at save time).
    if args.llm_positives == "exclude":
        cr_frac, cl_frac = 0.0, 0.0
        variant = "real-positives (LLM negatives only)"
        if args.cross_register_fraction or args.cross_length_fraction:
            print(
                "[llm-positives=exclude] Overriding cross_register_fraction="
                f"{args.cross_register_fraction} and cross_length_fraction="
                f"{args.cross_length_fraction} → 0.0; emitting hard_neg only."
            )
    else:
        cr_frac, cl_frac = args.cross_register_fraction, args.cross_length_fraction
        variant = "mixed (LLM positives + LLM negatives)"

    if cr_frac + cl_frac >= 1.0:
        parser.error(
            "cross-register-fraction + cross-length-fraction must be < 1.0 "
            "(the remainder is the hard-negative fraction); got "
            f"{cr_frac} + {cl_frac}"
        )

    # Resolve generation backends. --generators (a mix) takes precedence; the
    # single --model is the back-compat fallback (one local HF generator).
    gen_specs = args.generators or [f"hf:{args.model}"]
    for spec in gen_specs:
        parse_spec(spec)  # fail fast on a bad spec before loading anything heavy

    rng = random.Random(args.seed)
    cfg = load_config(args.config)

    print(f"Loading '{args.from_split}' split from {cfg.data.processed_dir}")
    ds_dict = load_from_disk(cfg.data.processed_dir)
    if args.from_split not in ds_dict:
        parser.error(
            f"split '{args.from_split}' not in {cfg.data.processed_dir} "
            f"(available: {list(ds_dict.keys())})."
        )
    src_ds = ds_dict[args.from_split]
    if args.from_split != "train":
        print(
            f"  WARNING: generating from the '{args.from_split}' split. These synthetics "
            "are for EVALUATION ONLY (unseen-sender impersonation slices in "
            "build_ood_eval.py). Do NOT point a training config's "
            "data.augmentation.synthetic_path at this dataset — it would leak "
            f"{args.from_split} senders into training. The output carries a "
            f"source_split='{args.from_split}' column to make the provenance explicit."
        )

    sender_to_texts: dict[str, list[str]] = defaultdict(list)
    for text, sid in zip(src_ds["text"], src_ds["sender_id"]):
        sender_to_texts[sid].append(text)

    senders = sorted(sender_to_texts.keys())
    if args.max_senders:
        senders = senders[: args.max_senders]

    n_cross_len_target = round(cl_frac * args.n_per_sender)
    n_cross_reg_target = round(cr_frac * args.n_per_sender)
    n_hard_target      = args.n_per_sender - n_cross_reg_target - n_cross_len_target
    print(
        f"VARIANT: {variant}\n"
        f"Found {len(senders)} senders.  "
        f"Per sender: {n_hard_target} hard-neg + {n_cross_reg_target} cross-register "
        f"+ {n_cross_len_target} cross-length = {args.n_per_sender} total.  "
        f"Building {len(gen_specs)} generator(s): {', '.join(gen_specs)} ..."
    )
    generators = build_generators(
        gen_specs,
        load_in_4bit=args.load_in_4bit,
        request_workers=args.request_workers,
    )
    n_gen = len(generators)
    gen_by_name: dict[str, int] = {g.name: 0 for g in generators}

    # Output accumulators
    out_texts:            list[str] = []
    out_sender_ids:       list[str] = []
    out_source_sender_ids: list[str] = []
    out_modes:            list[str] = []
    out_topics:           list[str] = []
    out_context_regs:     list[str] = []
    out_buckets:          list[str] = []
    out_generators:       list[str] = []

    preprocess_cfg = cfg.data.preprocessing

    # Per-mode and per-bucket accept/generate counters.
    _MODES = ("hard_neg", "cross_register", "cross_length")
    gen_by_mode: dict[str, int] = {m: 0 for m in _MODES}
    acc_by_mode: dict[str, int] = {m: 0 for m in _MODES}
    gen_by_bucket: dict[str, int] = {b: 0 for b in _LENGTH_BUCKETS}
    acc_by_bucket: dict[str, int] = {b: 0 for b in _LENGTH_BUCKETS}
    total_generated  = 0
    total_accepted   = 0

    total_to_generate = len(senders) * args.n_per_sender
    gen_start = time.monotonic()
    job_counter = 0  # global, drives round-robin generator assignment

    overall_bar = tqdm(total=total_to_generate, desc="Overall", unit="email", dynamic_ncols=True)
    sender_bar  = tqdm(senders, desc="Senders", unit="sender", dynamic_ncols=True, leave=True)

    for sid in sender_bar:
        texts = sender_to_texts[sid]
        if len(texts) < args.n_examples:
            sender_bar.write(f"  skip {sid}: only {len(texts)} emails (need {args.n_examples})")
            overall_bar.update(args.n_per_sender)
            continue

        # Build all generation jobs for this sender up front.
        jobs = _plan_sender_jobs(
            sid, texts,
            n_per_sender=args.n_per_sender,
            cross_register_fraction=cr_frac,
            cross_length_fraction=cl_frac,
            n_examples=args.n_examples,
            min_words=args.min_words,
            rng=rng,
        )
        # Round-robin a generator onto each job (global counter → even split
        # across the whole run, so every sender gets a mix of generators rather
        # than the first sender monopolising generator 0).
        gen_idx_of: list[int] = []
        for job in jobs:
            gi = job_counter % n_gen
            gen_idx_of.append(gi)
            job["generator_name"] = generators[gi].name
            job_counter += 1

        # Group jobs by generator so each backend still runs its prompts in
        # batches, then scatter results back into job order.
        raw_outputs: list[str] = [""] * len(jobs)
        jobs_by_gen: dict[int, list[int]] = defaultdict(list)
        for i, gi in enumerate(gen_idx_of):
            jobs_by_gen[gi].append(i)

        batch_bar = tqdm(
            total=len(jobs), desc=f"  {sid[:30]}", unit="email",
            leave=False, dynamic_ncols=True,
        )
        for gi, job_idxs in jobs_by_gen.items():
            gen = generators[gi]
            for bstart in range(0, len(job_idxs), args.batch_size):
                chunk = job_idxs[bstart : bstart + args.batch_size]
                batch_prompts = [jobs[i]["prompt"] for i in chunk]
                t0 = time.monotonic()
                outs = gen.generate_batch(batch_prompts)
                elapsed_batch = time.monotonic() - t0
                for i, out in zip(chunk, outs):
                    raw_outputs[i] = out
                rate = len(chunk) / elapsed_batch if elapsed_batch > 0 else 0.0
                overall_bar.update(len(chunk))
                batch_bar.update(len(chunk))
                batch_bar.set_postfix(gen=gen.name[:18], rate=f"{rate:.2f}e/s")
        batch_bar.close()

        # Filter and store accepted outputs.
        sid_accepted = 0
        sid_cross_len_accepted = 0
        for raw, job in zip(raw_outputs, jobs):
            mode = job["mode"]
            bucket = job["target_len_bucket"]
            total_generated += 1
            gen_by_mode[mode] += 1
            gen_by_bucket[bucket] += 1

            cleaned = preprocess(raw, preprocess_cfg)
            if cleaned is None:
                continue
            if not _quality_ok(
                cleaned, job["examples"], job["min_words"], args.max_overlap,
                max_words=job["max_words"],
            ):
                continue

            out_texts.append(cleaned)
            out_sender_ids.append(job["sender_id"])
            out_source_sender_ids.append(sid)
            out_modes.append(mode)
            out_topics.append(job["topic"])
            out_context_regs.append(job["context_register"])
            out_buckets.append(bucket)
            out_generators.append(job["generator_name"])
            gen_by_name[job["generator_name"]] += 1

            total_accepted += 1
            acc_by_mode[mode] += 1
            acc_by_bucket[bucket] += 1
            sid_accepted += 1
            if mode == "cross_length":
                sid_cross_len_accepted += 1

        overall_rate = total_accepted / total_generated if total_generated else 0.0
        elapsed = time.monotonic() - gen_start
        avg_rate = total_generated / elapsed if elapsed > 0 else 0.0
        overall_bar.set_postfix(
            accepted=total_accepted,
            accept_rate=f"{overall_rate:.0%}",
            rate=f"{avg_rate:.2f}e/s",
        )
        sender_bar.write(
            f"  {sid}: {sid_accepted}/{len(jobs)} accepted "
            f"(cross_length: {sid_cross_len_accepted}/{n_cross_len_target})"
        )

    overall_bar.close()
    sender_bar.close()

    def _rate(mode: str) -> float:
        return acc_by_mode[mode] / gen_by_mode[mode] if gen_by_mode[mode] else 0.0

    print(f"\nDone. {total_accepted}/{total_generated} accepted overall.")
    for m in _MODES:
        print(f"  {m:<15} {acc_by_mode[m]}/{gen_by_mode[m]} ({_rate(m):.0%})")
    print("  by length bucket (accepted/generated):")
    for b in _LENGTH_BUCKETS:
        br = acc_by_bucket[b] / gen_by_bucket[b] if gen_by_bucket[b] else 0.0
        print(f"    {b:<6} {acc_by_bucket[b]}/{gen_by_bucket[b]} ({br:.0%})")
    print("  by generator (accepted rows — use this column for OOD held-out splits):")
    for name, cnt in sorted(gen_by_name.items()):
        share = cnt / total_accepted if total_accepted else 0.0
        print(f"    {name:<45} {cnt} ({share:.0%})")

    # Hard guarantee: in real-positives mode, no LLM text may be stored under a
    # real sender_id. Every emitted row must be a hard_neg (sid__syn).
    if args.llm_positives == "exclude":
        bad = [s for s in out_sender_ids if not s.endswith("__syn")]
        assert not bad and acc_by_mode["cross_register"] == 0 and acc_by_mode["cross_length"] == 0, (
            f"llm-positives=exclude violated: {len(bad)} rows stored under a real "
            "sender_id (expected all sid__syn)."
        )
        print(f"[llm-positives=exclude] Verified: all {total_accepted} rows are hard_neg (sid__syn).")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_dict({
        "text":             out_texts,
        "sender_id":        out_sender_ids,
        "source_sender_id": out_source_sender_ids,
        "generation_mode":  out_modes,
        "topic":            out_topics,
        "context_register": out_context_regs,
        "target_len_bucket": out_buckets,
        "generator":        out_generators,
        # Provenance: which real split the senders/style examples came from.
        # build_ood_eval.py uses this to keep eval-only (test/validation) synthetics
        # out of training and to pick the right real split for impersonation pairs.
        "source_split":     [args.from_split] * len(out_texts),
    }).save_to_disk(str(out_path))
    print(f"Saved to {out_path}")

    if args.wandb:
        stats = {
            "total_generated": total_generated,
            "total_accepted": total_accepted,
            "overall_accept_rate": (total_accepted / total_generated) if total_generated else 0.0,
            "n_senders": len(senders),
            "n_per_sender": args.n_per_sender,
            "llm_positives": args.llm_positives,
            "variant": variant,
            "cross_register_fraction": cr_frac,
            "cross_length_fraction": cl_frac,
            "dataset_rows": len(out_texts),
            "generators": ",".join(gen_specs),
            "n_generators": n_gen,
            "from_split": args.from_split,
            "eval_only": args.from_split != "train",
        }
        for name, cnt in gen_by_name.items():
            stats[f"generator_rows/{name}"] = cnt
        for m in _MODES:
            stats[f"{m}_generated"] = gen_by_mode[m]
            stats[f"{m}_accepted"] = acc_by_mode[m]
            stats[f"{m}_accept_rate"] = _rate(m)
            stats[f"n_{m}_rows"] = acc_by_mode[m]
        for b in _LENGTH_BUCKETS:
            stats[f"bucket_{b}_generated"] = gen_by_bucket[b]
            stats[f"bucket_{b}_accepted"] = acc_by_bucket[b]
        _log_wandb(cfg, args, out_path, stats, modes=_MODES)


def _log_wandb(cfg, args, out_path, stats: dict, modes: tuple[str, ...]) -> None:
    """Log generation stats as one W&B run. Sectioning (group/job_type/name)
    comes from WANDB_* env vars — we don't pass those kwargs so wandb reads them
    natively. A 'syn-vN' tag is derived from the output dir name so the run is
    filterable alongside the matching train/probe/ablate runs."""
    import re
    import wandb

    stem = out_path.name
    m = re.search(r"v(\d+)", stem)
    ver_tag = f"syn-v{m.group(1)}" if m else stem
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        tags=[*cfg.wandb.tags, "synthetic-generation", ver_tag,
              f"llm-pos-{args.llm_positives}", f"split-{args.from_split}",
              *(["eval-only"] if args.from_split != "train" else [])],
        notes=f"Synthetic generation → {out_path} "
              f"(variant={stats['variant']}, from_split={args.from_split}, "
              f"generators={stats['generators']})",
        config={
            "output": str(out_path),
            "model": args.model,
            "generators": stats["generators"],
            "from_split": args.from_split,
            "eval_only": stats["eval_only"],
            "n_per_sender": args.n_per_sender,
            "llm_positives": args.llm_positives,
            "variant": stats["variant"],
            "cross_register_fraction": stats["cross_register_fraction"],
            "cross_length_fraction": stats["cross_length_fraction"],
            "min_words": args.min_words,
            "max_overlap": args.max_overlap,
            "seed": args.seed,
        },
    )
    wandb.summary.update(stats)
    table = wandb.Table(
        columns=["mode", "generated", "accepted", "accept_rate"],
        data=[
            [m, stats[f"{m}_generated"], stats[f"{m}_accepted"], stats[f"{m}_accept_rate"]]
            for m in modes
        ] + [["overall", stats["total_generated"], stats["total_accepted"], stats["overall_accept_rate"]]],
    )
    wandb.log({"acceptance": table})
    print(f"[wandb] logged generation stats to run {run.name} (tags: {ver_tag}, synthetic-generation)")
    run.finish()


if __name__ == "__main__":
    main()
