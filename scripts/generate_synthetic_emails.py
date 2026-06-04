"""Generate synthetic emails for contrastive training.

Two generation modes are interleaved per sender:

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

Splits and defaults
    --cross-register-fraction 0.4  → 40 % of n-per-sender are cross-register
                                      positives; 60 % are hard negatives.

Quality gates (applied after preprocessing)
    --min-words 15          reject near-empty outputs
    --max-overlap 0.40      reject outputs that are > 40 % Jaccard-overlap
                            with any style-context example (catches near-copies)

Usage:
    python scripts/generate_synthetic_emails.py \\
        --config configs/base.yaml \\
        --model mistralai/Mistral-7B-Instruct-v0.3 \\
        --n-per-sender 15 \\
        --n-examples 5 \\
        --cross-register-fraction 0.4 \\
        --output data/synthetic/enron_synthetic \\
        --load-in-4bit

Model recommendations:
    mistralai/Mistral-7B-Instruct-v0.3  -- strong style following, ~7 GB VRAM (4-bit)
    meta-llama/Llama-3.1-8B-Instruct    -- close second, same VRAM

Output Arrow dataset columns:
    text                  preprocessed generated email body
    sender_id             real sid (cross_register) or sid__syn (hard_neg)
    source_sender_id      always the real sid
    generation_mode       "hard_neg" | "cross_register"
    topic                 topic string passed to the LLM
    context_register      register of style-context examples: "formal" | "casual" | "mixed"

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

import torch
from datasets import Dataset, load_from_disk

# torch.nn.Module.set_submodule was added in PyTorch 2.5; patch it for older builds
if not hasattr(torch.nn.Module, "set_submodule"):
    def _set_submodule(self, target: str, module: "torch.nn.Module") -> None:
        atoms = target.split(".")
        parent = self.get_submodule(".".join(atoms[:-1])) if len(atoms) > 1 else self
        setattr(parent, atoms[-1], module)
    torch.nn.Module.set_submodule = _set_submodule

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from email_fraud.config import load_config
from email_fraud.data.preprocessing import preprocess

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
# Register detection — lightweight heuristic classifier.
# Returns "formal", "casual", or "terse".
# Used to select which style-context emails to show for cross-register prompts.
# ---------------------------------------------------------------------------

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


def _detect_register(text: str) -> str:
    """Classify a single email as 'formal', 'casual', or 'terse'."""
    words = text.lower().split()
    if len(words) < 20:
        return "terse"
    word_set = set(words)
    formal_hits = len(word_set & _FORMAL_SIGNALS)
    casual_hits = len(word_set & _CASUAL_SIGNALS)
    if casual_hits > formal_hits:
        return "casual"
    return "formal"


def _partition_by_register(
    texts: list[str],
) -> dict[str, list[str]]:
    """Return a dict mapping register → list of texts."""
    buckets: dict[str, list[str]] = {"formal": [], "casual": [], "terse": []}
    for t in texts:
        buckets[_detect_register(t)].append(t)
    return buckets


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_hard_neg_prompt(examples: list[str], topic: str) -> str:
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
        "Do NOT copy any sentences from the examples above.\n"
        "Output only the email body. No subject line, no metadata."
    )


def _build_cross_register_prompt(
    examples: list[str],
    context_register: str,
    topic: str,
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
) -> bool:
    """Return True if the generated text passes basic quality gates.

    Rejects:
      - Outputs shorter than min_words words.
      - Outputs that are close paraphrases / near-copies of any style-context
        example (Jaccard overlap > max_overlap on word sets).
    """
    words = text.split()
    if len(words) < min_words:
        return False

    if max_overlap < 1.0:
        text_words = set(w.lower() for w in words)
        for ex in examples:
            ex_words = set(w.lower() for w in ex.split())
            if _jaccard(text_words, ex_words) > max_overlap:
                return False

    return True


# ---------------------------------------------------------------------------
# Model loading and batched generation
# ---------------------------------------------------------------------------

def _load_model(model_name: str, load_in_4bit: bool):
    quant_cfg = None
    if load_in_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_cfg,
        device_map="auto",
        torch_dtype=torch.float16 if not load_in_4bit else None,
    )
    model.eval()
    return tokenizer, model


def _generate_batch(
    prompts: list[str],
    tokenizer,
    model,
    max_new_tokens: int = 300,
    temperature: float = 0.85,
    top_p: float = 0.9,
) -> list[str]:
    """Run a list of prompts through the model and return generated texts."""
    messages_batch = [[{"role": "user", "content": p}] for p in prompts]
    formatted = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in messages_batch
    ]

    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    results = []
    prompt_len = inputs["input_ids"].shape[1]
    for output_ids in outputs:
        generated_ids = output_ids[prompt_len:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        results.append(text)
    return results


# ---------------------------------------------------------------------------
# Per-sender job planning
# ---------------------------------------------------------------------------

def _plan_sender_jobs(
    sid: str,
    texts: list[str],
    n_per_sender: int,
    cross_register_fraction: float,
    n_examples: int,
    rng: random.Random,
) -> list[dict]:
    """Return a list of generation job descriptors for one sender.

    Each job is a dict with keys:
        prompt          str    full LLM prompt
        sender_id       str    where to store the result (real sid or sid__syn)
        topic           str
        context_register str
        mode            str    "hard_neg" | "cross_register"
        examples        list[str]   the style-context texts used (for quality filter)
    """
    n_cross = round(cross_register_fraction * n_per_sender)
    n_hard = n_per_sender - n_cross

    by_register = _partition_by_register(texts)
    formal_pool = by_register["formal"]
    casual_pool = by_register["casual"]

    jobs: list[dict] = []

    # --- Hard-negative jobs ---
    for _ in range(n_hard):
        n_ex = min(n_examples, len(texts))
        ex = rng.sample(texts, n_ex)
        ex_trunc = [e[:600] for e in ex]
        topic = rng.choice(_BUSINESS_TOPICS)
        jobs.append({
            "prompt": _build_hard_neg_prompt(ex_trunc, topic),
            "sender_id": f"{sid}__syn",
            "topic": topic,
            "context_register": "mixed",
            "mode": "hard_neg",
            "examples": ex_trunc,
        })

    # --- Cross-register jobs ---
    for _ in range(n_cross):
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
        jobs.append({
            "prompt": _build_cross_register_prompt(ex_trunc, context_reg, topic),
            "sender_id": sid,          # real ID → treated as positive in contrastive loss
            "topic": topic,
            "context_register": context_reg,
            "mode": "cross_register",
            "examples": ex_trunc,
        })

    return jobs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic emails (hard negatives + cross-register positives)"
    )
    parser.add_argument("--config",     required=True)
    parser.add_argument(
        "--model",
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="HuggingFace model ID (default: Mistral-7B-Instruct-v0.3)",
    )
    parser.add_argument("--n-per-sender",  type=int,   default=15,
                        help="Total synthetic emails to generate per sender (default: 15)")
    parser.add_argument("--n-examples",    type=int,   default=5,
                        help="Real emails to use as style context per generation (default: 5)")
    parser.add_argument("--cross-register-fraction", type=float, default=0.4,
                        help="Fraction of n-per-sender to generate as cross-register positives "
                             "(stored under real sender_id). Default: 0.4")
    parser.add_argument("--min-words",     type=int,   default=15,
                        help="Minimum word count to accept a generated email (default: 15)")
    parser.add_argument("--max-overlap",   type=float, default=0.40,
                        help="Maximum Jaccard overlap with any style-context example (default: 0.40)")
    parser.add_argument("--output",        required=True)
    parser.add_argument("--load-in-4bit",  action="store_true")
    parser.add_argument("--batch-size",    type=int,   default=4)
    parser.add_argument("--max-senders",   type=int,   default=None)
    parser.add_argument("--seed",          type=int,   default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cfg = load_config(args.config)

    print(f"Loading training split from {cfg.data.processed_dir}")
    ds_dict = load_from_disk(cfg.data.processed_dir)
    train_ds = ds_dict["train"]

    sender_to_texts: dict[str, list[str]] = defaultdict(list)
    for text, sid in zip(train_ds["text"], train_ds["sender_id"]):
        sender_to_texts[sid].append(text)

    senders = sorted(sender_to_texts.keys())
    if args.max_senders:
        senders = senders[: args.max_senders]

    n_cross_target = round(args.cross_register_fraction * args.n_per_sender)
    n_hard_target  = args.n_per_sender - n_cross_target
    print(
        f"Found {len(senders)} senders.  "
        f"Per sender: {n_hard_target} hard-neg + {n_cross_target} cross-register "
        f"= {args.n_per_sender} total.  "
        f"Loading model {args.model}..."
    )
    tokenizer, model = _load_model(args.model, args.load_in_4bit)

    # Output accumulators
    out_texts:            list[str] = []
    out_sender_ids:       list[str] = []
    out_source_sender_ids: list[str] = []
    out_modes:            list[str] = []
    out_topics:           list[str] = []
    out_context_regs:     list[str] = []

    preprocess_cfg = cfg.data.preprocessing

    total_generated  = 0
    total_accepted   = 0
    cross_generated  = 0
    cross_accepted   = 0

    total_to_generate = len(senders) * args.n_per_sender
    gen_start = time.monotonic()

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
            cross_register_fraction=args.cross_register_fraction,
            n_examples=args.n_examples,
            rng=rng,
        )
        prompts = [j["prompt"] for j in jobs]

        # Run in batches.
        raw_outputs: list[str] = []
        batch_bar = tqdm(
            range(0, len(prompts), args.batch_size),
            desc=f"  {sid[:30]}",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )
        for batch_start in batch_bar:
            batch_prompts = prompts[batch_start : batch_start + args.batch_size]
            t0 = time.monotonic()
            raw_outputs.extend(_generate_batch(batch_prompts, tokenizer, model))
            elapsed_batch = time.monotonic() - t0
            rate = len(batch_prompts) / elapsed_batch if elapsed_batch > 0 else 0.0
            overall_bar.update(len(batch_prompts))
            batch_bar.set_postfix(generated=len(raw_outputs), rate=f"{rate:.2f}e/s")

        # Filter and store accepted outputs.
        sid_accepted = 0
        sid_cross_accepted = 0
        for raw, job in zip(raw_outputs, jobs):
            total_generated += 1
            if job["mode"] == "cross_register":
                cross_generated += 1

            cleaned = preprocess(raw, preprocess_cfg)
            if cleaned is None:
                continue
            if not _quality_ok(cleaned, job["examples"], args.min_words, args.max_overlap):
                continue

            out_texts.append(cleaned)
            out_sender_ids.append(job["sender_id"])
            out_source_sender_ids.append(sid)
            out_modes.append(job["mode"])
            out_topics.append(job["topic"])
            out_context_regs.append(job["context_register"])

            total_accepted += 1
            sid_accepted += 1
            if job["mode"] == "cross_register":
                cross_accepted += 1
                sid_cross_accepted += 1

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
            f"(cross_register: {sid_cross_accepted}/{n_cross_target})"
        )

    overall_bar.close()
    sender_bar.close()

    cross_rate = cross_accepted / cross_generated if cross_generated else 0.0
    hard_gen   = total_generated - cross_generated
    hard_acc   = total_accepted - cross_accepted
    hard_rate  = hard_acc / hard_gen if hard_gen else 0.0
    print(
        f"\nDone. {total_accepted}/{total_generated} accepted overall."
        f"\n  hard_neg:       {hard_acc}/{hard_gen} ({hard_rate:.0%})"
        f"\n  cross_register: {cross_accepted}/{cross_generated} ({cross_rate:.0%})"
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_dict({
        "text":             out_texts,
        "sender_id":        out_sender_ids,
        "source_sender_id": out_source_sender_ids,
        "generation_mode":  out_modes,
        "topic":            out_topics,
        "context_register": out_context_regs,
    }).save_to_disk(str(out_path))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
