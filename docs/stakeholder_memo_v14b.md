# Catching Email Impersonation with Writing-Style Analysis
### A Stakeholder Briefing on the Email-Fraud Detection System (model version v14b)

*Prepared 2026-06-18. This memo is written for a non-technical audience. It explains
what the project is trying to do, how the system works, the path we took to make it
work, how we measure it, and what the final results mean. Every number cited comes from
our committed evaluation records; the charts referenced live in `docs/figures/`.*

---

## 1. Executive summary

We built a system that answers one question: **"Was this email really written by the
person it claims to be from?"**

This is the core of *business email compromise* (BEC) fraud — an attacker sends a message
that looks legitimate (right name, right address, plausible content) but was actually
written by someone else, or by an AI pretending to be that person. The email address and
the words can be faked. What is much harder to fake is a person's **writing style** — their
characteristic rhythm, word choices, punctuation habits, and sentence structure. Our system
learns each person's style from their past emails and flags new messages that don't match.

The hardest version of this problem, and the one that consumed most of the project, is
catching **AI-generated forgeries from AI models we have never seen before**. An attacker
won't tell us which AI they used. A system that only catches forgeries from the one AI it
was trained on is worthless in practice.

**The headline result:** our latest model (v14b) catches **88% of AI forgeries** while
raising a false alarm on only **5% of genuine emails** — *and it does this against Claude
and Gemini, two AI models that were deliberately kept out of training entirely.* The
previous best model (v12) caught only 39% under the same conditions. The overall accuracy
score (AUC, explained below) rose from 0.85 to **0.975**, which is squarely in
production-grade territory.

| Measure (on AI models never seen in training) | Earlier model (v12) | **Current model (v14b)** |
|---|---|---|
| Forgeries caught at a 5% false-alarm budget | 39% | **88%** |
| Overall accuracy (AUC) | 0.854 | **0.975** |
| Forgeries from Gemini detected (AUC) | 0.746 | **0.953** |
| Forgeries from Claude detected (AUC) | 0.829 | **0.972** |

---

## 2. The objective

When a company deploys our system, it works in two phases:

1. **Enrollment.** We feed the system a batch of each employee's known-genuine historical
   emails. From these, the system builds a compact "style fingerprint" for that person — a
   mathematical summary of how they write.

2. **Scoring.** When a new email arrives claiming to be from that person, the system
   compares the new message against the stored fingerprint and returns a score: high if the
   writing style matches, low if it doesn't.

Two important properties:

- **The system never retrains for a new customer.** It learned a *universal* sense of "what
  makes writing styles distinct" once, during development. At deployment it only *enrolls*
  new people — a fast, one-time operation per email. This is what makes it deployable.
- **It must catch two kinds of impostor.** A *different human* writing as someone else, and
  an *AI* generating a forgery. The AI case is by far the harder of the two, because modern
  AI can mimic content and tone convincingly.

---

## 3. The model we built, and why

### 3.1 The encoder (the "style reader")

At the heart of the system is a component called an **encoder** — specifically a model
called **LUAR**, fine-tuned for our task. An encoder reads a piece of text and turns it
into a list of numbers (a "vector") that captures its essential character.

The crucial design choice was *which kind* of encoder to use. Most language models (like the
RoBERTa family) are trained to understand **topic and meaning** — they place two emails about
"the quarterly budget" close together regardless of who wrote them. That is exactly the wrong
behavior for us: an impersonator writing about the budget would look genuine.

LUAR is different. It was pre-trained specifically to **distinguish authors** — to place two
emails *by the same person* close together even on different topics, and two emails *by
different people* far apart even on the same topic. It reads **style, not subject**. In our
own head-to-head tests, a topic-based encoder scored near random (≈0.53) on our task, while
LUAR reached ≈0.95. This is the foundational reason the system works at all.

### 3.2 The fingerprint and the score

For each enrolled person we store three things:

- a **centroid** — the average position of their emails in style-space (the center of their
  "style cloud");
- a **spread** — how tightly their emails cluster around that center (some people write very
  consistently; others vary a lot);
- **k** — how many emails we have seen from them, which tells us how much to trust the
  fingerprint.

When a new email arrives, we measure how far it lands from the centroid, **expressed in units
of that person's own spread**. A message that lands within the person's normal range scores
high; one that lands far outside it scores low. We deliberately measure distance relative to
each person's *own* variability — a naturally inconsistent writer isn't penalized for being
inconsistent.

We call the final output the **style score** — a number from 0 to 1 that summarizes how
consistent the new email is with the claimed sender's established style. (More on turning
that score into a decision, and the performance we can guarantee on it, in Section 6.)

### 3.3 Lightweight fine-tuning (LoRA)

We did not retrain LUAR from scratch — that would be enormously expensive and risky. Instead
we used **LoRA**, a technique that freezes the original model and trains only a small set of
"adapter" weights on top. This makes each experiment fast and cheap to run, and lets us
iterate through many model versions (v11 through v14b) without rebuilding the foundation each
time.

---

## 4. The training pipeline — how we got the model to perform

### 4.1 The teaching method: learning by contrast

We train the model with a technique called **contrastive learning**. The idea is intuitive:
we show the model many examples and teach it to *pull together* things that belong together
(emails by the same person) and *push apart* things that don't (emails by different people).
Over many rounds, the model sharpens its internal notion of what makes each author distinct.

### 4.2 The secret ingredient: synthetic "hard negatives"

The single most important trick is what we feed the model as *negative examples*. Random
different-person emails are easy to tell apart — that teaches the model very little about the
hard case. So we **generate AI forgeries on purpose**: we take a real person and ask an AI to
write an email *imitating that person's style*, then we put the real email and the AI
imitation side by side and tell the model **"these two are different authors — learn to tell
them apart."**

These deliberately-difficult fakes are called **synthetic hard negatives**. They force the
model to find the subtle stylistic fingerprints that an AI imitation *fails* to reproduce.
As we'll see, these synthetics turned out to be absolutely load-bearing for the whole system.

### 4.3 The honesty rule (this is the most important part of the whole project)

It is easy to build a model that looks great in testing and fails in the real world. Our
guard against this is a strict **honesty rule**:

> We always test against AI models that were **never used in training.** During development
> we trained the model's forgeries using GPT and Llama, and we **completely held out Claude
> and Gemini** — the model never saw a single Claude or Gemini forgery. Then we measured the
> model *specifically* on Claude and Gemini forgeries.

This simulates the real world, where an attacker uses whatever AI they like. A model that
scores well on its training AI but collapses on an unseen one is *not* production-grade, and
this rule is what repeatedly exposed that failure mode and forced us to fix it for real.

---

## 5. The journey: what we tried, and what happened

The project moved through five model versions. Each one was a deliberate experiment that
taught us something. This is the story of how we got from "barely works" to "production-grade."

### v11 — the baseline that exposed the problem
The first version trained its AI forgeries using a **single** AI model (the Mistral family).
On its own training AI it looked fine. But under the honesty rule — tested on the unseen
Claude and Gemini — it was **barely better than a coin flip** (AUC 0.64 and 0.61; only 14% of
forgeries caught). **Diagnosis:** the model had learned to spot the quirks of *one specific
AI*, not a general sense of "this is an AI forgery." A shortcut, not real understanding. This
became the central problem the rest of the project chased.

### v12 — train on more than one AI (the first real fix)
**Hypothesis:** if we train on a *variety* of AIs, the model will generalize to AIs it hasn't
seen. We trained forgeries using **two** AIs (GPT and Llama) and kept Claude and Gemini held
out. **Result:** a large jump — forgeries caught rose from 14% to **39%**, overall AUC from
0.68 to 0.85. The hypothesis was confirmed: diversity in training transfers to the unseen.

### v13 — add yet another AI (the plateau)
**Hypothesis:** keep adding AIs, keep improving. We added a third AI (DeepSeek) and more data
volume. **Result:** essentially **no improvement** (Gemini stuck at ~0.75). Adding more AIs
had stopped paying off. **The key insight:** the bottleneck was *not* the variety of attacker
AIs. The model only knew **44 different people's** writing styles — too few to have a sharp,
general notion of "a person's style." It couldn't reliably tell a good imitation from the real
person no matter how many AI forgeries we showed it. This pivoted our whole strategy: stop
adding attackers, **improve the underlying author model itself.** (We also tested two cheap
shortcuts — an off-the-shelf "is this AI-written?" detector and a pre-built style model — and
*both failed*, confirming we had to do the harder work of retraining.)

### v14 — teach it many more people (one breakthrough, two regressions)
**Change:** we expanded the training data from 44 people to **844 people** by adding a large
corpus of blog authors, forcing the model to learn style across a huge variety of writers and
topics. To isolate the effect cleanly, this version **dropped the synthetic forgeries**.
**Result — a revealing split:**
- ✅ **Breakthrough:** the model became dramatically better at separating *style from topic*
  (our cleanest independent test jumped from 0.78 to 0.88 — the first real movement on this
  stuck problem in the entire project).
- ❌ **But forgery-catching collapsed** (because we removed the synthetics) — proving the
  synthetics are essential.
- ❌ **And email-specific performance dipped** (44 email writers got diluted among 800 bloggers).

This "failure" was actually the most informative result of the project: it **cleanly proved
that we need two separate ingredients** — *many identities* for general style understanding,
and *synthetic forgeries* for catching imitations. The production model needs **both at once.**

### v14b — the synthesis (our production model)
The reason v14 had to drop the synthetics was a **technical bottleneck in the data pipeline**:
the old data sampler sized each training round around the tiny pool of synthetic pairs,
which starved the 800 new authors. We fixed the sampler (`SyntheticBalancedSampler`) so it
properly sizes training to the large author pool *while still guaranteeing synthetic forgeries
in every batch.* We verified the fix is a no-op for the older versions, so it changed nothing
retroactively — it only unlocked the new combination.

**v14b combines everything: 844 people + synthetic forgeries.** And it beats every prior
version on every axis simultaneously:

| Capability | v12 | **v14b** |
|---|---|---|
| Catching Gemini forgeries (AUC) | 0.746 | **0.953** |
| Catching Claude forgeries (AUC) | 0.829 | **0.972** |
| Overall pool accuracy (AUC) | 0.854 | **0.975** |
| Forgeries caught at 5% false-alarm | 39% | **88%** |
| Separating style from topic | 0.779 | **0.857** |

**Why it works:** roughly five times more exposure to synthetic forgeries per training round,
*plus* a much sharper author model built from 844 people. Together, AI imitations now fall
*much further* outside a well-modeled person's style — no matter which AI wrote them. This
validated the core thesis of the whole project: **better authorship modeling is what catches
imitations.**

---

## 6. From a score to a service-level agreement (SLA)

### 6.1 Why a raw score is not yet a promise

The system outputs a **style score** between 0 and 1 — high when an email matches the claimed
sender's style, low when it doesn't (Section 3.2). But a raw score is not something we can put
in a contract. To make a *decision* (allow / flag) you need a **threshold**: a cutoff below
which an email is flagged as suspicious. And to make a *promise*, you need to express the
system's behavior at that threshold in terms a customer can hold us to.

A **service-level agreement (SLA)** is exactly that: the set of measurable performance
guarantees we commit to in production. For a fraud-detection system, an SLA has two halves —
**operational** guarantees (is the service up and fast?) and, more importantly here,
**detection-quality** guarantees (how well does it actually catch fraud, and how often does it
cry wolf?).

### 6.2 The detection-quality SLA: the false-alarm budget

The core of our quality SLA is the trade-off between catching fraud and raising false alarms.
We do not pick an arbitrary cutoff; we commit to a **false-alarm budget** and report what
detection rate we can guarantee within it. The contractual sentence reads:

> *"At a threshold that wrongly flags no more than 5% of genuine emails, the system catches at
> least 88% of AI forgeries — including forgeries from AI models it has never seen."*

The technical name for this number is **TPR at a fixed FPR** (true-positive rate at a fixed
false-positive rate). It is the single most SLA-relevant figure we produce, because a customer
can map it straight onto their own operations: a 5% false-alarm budget tells their security
team how much manual review they're signing up for, and the 88% catch rate tells them how much
fraud they're stopping in return.

### 6.3 Typical SLA markers for a production-grade detection system

Production fraud-detection SLAs are usually written against markers like the following. The
first three are the *detection-quality* terms that matter most for us; the last two are the
standard *operational* terms any production service is expected to carry.

| SLA term | Below par | Acceptable | Production-grade | **v14b delivers** |
|---|---|---|---|---|
| Overall detection accuracy (AUC) | < 0.85 | 0.90 – 0.95 | **> 0.95** | **0.975** ✅ |
| Fraud caught at ≤5% false alarms | < 50% | 60 – 80% | **> 85%** | **88%** ✅ |
| False-alarm rate on genuine email | > 10% | ~5% | **≤ 5%** | **5%** ✅ |
| Wrong-human "leak" guardrail | > 15% | ~10% | **≤ 10%** | **8.3%** ✅ |
| Latency / availability (operational) | — | — | e.g. <200ms, 99.9% uptime | scoring is one model pass + a lookup; well within reach |

v14b clears every detection-quality marker. On the operational side, scoring an email is a
single fast model pass plus a fingerprint lookup (no retraining at request time), so standard
latency and uptime SLAs are comfortably achievable — those will be finalized once the service
is wrapped for production traffic.

### 6.4 The guardrail term: don't forget the human impostors

There is a subtle trap. Because v14b became *so* good at catching AI forgeries, if we tune the
threshold purely on the (now-easy) AI forgeries, it can drift to a point where it lets *human*
impostors slip through. So we enforce a second, independent **guardrail**: the rate at which
the system wrongly accepts a *different human* (`FPR_other`) must stay **at or below 10%**.

We tested two ways of computing the score at the decision point:
- A simpler "linear" method caught 94% of AI forgeries **but leaked 30% of human impostors** —
  it violates the guardrail and **must not be shipped.**
- The **Mahalanobis** method (a smarter distance measure that accounts for the shape of each
  person's style cloud) catches **88%** of AI forgeries **and** holds the human-impostor leak
  to **8.3%** — inside the 10% guardrail.

**Deployment recommendation:** ship the v14b model with the **Mahalanobis** scorer, and set
the threshold on the *worst case* of both the AI-forgery and human-impostor tests — never on
the AI forgeries alone.

---

## 7. How we run evaluations (so the numbers are trustworthy)

Our evaluation deliberately mirrors deployment rather than the training setup. We build a
fixed test scenario called the **centroid probe**:

1. **Profile 44 senders.** Each is enrolled with a handful of their real emails, exactly as a
   real customer would be. This builds their style fingerprints.
2. **Throw three kinds of new email at each profile and score them all:**
   - **Genuine** (264 emails) — *real* messages from those same people, held out from
     enrollment. These **should score high.**
   - **Wrong-human impostors** (600 emails) — real emails from *completely different* people.
     These **should score low** (the easy case).
   - **AI forgeries** (327 emails) — AI-generated imitations of the profiled senders, from
     **Claude and Gemini, which were never in training.** These **should score low** (the hard
     case, and the whole point).
3. **Compute the metrics** below from how those three groups sort by score.

Because the genuine emails are held out from enrollment and the forgeries come from unseen
AIs, this is an honest dress rehearsal for the real world — not a memorization test.

### The metrics we collect, in plain terms

- **AUC (Area Under the Curve)** — the probability that a randomly chosen genuine email scores
  higher than a randomly chosen forgery. 0.5 is a coin flip; 1.0 is perfect. Our headline
  AUC is **0.975**. It is *threshold-free* — a pure measure of how well the score ranks
  genuine above fake.
- **TPR @ FPR (forgeries caught at a false-alarm budget)** — the operational number from
  Section 6.2. **88% at a 5% budget.**
- **FPR_other (the human-impostor guardrail)** — how often a different human is wrongly
  accepted. **8.3%**, inside our 10% limit.
- **Partial AUC** — like AUC but focused only on the low-false-alarm region we'd actually
  deploy in, where small improvements matter most.
- **Content-invariance (PAN cross-topic)** — a clean, fully independent test of whether the
  model is reading *style* rather than *topic*. **0.857.**

We also separate the *ranking* metrics (AUC, partial AUC), which only care about ordering,
from the *threshold* metrics (forgeries caught, false-alarm rates), which depend on where the
cutoff is placed. Improving one does not automatically improve the other, and we track both.

---

## 8. Reading the v14b charts (`docs/figures/`)

Five charts were produced to tell this story. Each is generated directly from the evaluation
records, so they are always faithful to the underlying numbers.

**Figure 1 — `fig1_cross_generator_auc.png` — "The headline."**
Side-by-side bars showing how well each model version catches forgeries from **Claude** and
**Gemini** (both held out of training). You can watch the bars climb from near-the-coin-flip
line at v11 up to ~0.95+ at v14b. This is the clearest single picture of the project's payoff:
the system learned to catch forgeries from AIs it had never seen.

**Figure 2 — `fig2_pool_progression.png` — "Overall progress."**
Two bars per version: overall accuracy (AUC) and forgeries-caught-at-5%-false-alarms, across
all five versions. Shows the steady climb, the v13 plateau, and the v14b jump to production
levels.

**Figure 3 — `fig3_split_of_effects.png` — "Why we needed the synthesis."**
This is the most important *explanatory* chart. It compares v12, v14, and v14b across three
different abilities: catching imitations, separating style from topic, and email-specific
performance. It visually proves the central lesson: **v14 won one ability but lost another;
only v14b wins all three at once.** It justifies why the final design combines both
ingredients.

**Figure 4 — `fig4_confusion_v12_vs_v14b.png` — "Before and after, in practice."**
Two "confusion matrices" — grids showing what actually happens to genuine emails, AI
forgeries, and human impostors at the real deployment threshold. Green cells are correct
decisions, red are errors. The story in one image: at the same false-alarm rate, **v12 let
61% of forgeries through; v14b lets only 12% through** — while still correctly allowing 95% of
genuine email.

**Figure 5 — `fig5_guardrail_scorer_choice.png` — "Why we pick the Mahalanobis scorer."**
Compares the two scoring methods on two bars each: forgeries caught (higher is better) and
human impostors wrongly accepted (lower is better), with the 10% guardrail line drawn in. It
shows visually that the simpler scorer breaches the guardrail (leaking 30% of human
impostors) while Mahalanobis stays under it — the basis for our deployment recommendation.

---

## 9. Honest caveats

We hold ourselves to the honesty rule in our reporting too:

- **One of our datasets is now partly "in-domain."** The blog corpus we trained on also
  appears in one of our style tests, so that particular score isn't fully independent. The
  *clean*, independent style test (PAN cross-topic, **0.857**) is the one we stand behind;
  the blog result corroborates it but doesn't count as independent evidence.
- **Unseen AIs transfer, but a brand-new future AI should still be spot-checked.** Claude and
  Gemini transferred well from GPT+Llama training, which is strong evidence of generalization
  — but before trusting the system blindly against a genuinely novel future AI vendor, we
  should run a quick verification.
- **The threshold must be set on the worst case** of the AI-forgery and human-impostor tests,
  not on the AI forgeries alone — as Section 6.4 explains.

---

## 10. Recommendation and next steps

**Recommendation:** promote **v14b with the Mahalanobis scorer** to production, with the
decision threshold anchored on the worst case of the AI-forgery and human-impostor tests. By
every production-grade marker — overall accuracy (0.975), forgeries caught (88%), false-alarm
rate (5%), and the human-impostor guardrail (8.3%) — it qualifies.

**Sensible next steps**, all incremental and low-risk:
- **Per-sender threshold calibration** — tune the cutoff to each person's own style
  variability rather than using one global cutoff, the biggest remaining lever on real-world
  precision.
- **A spot-check protocol for new AI vendors** — a small, fast test to run whenever a new
  attacker AI appears.
- **Production monitoring** — watch the score distribution over time so we notice if the
  threshold ever needs re-anchoring.

All training runs are archived for review under the team's experiment-tracking workspace
(`klconvergence/email-fraud-detection`).
