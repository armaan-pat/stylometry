# The Email Impersonation Detector, Explained From Scratch

*A companion to the whitepaper (`docs/whitepaper/whitepaper.pdf`) for readers with little or no
machine-learning background. Everything here describes the same system and the same results —
just with the "why" spelled out at every step. No prior knowledge is assumed except knowing what
an email is.*

---

## Table of contents

1. [The problem we're solving](#1-the-problem-were-solving)
2. [The core idea: writing style as a fingerprint](#2-the-core-idea-writing-style-as-a-fingerprint)
3. [Crash course: what a model and an embedding actually are](#3-crash-course-what-a-model-and-an-embedding-actually-are)
4. [The architecture, piece by piece](#4-the-architecture-piece-by-piece)
5. [How it runs day to day: enrollment and scoring](#5-how-it-runs-day-to-day-enrollment-and-scoring)
6. [How we measure success: metrics from zero](#6-how-we-measure-success-metrics-from-zero)
7. [The story of getting it to work (v11 → v14b)](#7-the-story-of-getting-it-to-work-v11--v14b)
8. [The honesty rules: how we avoid fooling ourselves](#8-the-honesty-rules-how-we-avoid-fooling-ourselves)
9. [What it still can't do](#9-what-it-still-cant-do)
10. [Glossary](#10-glossary)

---

## 1. The problem we're solving

**Business email compromise (BEC)** is a fraud pattern: you receive an email that appears to come
from your CFO, your vendor, or your colleague, asking you to wire money, change payment details,
or share credentials. The *address* can be spoofed or the account genuinely compromised. The
*content* looks routine. Nothing about the message screams "fraud" — that's the point.

Historically, the weak spot of these attacks was the writing itself. The attacker doesn't know how
your CFO actually writes — her odd greetings, her comma habits, whether she says "pls" or
"please" — so careful readers sometimes noticed something was off.

Large language models (LLMs — systems like ChatGPT, Claude, Gemini) removed that weak spot. If an
attacker has a handful of the victim's real emails — from a leaked inbox, a public mailing list,
or a prior compromise — they can paste those into an LLM and say: *"Write a new email in exactly
this person's style."* The output is fluent, in-character, and cheap to produce at scale. We call
this a **few-shot imitation** ("few-shot" just means the LLM was shown a few examples to copy).

So the question the project answers is:

> **Can a machine tell, from the writing alone, that an email claiming to be from Alice was not
> actually written by Alice — even when the forger is a state-of-the-art LLM that studied Alice's
> real emails?**

The answer turned out to be yes — but only after discovering that the obvious approaches fail, and
why.

---

## 2. The core idea: writing style as a fingerprint

Everyone writes a little differently, in ways that are surprisingly stable and surprisingly hard
to fake completely: sentence length rhythms, punctuation habits, greeting and sign-off choices,
how you hedge ("I think maybe..." vs "It is likely that..."), typo patterns, formatting quirks.
Linguists call this an **idiolect** — your personal dialect. The field that studies it is called
**stylometry**, and it's old: it has been used to attribute anonymous pamphlets and disputed
authorship for over a century, originally by counting word frequencies by hand.

Our system automates and modernizes this:

1. **Enrollment.** For each person in a company, take a set of their known-genuine historical
   emails and distill from them a compact "style fingerprint."
2. **Scoring.** When a new email arrives claiming to be from that person, compare its style to the
   stored fingerprint. If the style doesn't match, raise a flag — *regardless of what the email
   says or which server it came from*.

Notice what this design buys us. A content filter asks "does this email look like fraud?" — but
fraudulent requests are designed to look routine. A machine-text detector asks "does this look
LLM-written?" — but as we'll see in Section 7, a good imitation doesn't look LLM-written. Our
question is different: **"does this look like *Alice*?"** The attacker controls their own tools,
but they don't control what Alice's writing actually looks like. That asymmetry is the entire bet
of the project, and the results bear it out.

Two important design properties, decided up front:

- **The system never learns from the customer's data at training time.** The fingerprint-making
  machinery is built once, using public email corpora. Deploying to a new company means running
  each sender's emails through it — no retraining, no model updates. This matters for privacy,
  for cost, and for predictability (a system that silently retrains on your mail is a system whose
  behavior you can't audit).
- **The system may abstain.** If we have fewer than 5 emails from a sender, we don't have enough
  evidence of their style to make a fair judgment, so the system says "not enough history" rather
  than guessing.

---

## 3. Crash course: what a model and an embedding actually are

You can understand this whole project with two concepts.

### 3.1 A model is a tunable text-to-numbers machine

A neural network "model" is a very large mathematical function. It takes input (here: the text of
an email), passes it through millions of simple arithmetic operations governed by millions of
adjustable knobs (called **parameters** or **weights**), and produces output numbers. "Training" a
model means automatically adjusting those knobs, a tiny bit at a time over many examples, so the
output numbers become *useful* for some task. Nobody sets the knobs by hand; an algorithm nudges
them in whichever direction reduces the model's mistakes on the training examples.

The crucial consequence: **a model becomes good at exactly what its training pushed it toward —
nothing more.** Much of this project's story is about discovering that our training was
accidentally pushing toward the wrong thing, and fixing that.

### 3.2 An embedding is a location on a map

Our model's output for an email is a list of 128 numbers. Think of that list as **coordinates on a
map** — not a map of places, but a *map of writing styles*. This list is called an **embedding**.

The whole game is to train the model so that the map has one property:

> **Emails written by the same person land close together on the map. Emails written by
> different people land far apart.**

If we achieve that, everything else becomes simple geometry:

- Alice's fingerprint = the *center point* of where her known emails land (her **centroid** — just
  the average of her emails' coordinates).
- Scoring a new "email from Alice" = measure how far its coordinates land from Alice's center. Near
  the center → probably really Alice. Far away → someone else's hand wrote it.

"Distance" here is measured with **cosine similarity** — a standard way of comparing two lists of
numbers that asks "do these point in the same direction?" You don't need the formula; just read
every "distance" below as "how different are these two styles, as the model sees it."

Why 128 numbers and not, say, 2? Because style has many independent dimensions — formality,
punctuation habits, vocabulary, sentence rhythm, greeting style, and hundreds of subtler things.
A 2-number map couldn't keep thousands of authors mutually separated; 128 gives the map enough
room. (We can't label what each of the 128 numbers "means" — the model finds its own features —
and we don't need to; we only need the same-person-close, different-person-far property.)

### 3.3 The hard part

Nothing guarantees the map organizes itself by *author*. Left to its own devices, a text model's
map organizes by **topic** — all the emails about invoices cluster together, all the ones about
lunch plans cluster together — because topic is the loudest signal in text. A topic map is useless
for us: a forger writing about invoices would land right next to Alice's real invoice emails.

So the project is really about **forcing the map to organize by authorship instead of topic**, and
then further forcing it to keep LLM imitations of Alice away from real Alice. Sections 4 and 7
describe how.

---

## 4. The architecture, piece by piece

"Architecture" = what the components are and why each one is shaped the way it is.

### 4.1 The encoder: LUAR, a model pretrained to recognize authors

The **encoder** is the text-to-coordinates machine at the heart of everything. We did not build it
from scratch; we started from a published model called **LUAR** (Learning Universal Authorship
Representations), which was pretrained on millions of Reddit comments with one task: *given two
bundles of comments, decide whether the same person wrote both.* In other words, LUAR arrives
already biased toward noticing authorship signals rather than topic.

**Why this choice mattered enormously.** Early in the project we ran a bake-off. We took two famous
general-purpose text models (RoBERTa and MPNet — excellent at topic-related tasks) and LUAR, and
fine-tuned each on our email data identically. The general-purpose models scored **0.53** on our
verification measure — barely above a coin flip (0.50). LUAR scored **0.956**. Fine-tuning could
not rescue the general-purpose models, and we could see why in the diagnostics: their maps placed
same-sender emails close together, *but different-sender emails were just as close* — everything
was bunched by topic, and no useful separation existed. The lesson, in plain terms: **you cannot
easily fine-tune away a model's fundamental worldview.** A model raised to compare authors sees
authors; a model raised to fill in missing words sees topics. Start from the right worldview.

**Episodes: judging a writer by several emails at once.** LUAR has a distinctive design: instead
of embedding one text at a time, it reads a small *bundle* of texts by the same person (an
"episode") and produces one embedding for the bundle. Why is that a good idea? Because style is a
*statistical* property. One short email — "Sounds good, see you at 3" — carries almost no style
information. Eight emails carry a lot: patterns start repeating, and the noise of any single email
averages out. It's the difference between identifying a musician from one note versus from a whole
phrase.

### 4.2 LoRA: fine-tuning without breaking what works

LUAR knows Reddit, not corporate email. We need to adapt it — but *carefully*. Retraining all of a
large model's millions of parameters on our (relatively small) email data risks two failures:
it's expensive, and worse, it can overwrite the broad authorship knowledge from pretraining with
narrow quirks of our small dataset (a failure mode called catastrophic forgetting — like an
experienced doctor cramming so hard for one exam that they forget general medicine).

**LoRA** (Low-Rank Adaptation) is the standard solution: freeze the original model entirely, and
bolt on small, trainable "adjustment dials" alongside it (in our setup, well under 1% of the total
parameters). Training only moves the dials. The original knowledge stays intact underneath, and
the adapter learns just the *delta*: "here is how email style differs from Reddit style; here is
what to attend to for this task." We get adaptation at a fraction of the cost and risk.

### 4.3 The training signal: contrastive learning, or push-and-pull

How do you *tell* a model "organize your map by author"? With a **contrastive** objective. Each
training step works like this:

1. Assemble a batch of emails from several senders — in our runs, 16 senders × 8 emails each.
2. Encode them all into map coordinates.
3. Apply the rule: **pull together** the points that share an author; **push apart** the points
   that don't.
4. Nudge the model's dials in whatever direction accomplishes that a bit better. Repeat tens of
   thousands of times.

The specific version we use (called supervised contrastive loss, applied over LUAR-style episodes)
has a "temperature" setting that controls something important: **how much extra attention the
hardest cases get.** We set it low, which means the model's effort concentrates on the pairs that
are *nearly* confusable — different authors whose points currently sit close together. That's
deliberate, because our nightmare case is exactly a near-confusable pair: an LLM imitation of
Alice sitting right next to real Alice. Which brings us to the most important trick in the system.

### 4.4 Synthetic hard negatives: hiring forgers to train the guard

If you want a guard who catches forgeries, show them forgeries during training. We do this
literally:

- For each training sender, we use external LLMs to generate **imitation emails** — the LLM is
  prompted with several of that sender's real emails and told to write new mail as that person.
  These are exactly the attacks we expect in the wild.
- Each imitation set is registered under a *separate* identity: `alice__syn` ("synthetic Alice")
  is a different label than `alice`.
- The batch-building machinery guarantees that several (real, imitation) pairs appear in *every*
  training batch, side by side.

Because real-Alice and fake-Alice carry different labels, the push-apart rule applies **between**
them. Every single training step, the model is forced to answer: *what, exactly, distinguishes
real Alice from a competent LLM imitation of Alice?* Whatever features let it answer — the residue
of style an LLM can't quite copy — are precisely the features we want it to sharpen. These
imitations are the "hardest negatives" in every batch, and the low temperature (above) makes them
dominate the learning signal.

Two rules about synthetics were learned through painful experience and are now enforced in code:

- **LLM text is never used as a positive example of a sender.** An earlier version tried using
  LLM-generated text as *additional genuine examples* (to teach robustness across writing
  registers). It backfired — it taught the model that LLM-flavored text can be "genuine," which
  eroded exactly the boundary we need. A controlled five-way experiment later confirmed this was
  the cause. Now: synthetic text appears only on the impostor side, ever.
- **You must not evaluate training progress on synthetics alone.** Covered in Section 8 — this one
  is subtle and nearly shipped a broken model.

### 4.5 Data augmentation: random cropping

During training we sometimes chop emails down to random short fragments (5–60 words). Why: real
incoming mail is often two lines long, and we need the model to not fall apart on short texts. It's
the same principle as training a face recognizer on partially cropped photos so it copes with
partial views. (This helped short-email performance; it did *not* fully solve a related problem —
comparing a short text against a long one — which remains our weakest axis; see Section 9.)

---

## 5. How it runs day to day: enrollment and scoring

Training (everything in Section 4) happens once, on our side, on public data. What a deployed
system does is much simpler — and involves **no learning at all**, just encoding and geometry.

### 5.1 Enrollment: building a sender's profile

For each sender in the customer's archive:

1. **Clean each email.** Strip quoted reply chains and signature blocks (they're boilerplate, not
   the sender's live writing), and mask things like URLs, email addresses, and phone numbers.
   Why mask? Two reasons: those tokens are content, not style — and worse, they're *identity
   leaks*. If Alice's signature contains her name, a model could "verify" her by reading the
   signature — trivially copyable by any forger. We force the model to rely on style alone by
   deleting the shortcuts.
2. **Encode each email** into its 128-number map location.
3. **Summarize** into a profile of three things:
   - **centroid** — the average location: "the center of Alice-style territory";
   - **spread** — how widely her own emails scatter around that center: "how consistent a writer
     is Alice?" (some people write nearly identically every time; others vary a lot with context);
   - **k** — how many emails went into the profile: "how much evidence do we have?"

That's the whole fingerprint: a point, a radius, and a count. Adding a new sender or a new email
later is instant — encode, update the average. Nothing retrains.

### 5.2 Scoring: judging one incoming email

When an email arrives claiming to be from Alice: clean it, encode it, and measure how far it lands
from Alice's centroid. But "far" has to be judged *relative to Alice herself* — and this is where
the scoring math earns its keep.

**Why raw distance isn't enough.** Suppose Bob's incoming email is 0.3 map-units from his centroid,
and Alice's incoming email is also 0.3 units from hers. Same distance — same verdict? No. If Bob is
an extremely consistent writer (spread 0.1), landing 0.3 away is *three times* his normal
variation — deeply suspicious. If Alice is a chameleon (spread 0.3), 0.3 away is a perfectly
ordinary Tuesday. So we always divide the distance by the sender's own spread. The result is a
**z-score**: "how many 'personal units of normal variation' away is this email?" z = 1 is
ordinary; z = 3 is way out of character. The baseline scorer in the codebase (`linear_z3`) is just
a repackaging of this z-score into a friendly 0-to-1 scale.

**The upgrade: Mahalanobis distance, or knowing the *shape* of someone's territory.** The z-score
treats a sender's territory as a circle: one number for spread, same in all directions. But real
writers don't vary uniformly. Alice might vary a lot in formality (quick notes vs. board memos)
while being rigidly consistent in punctuation. Her territory is a stretched ellipse, not a circle.
An email that deviates along her *formality* axis is normal-for-Alice; the same amount of deviation
along her *punctuation* axis is alarming. **Mahalanobis distance** is the classical statistical
tool for exactly this: it measures distance in units of the sender's own variation *per direction*,
so "unusual in a way this person is never unusual" scores as farther than "unusual in a way this
person often is."

Estimating that ellipse-shape from only 8–16 enrollment emails is statistically dicey (you're
estimating a lot of shape from a little data), so we apply a standard stabilizer called
**Ledoit–Wolf shrinkage** — intuitively, it blends the measured ellipse with a plain circle, leaning
toward the circle when the data is thin and toward the measured shape as evidence accumulates.
Cautious when ignorant, precise when informed.

**Graceful degradation.** For senders with fewer than 5 enrolled emails, no shape can be estimated
at all — the scorer falls back to the simple circular comparison, and the system reports low
confidence / abstains. The system also assigns confidence tiers by k (how many emails back the
profile), so downstream consumers know how much to trust each verdict.

Why did the scorer choice end up mattering so much? Not for the reason you'd guess (better
rankings) — the real reason is about *thresholds*, and it's one of the project's most interesting
findings. See Sections 6.4 and 7.6.

---

## 6. How we measure success: metrics from zero

This section builds every number in the whitepaper from scratch. It's worth reading slowly — most
of the project's hard-won lessons are about *measurement*, not modeling.

### 6.1 The four outcomes and the two kinds of error

For each incoming email, the system ultimately makes a binary call: allow or flag. Reality is also
binary: genuine or impostor. That gives four outcomes:

| | System allows | System flags |
|---|---|---|
| **Genuinely from claimed sender** | ✅ correct | ❌ **false alarm** |
| **Impostor** | ❌ **miss** | ✅ correct |

The two errors have very different costs, and they trade off against each other:

- A **false alarm** wrongly flags a real email. Each one costs trust, and too many make people
  ignore the system entirely (security teams call this alert fatigue; a fire alarm that cries wolf
  gets unplugged).
- A **miss** lets a forgery through. One miss can cost a wire transfer.

Two standard rates describe a system's behavior: the **false positive rate (FPR)** — what fraction
of genuine emails get wrongly flagged — and the **true positive rate (TPR)** — what fraction of
forgeries get caught.

### 6.2 The threshold: a sensitivity dial

The system doesn't natively output "allow/flag" — it outputs a *score* (how Alice-like is this
email?). Someone must choose a cutoff: flag everything scoring below the line. That cutoff — the
**threshold** — is a sensitivity dial:

- Slide it toward paranoid → catch more forgeries (TPR ↑) but annoy more real senders (FPR ↑).
- Slide it toward permissive → fewer false alarms, more misses.

There is no "correct" threshold in the abstract; it's a business decision about tolerable
false-alarm cost. Our convention throughout the project is to ask: **if we allow a 5% false-alarm
rate (and, secondarily, a strict 1%), what fraction of forgeries do we catch?** Those numbers are
written **TPR@5%** and **TPR@1%**. The final model catches **~91% of forgeries at the 5% budget**
(averaged across evaluation reruns).

Why fixate on low-FPR performance instead of overall accuracy? Because deployment lives there. A
system that's brilliant "on average" but only when allowed a 30% false-alarm rate is unshippable.
The low-FPR region of performance is the *only* region that matters operationally — and, as we'll
see, it's exactly the region where naive approaches quietly fail.

### 6.3 AUC: grading the ranking, before any threshold

One more standard metric appears everywhere in our tables: **AUC** ("area under the ROC curve" —
the name matters less than the meaning). Here's the cleanest way to understand it:

> **Pick one genuine email and one forgery at random. AUC is the probability that the system
> scores the genuine one higher.**

- AUC = 0.5 → coin flip; the scores carry no information.
- AUC = 1.0 → every genuine email outranks every forgery; perfect separation.
- Our final model: **0.980** on the hardest test.

AUC's virtue is that it grades the *entire ranking* at once, independent of any threshold choice —
a clean summary of "does the model see the difference at all?" Its vice is the flip side: **a good
AUC does not guarantee a good threshold exists at your false-alarm budget**, and it says nothing
about where to put the threshold. That's why we always report AUC *and* TPR@5% *and* the guardrail
below. Single-number summaries are where bad decisions hide.

### 6.4 The guardrail: the second kind of impostor

Everything above treats "impostor" as one category. It isn't. There are two, and they behave
completely differently:

- **LLM forgeries** — machine-written imitations. The hard, novel threat.
- **Wrong-human impostors** — mail genuinely written by some *other person*. The classical case
  (colleague using someone's account, human-typed fraud).

Here's the trap: as our model got better and better at spotting LLM forgeries, the LLM forgeries
became the *easy* class — and a threshold calibrated against them ("flag anything scoring worse
than the best 5% of forgeries") drifted into territory that misjudged *humans*. The measured example
from our own system: at the same operating point, the simple scorer catches 94% of forgeries but
**wrongly flags 25–30% of legitimate-but-different-human mail**; the Mahalanobis scorer catches 91%
while keeping that leak at **8%**.

So every result in the project carries a **guardrail metric**: `FPR_other` — the error rate on the
wrong-human axis at the chosen threshold — with a hard requirement that it stay **≤ 10%**. And the
deployment rule that follows: **anchor the threshold on the *worst* of the two impostor axes,
never on the synthetic pool alone.** The better the model gets on synthetics, the more degenerate
a synthetics-only threshold becomes — our novel-vendor test (Section 7.7) demonstrated this
vividly: near-perfect forgery separation (AUC 0.996) made the synthetic-anchored threshold
essentially meaningless, while the underlying rankings stayed excellent.

### 6.5 Error bars: why every number is really two numbers

Our headline is written **0.980 ± 0.003**, not 0.980. The "± 0.003" answers a question you should
always ask of any measured number: *if we measured again, how different would it come out?*

Two sources of randomness could make a single measurement misleading, and we control for both:

- **Which test examples we happened to draw.** Our evaluation randomly samples senders, enrollment
  emails, and impostors. A different random draw gives a slightly different number. So we repeat
  the entire evaluation **five times with different random draws** and report the average ± the
  variation. We also use a statistical resampling technique (the **bootstrap**: re-computing the
  metric thousands of times on reshuffled samples of the same data) to put uncertainty ranges on
  individual measurements. The payoff: when we say "v12 (0.871) beats v11 (0.690)," we can check
  that the gap (0.18) dwarfs the measurement noise (±0.02) — the improvement is real, not luck.
- **The randomness of training itself.** Model training starts from random settings and shuffles
  data randomly; maybe our good model was a lucky roll. So we retrained the final model from
  scratch with different randomness. Result: 0.976 vs 0.980 — the same model quality, within noise.
  The recipe is reproducible, not a fluke. (One honest nuance surfaced: the threshold-level numbers
  like TPR@5% wobble more across retrains (~±0.04) than AUC does, so we quote them as a range,
  0.87–0.91, and recommend calibrating thresholds per deployment.)

Why belabor this? Because the project's history includes a version that looked great under one
evaluation lens and was actually broken (Section 7.1). Error bars and repeated measurement are not
academic niceties here; they're the difference between knowledge and anecdote.

---

## 7. The story of getting it to work (v11 → v14b)

This is the heart of the project, told as what it was: a sequence of experiments, each changing
one thing, several failing informatively. Version numbers are just successive model generations.

| Version | One-line description | Held-out forgery AUC |
|---|---|---|
| v11 | trained against one forger (Mistral) | 0.690 ± 0.020 |
| v12 | trained against two forgers (GPT + Llama) | 0.871 ± 0.016 |
| v13 | added a third forger (DeepSeek) | 0.872 ± 0.021 — plateau |
| v14 | 19× more training authors, synthetics removed | 0.610 ± 0.016 — collapse (but see below) |
| **v14b** | **many authors + synthetics together** | **0.980 ± 0.003** |

("Held-out" will be defined in a moment — it's the crux.)

### 7.1 v11: the trap — training the guard against a single forger

v11 was trained with synthetic imitations from one LLM family (Mistral). Evaluated against Mistral
forgeries, it looked superb: it caught **91%** of them at the strict 1% false-alarm budget. Shipped
confidence, on that evidence, would have been high.

Then we tested it against forgeries from LLMs it had never seen — Claude and Gemini. The catch
rate at the same budget: **4.5%**. Not 91. *Four and a half.*

What happened? The model hadn't learned "what forgery looks like." It had learned **"what Mistral
sounds like"** — the specific statistical quirks of one generator's prose. Like a bank teller
trained only on one counterfeiter's bills, flawless against those and helpless against any other
press. In ML this failure is called **learning a shortcut**: the training data offered an easy
signal (Mistral's accent) that correlated perfectly with the right answer during training and not
at all in the real world, where attackers choose their own tools.

This single experiment reframed the entire project. From then on, the iron rule:

> **All headline evaluation uses forgeries from generators the model never trained on.**
> Training forgers: GPT-4o-mini and Llama-3.1-70B. Evaluation forgers: Claude-3.5-Haiku and
> Gemini-2.5-Flash — never used in training, for any version. ("Held-out" = held out of training.)

Every AUC in the table above is measured under this rule. That's why v11's number there is 0.690 —
that *is* its true performance against a realistic attacker.

### 7.2 Detour: why we couldn't just buy a solution off the shelf

Before spending heavily on retraining, we tested the two obvious shortcuts. Both failed, and both
failures taught us something. (We ran these cheap tests *first*, deliberately — eliminating
dead-ends before committing GPU money to the expensive path.)

**Shortcut 1: general-purpose "was this written by an AI?" detectors.** Published zero-shot
detectors (Fast-DetectGPT, Binoculars — "zero-shot" meaning they need no training on specific
generators) work by measuring how *statistically predictable* text is. LLM text tends to be
smoother and more probable-word-after-probable-word than human text; these detectors measure that
smoothness. On generic AI text, they work reasonably well. On our imitation emails they scored
**AUC 0.49–0.54 — coin-flip territory** — and against Claude's imitations, *below* 0.50, meaning
the imitations registered as **more human than the real humans**.

Why? Think about what real corporate email is: rushed, typo-ridden, fragmentary, weird. Highly
*unpredictable*. Now think about what an imitation is: an LLM writing fluently while copying a
human's patterns. The imitation is engineered — by the very nature of the attack — to sit in the
"looks human" zone of exactly the statistic these detectors measure. Asking "is this generic AI
text?" is the wrong question, because a good imitation *isn't generic AI text*. The right question
is "is this *Alice*?" — and only a personalized system can ask it.

This result also explained, in hindsight, what v11–v13's forgery-detection ability actually was:
sensitivity to generator quirks, not true "AI-ness" detection — which is exactly why it plateaued
and didn't transfer.

**Shortcut 2: someone else's style model.** A published "content-independent style embedding"
(StyleDistance) promised maps organized by pure style. Swapped into our pipeline, it was worse than
our own fine-tuned encoder on **every** test slice — including the cross-topic tests its design
specifically targets, and dramatically worse on forgeries (0.56 vs our 0.83). The lesson wasn't
"style-independence is a bad idea" — it's that a model trained on foreign data (synthetic
paraphrases, no email) can't compete with one adapted to the actual domain and threat. The
*technique* was worth stealing; the *artifact* wasn't. Any fix would have to be trained into our
own encoder.

With both shortcuts dead, exactly one path remained: make our own authorship model better.

### 7.3 v12: fighting variety with variety

Hypothesis: if training against one forger teaches that forger's accent, training against *several
different* forgers should force the model to find what forgeries share — or better, to stop relying
on generator quirks at all and lean harder on the sender's authentic style.

v12 trained with imitations from two vendors (GPT-4o-mini + Llama-3.1-70B), still evaluating only
on the held-out Claude+Gemini. It jumped from 0.690 to **0.871**. Big, real (far outside the error
bars), and the first confirmation that the gap was attackable. But at a deployable threshold it
still caught only ~39–45% of forgeries. Good; not shippable.

### 7.4 v13: the plateau that redirected everything

Obvious next move: add a *third* vendor (DeepSeek), and more synthetic volume. Result: **0.872** —
statistically identical to v12. More generator diversity had stopped paying.

Plateaus are informative: they tell you the bottleneck is elsewhere. The diagnosis went like this.
The model knew only **44 sender identities** during training (that's every usable sender in the
public Enron email corpus — the classic research dataset of real corporate email). With so few
examples of "what distinguishes one person from another," the model's concept of personal style
was too coarse — its territories on the style map were big fuzzy blobs. A competent imitation
lands *inside* a fuzzy blob. No amount of additional forgery examples fixes blurry blobs; you need
the blobs to be sharp. Which means: **more authors**, not more forgers.

### 7.5 v14: the experiment that "failed" perfectly

v14 expanded training identities from 44 to **844** authors by adding 800 bloggers from a public
blog corpus. It also — due to a mechanical limitation in the batch-construction code at the time —
dropped the synthetic imitations from training. Results, on the two axes we care about:

- **Content-invariance: breakthrough.** On a standard external benchmark (PAN cross-topic
  authorship verification — deciding whether two texts *on completely different topics* share an
  author, using data we never train on), performance jumped 0.779 → **0.879**. That number had
  been stuck for the project's entire history. Why did more authors fix it? With only 44 authors,
  each author's favorite *topics* are a usable cheat for telling them apart ("emails about
  gas trading = probably that guy"). With 844 authors across wildly different subject matter, topic
  stops being discriminative — hundreds of authors share topics — and the pull-together/push-apart
  pressure has no choice but to latch onto *how* people write instead of *what* they write about.
- **Forgery detection: collapse.** AUC fell to 0.610, barely above chance. With no imitations in
  training, nothing pushed fake-Alice away from real-Alice, and the model simply lost that skill.

So v14 shipped nothing but *proved* everything: it isolated the two levers cleanly.
**Identity diversity buys content-invariance. Synthetic imitations buy forgery detection.** Each
lever moves its own axis; neither substitutes for the other. A "failure" this informative is
better than most successes.

### 7.6 v14b: the synthesis — and why one sampler bug had held it back

The reason v14 had to drop synthetics was mundane: the component that assembles training batches
tied the *length of a training epoch* to the number of synthetic pairs (44), which meant the 800
blog authors would barely ever be visited. Fixing the sampler let one training run do both things
at once: guarantee synthetic (real, imitation) pairs in every batch *and* cycle through all 844
authors. (The fix was verified to leave all previous versions' behavior byte-for-byte identical —
so no old result was silently changed by it.)

v14b = 844 authors + multi-vendor synthetics + the fixed sampler. Result: **0.980 ± 0.003**, the
best score on *every* axis simultaneously — held-out forgeries (Claude 0.972, Gemini 0.953),
wrong-humans (0.967), cross-topic (0.857), short texts, everything. In deployment terms, at the
same 5% false-alarm budget where v12 let 61% of forgeries through, **v14b lets 12% through** —
against generators it has never seen.

**The most interesting scientific finding** came from completing the picture. Notice the versions
give us three corners of a 2×2 grid (few/many authors × with/without synthetics). We trained the
missing fourth corner (44 authors, no synthetics) specifically so the comparison would be clean:

| Held-out forgery AUC | no synthetics | with synthetics |
|---|---|---|
| **44 authors** | 0.640 | 0.871 |
| **844 authors** | 0.610 | **0.980** |

Read it carefully — the levers are **not independent**:

- Left column: going 44 → 844 authors *without* synthetics does **nothing** for forgery detection
  (0.640 → 0.610; flat).
- Top row: synthetics alone are worth +0.23 (0.640 → 0.871).
- Bottom-right: with both, the synthetic gain grows to **+0.37** (0.610 → 0.980).

The whole (0.980) exceeds what the parts predict — a **super-additive interaction**. The intuition:
a sharper style map (more authors) is only useful against forgeries if training also *shows* the
model forgeries to separate; and forgery examples are only fully exploitable if the map is sharp
enough that imitations have somewhere distinct to be pushed *to*. Sharp blobs + an adversary to
push away from them = imitations have nowhere left to hide. Either ingredient alone is a lock
without a key.

### 7.7 The final exam: forgers from outside the whole project

One last worry. Over months of iteration, Claude+Gemini stopped being "unseen" in a subtle sense:
*we* saw their scores repeatedly and made decisions accordingly. Could the project as a whole have
overfit to them? So we ran a test with fresh forgeries from **Qwen-2.5-72B and DeepSeek-V3** —
vendors involved in neither training nor our recurring evaluations. v14b scored **AUC 0.996**. If
anything, stronger. The skill generalizes to attackers' tools that nobody — model or researchers —
ever tuned against.

(That same test produced a beautifully instructive artifact: with forgeries separated near-
perfectly, the automatic synthetic-anchored threshold landed in a no-man's-land and produced a
garbage wrong-human error reading with enormous variance — a *threshold* pathology, not a model
one, since the wrong-human ranking remained excellent at 0.97. It's the strongest demonstration of
Section 6.4's rule: as forgery detection saturates, you *must* anchor your operating point on the
wrong-human axis.)

---

## 8. The honesty rules: how we avoid fooling ourselves

A theme runs through everything above: in this domain, **the ways to accidentally cheat outnumber
the ways to genuinely succeed**, and most of the project's discipline goes into measurement. The
rules, collected:

1. **Evaluate on attackers you didn't train against.** (Section 7.1.) The single most important
   rule. Anything else grades the model on memorization.
2. **Always report the wrong-human axis alongside the forgery axis.** (Section 6.4.) Optimizing
   one impostor class can silently break the other. Guardrail: ≤10% wrong-human false-accepts.
3. **Never let the training process grade itself on the easy class.** During training, the system
   periodically saves checkpoints and keeps the "best" one — best *according to some monitored
   metric*. An early version monitored forgery-detection only; it dutifully selected a checkpoint
   that was great at forgery detection *and leaked 60% of wrong-human mail*. This is
   **Goodhart's law** — "when a measure becomes a target, it ceases to be a good measure" — running
   at machine speed: the selection process will exploit any blind spot in its target. Fix: the
   monitor now watches the *worst* of the two impostor axes, so there is no blind spot to exploit.
4. **Distrust in-training dashboards for cross-version comparisons.** Each training run's live
   metrics are computed against that run's own training-style data. By that dashboard, v11 was our
   best model ever. On the real task it was the worst. The only comparable numbers come from the
   one fixed held-out evaluation that every version takes identically.
5. **Repeat measurements; publish error bars; reproduce the headline from scratch.**
   (Section 6.5.)
6. **Report the failures.** The zero-shot detector failure, the StyleDistance failure, the v14
   collapse, the threshold artifact — each constrains what a reader should believe and what a
   future maintainer should re-attempt. A results document with only wins is a red flag, not a
   comfort.

---

## 9. What it still can't do

Stated plainly, because trust in the numbers above depends on candor about these:

- **Comparing a very short text to a very long one is our weakest skill.** Length changes writing
  style (a two-line reply and a two-page memo by the same person genuinely differ), and
  early-generation models scored near chance when verifying across a large length gap on unseen
  authors. The current model is far better (AUC 0.93 on mixed-length pairs) but it's still the
  weakest slice at deployment thresholds (catching 71% at the 5% budget). One plausible fix — using
  only similar-length enrollment emails for comparison — was tested and made things *worse* (it
  throws away evidence). The real fixes (length-aware thresholds; cross-length adversarial
  examples in training) are queued but unproven.
- **The evaluation rests on 44 profiled senders.** That's every usable sender in the public Enron
  corpus, and it makes the strictest metric (TPR@1%) statistically noisy (±0.05–0.06). The
  headline metrics (AUC, TPR@5%) are solid; the 1% numbers deserve skepticism until we add a second
  evaluation corpus.
- **Threshold-level numbers wobble across retrains.** AUC reproduces almost exactly; TPR@5% varies
  ~±0.04 between training runs. Quote 0.87–0.91, not a point value, and calibrate the threshold on
  each deployment's own data.
- **We test attackers who imitate, not attackers who probe.** Our adversary writes imitations
  blind. An attacker with access to the system's scores, iterating until acceptance, is a
  different (harder) threat we have not yet evaluated.
- **English business email and blogs only.** Other languages, and radically different registers,
  are unmeasured.
- **One evaluation corpus vs. training overlap caveat.** Our blog-based content-invariance number
  (0.909) partially overlaps the training data source; the *clean* out-of-domain evidence is the
  PAN cross-topic benchmark (0.857). We always cite the clean number first.

---

## 10. Glossary

| Term | Plain meaning |
|---|---|
| **AUC** | Probability the system ranks a random genuine email above a random forgery. 0.5 = useless, 1.0 = perfect. Grades ranking quality, ignores thresholds. |
| **BEC** | Business email compromise — fraud via impersonation emails. |
| **Bootstrap** | Statistical resampling trick to estimate how much a measured number would vary; where our confidence ranges come from. |
| **Centroid** | The average map position of a sender's enrolled emails — the center of their style fingerprint. |
| **Contrastive learning** | Training by pull-together (same author) / push-apart (different author) pressure on map positions. |
| **Embedding** | A text converted into a list of numbers — a position on a learned "map" where distance means dissimilarity. |
| **Encoder** | The model that turns an email into its embedding. |
| **Enrollment** | Building a sender's profile (centroid, spread, count) from known-genuine emails. No learning involved. |
| **Few-shot imitation** | A forgery written by an LLM that was shown a few of the victim's real emails to copy. Our central threat. |
| **FPR / false-alarm rate** | Fraction of genuine emails wrongly flagged. |
| **FPR_other (guardrail)** | Fraction of *wrong-human* (different real person) emails wrongly accepted at the operating threshold. Must stay ≤10%. |
| **Goodhart's law** | When a measure becomes a target, it stops measuring. Why we monitor the worst-case axis, not the convenient one. |
| **Held-out** | Deliberately excluded from training so it can serve as a fair test. Our headline forgers (Claude, Gemini) are held-out. |
| **LLM** | Large language model — ChatGPT-class text generator. |
| **LoRA** | Fine-tuning method that freezes the original model and trains small add-on adjusters — cheap, and preserves pretrained knowledge. |
| **LUAR** | The authorship-pretrained encoder we build on; pretrained to judge "same author or not?" at Reddit scale. |
| **Mahalanobis distance** | Distance measured relative to a sender's own per-direction variability — knows the *shape* of their normal variation, not just its size. |
| **Seed** | The starting value for randomness. Re-running with different seeds tests whether a result was luck. |
| **Shortcut (learning)** | When a model succeeds via an accidental easy signal (e.g., one generator's quirks) that won't exist in the real world. |
| **Spread** | How widely a sender's own emails scatter around their centroid — how consistent a writer they are. |
| **Stylometry** | Identifying authors by quantifiable writing-style features. |
| **Synthetic hard negatives** | LLM-written imitations of training senders, injected into training as adversaries to push away from the real sender. |
| **Threshold / operating point** | The score cutoff separating "allow" from "flag" — the sensitivity dial; a business choice, not a model property. |
| **TPR@5% (catch rate)** | Fraction of forgeries caught when the threshold is set to allow at most 5% false alarms. Our primary deployment metric: 0.87–0.91 for the final model. |
| **z-score** | "How many units of *this sender's own normal variation* away is this email?" — distance normalized per person. |

---

*For the compact, evidence-linked version of everything here, see the whitepaper
(`docs/whitepaper/whitepaper.pdf`). Every number in this document appears there with its error
bars and its source file in `results/`.*
