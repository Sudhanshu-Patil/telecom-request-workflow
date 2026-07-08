# Calibration note — confidence on garbled input

**Sample:** row 16 of `data/sample_requests.csv`, a deliberately garbled message:

> hi sim not wrking plz do the needful... recharge done yesterda ₹239 but stil showing no plan activ?? also net very slow som times. 98220 monthly custmer

**Routing gate:** classifier confidence < 0.75 → HUMAN_REVIEW. The router reads
*confidence + request_type only*; nothing below changes that.

All runs 2026-07-08, Groq API, temperature 0.1, one live sample per condition
unless stated.

## 1 · Observation (llama-3.3-70b-versatile)

In the 16-row acceptance batch, row 16 classified as **NETWORK_COMPLAINT @
0.80** with `sub_topic: "other"` — confident enough to clear the gate while
simultaneously unable to name a sub-topic. It routed to the network branch
instead of human review. (The other ambiguous sample, row 15, correctly
returned ESCALATION @ 0.90 → HUMAN_REVIEW.)

## 2 · Prompt calibration attempted — measured ineffective

One instruction was added to the classifier prompt:

> Hard calibration rule: a mixed, garbled, or incoherent message -- especially
> one where no listed sub_topic clearly applies and you would return "other" --
> MUST receive confidence below 0.75.

Re-run on the same input: **NETWORK_COMPLAINT @ 0.80, sub_topic "other" —
unchanged.** The model produced the exact combination the rule forbids.
Conclusion: prompt-level confidence calibration of llama-3.3-70b is unreliable
at the margin. The instruction was kept (it can only push in the safe
direction) but could not be trusted as the mechanism.

## 3 · Model switch (openai/gpt-oss-120b)

`llama-3.3-70b-versatile` was deprecated on the Groq free tier, so `GROQ_MODEL`
moved to `openai/gpt-oss-120b`. Full 16-row live re-run:

- **Row 16: BILLING_DISPUTE @ 0.68 → HUMAN_REVIEW** (confidence gate) ✓
- Row 15: ESCALATION @ 0.85 → HUMAN_REVIEW (explicit override) ✓
- Rows 1–14: identical routing to before, confidence **0.95–0.98**
- Zero errors

The confidence distribution is now cleanly separated around the 0.75 gate:
~0.97 on clear-cut messages vs ~0.68 on the garbled one — the behaviour the
prompt asks for, honoured by the stronger model.

## 4 · Deterministic guardrail (structural, model-independent)

Sampling varies (across runs the same input has returned `sub_topic` "other"
and "refund_request"), and step 2 showed the "confident-but-can't-name-it"
failure is real. So the safety property is now enforced in code, applied
post-validation in `llm.classify` — **not** in the router:

> if `sub_topic == "other"` and `request_type != ESCALATION` and confidence >
> 0.70 → cap confidence at 0.70 and append *"confidence capped: no clear
> sub-topic identified"* to the reasoning.

0.70 sits below the 0.75 gate, so a capped result always reaches a human. The
ESCALATION exemption exists because that label routes to HUMAN_REVIEW by type
regardless. Covered by three smoke tests (cap fires; clear/escalation/already-low
results untouched; cap < threshold invariant).

### Verification (live, guardrail active)

| Row | Result | Routed | Note |
|---|---|---|---|
| 1 (clear billing) | BILLING_DISPUTE @ 0.98, `overcharge` | billing branch | unaffected |
| 15 (angry, churn threat) | ESCALATION @ 0.90, `churn_threat` | HUMAN_REVIEW | type override |
| 16 (garbled) | BILLING_DISPUTE @ 0.68, `refund_request` | HUMAN_REVIEW | model below gate on its own this run; guardrail is the backstop for runs where it isn't |

**Net effect:** "high confidence with no identifiable sub-topic" can no longer
clear the gate on any model, and the routing contract remains two fields only.
