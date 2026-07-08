# Architecture & Design — Telecom Incoming Request Processing Workflow

## 1. Problem framing

A telecom operations team receives a continuous stream of customer messages — billing disputes, network complaints, service requests, and general enquiries — through a shared inbox. Today a human reads each one, decides what it is, and manually kicks off the right process. This is slow, inconsistent, and judgment-dependent.

This prototype demonstrates that an LLM-driven workflow can (a) classify each request with calibrated confidence, (b) execute a *distinct multi-step remediation path* per type — not a generic reply, and (c) keep a full audit trail so an operations team can trust and verify every automated decision. Anything the system is not confident about is routed to a human, never guessed.

## 2. System design

```
                        ┌─────────────────────────────────────────────┐
                        │              STREAMLIT UI                    │
                        │  Process │ Batch CSV │ Case Log │ Dashboard  │
                        └───────────────┬─────────────────────────────┘
                                        │ raw request text
                                        ▼
                              ┌──────────────────┐
                              │   INTAKE node    │  normalize, assign case_id
                              └────────┬─────────┘
                                       ▼
                              ┌──────────────────┐     LLM (Groq / JSON mode)
                              │  CLASSIFY node   │────► type, urgency,
                              └────────┬─────────┘     confidence, entities,
                                       │               reasoning
                        conditional edge (router):
                        confidence < 0.75 OR type == ESCALATION → HUMAN_REVIEW
                                       │
        ┌──────────────┬───────────────┼───────────────┬──────────────┐
        ▼              ▼               ▼               ▼              ▼
 ┌────────────┐ ┌────────────┐ ┌─────────────┐ ┌────────────┐ ┌────────────┐
 │  BILLING   │ │  NETWORK   │ │  SERVICE    │ │  GENERAL   │ │   HUMAN    │
 │  DISPUTE   │ │ COMPLAINT  │ │  REQUEST    │ │  ENQUIRY   │ │   REVIEW   │
 │ (High)     │ │ (High)     │ │ (Medium)    │ │ (Low)      │ │ (Critical) │
 │ extract →  │ │ extract →  │ │ extract →   │ │ sub-topic →│ │ pause auto │
 │ ack draft →│ │ ack draft →│ │ route Prov →│ │ KB answer →│ │ → urgent   │
 │ route Bill →│ │ route NOC →│ │ confirm →   │ │ resolve →  │ │ ack → notify│
 │ SLA 24h →  │ │ SLA 8h →   │ │ SLA 48h     │ │ log        │ │ supervisor │
 │ f-up 2h    │ │ f-up 4h    │ │             │ │            │ │ → HITL queue│
 └─────┬──────┘ └─────┬──────┘ └──────┬──────┘ └─────┬──────┘ └─────┬──────┘
       └──────────────┴───────┬───────┴───────────────┴──────────────┘
                              ▼
                     ┌──────────────────┐
                     │  LOG_OUTCOME     │  case + every action row → SQLite
                     └──────────────────┘  (audit trail feeds Case Log +
                                            Dashboard tabs)
```

## 3. Key design decisions (and why — this is Slide 3 material)

**LangGraph state machine over if/else.** The branching *is* the assessment. A `StateGraph` with a conditional router edge makes the remediation topology explicit, testable, and extensible — adding a sixth branch is a taxonomy entry + node registration, zero changes to routing logic. It also produces an honest architecture diagram rather than a flowchart drawn after the fact.

**Config-driven taxonomy.** Request types, urgency, SLAs, and step sequences live in one data structure (`taxonomy.py`). Branch nodes are generic executors that read their step list from config. This is what "completeness of remediation strategy" looks like in code: the strategy is inspectable in one file, not scattered across functions.

**Confidence-gated human-in-the-loop.** The classifier must return a self-assessed confidence. Below 0.75 — or on explicit escalation signals — the request bypasses automation entirely and lands in a human-review queue with the model's reasoning attached. The system's core promise: *it never acts on a guess.* This implements the brief's "escalation override" optional enhancement as a first-class branch.

**Structured output with repair-and-fallback.** Classification uses LLM JSON mode validated by Pydantic. Parse failure → one repair attempt → fall back to HUMAN_REVIEW. The pipeline cannot crash on a malformed model response; worst case is a human looks at it, which is the correct failure mode for operations.

**Simulated integrations, honestly labeled.** Routing notices, supervisor alerts, and SLA timers are rendered in the UI and written to the audit log with a `simulated: true` flag. In production each becomes a webhook (Slack/email/CRM). Drawing this line explicitly is deliberate: the prototype proves the decision logic; integration is plumbing.

**SQLite audit trail as the spine.** Every classification decision and every remediation action is a row. The Case Log tab (full trail with filters) and Dashboard tab (volume by type, status mix, urgency distribution) are just reads over this table — three of the brief's four optional enhancements fall out of one design choice.

## 4. Data contracts

**ClassificationResult** (LLM output, Pydantic-validated):
`request_type`, `urgency`, `confidence: float`, `sub_topic`, `extracted_entities: dict`, `reasoning`

**CaseRecord**: `case_id`, `raw_text`, `classification (json)`, `branch`, `status (OPEN|RESOLVED|HUMAN_REVIEW)`, `sla_deadline`, `created_at`

**ActionRecord**: `case_id`, `step_name`, `detail`, `artifact (json, e.g. draft text)`, `simulated: bool`, `timestamp`

## 5. Sample dataset design

`data/sample_requests.csv` — 16 requests: 4 realistic messages per core type (varied tone, lengths, Indian telecom context: ₹ amounts, plan names, city names), plus 2 deliberately ambiguous messages (e.g., a billing question that reads like a complaint, a garbled message) that should land in HUMAN_REVIEW. The demo narrative: "and here's what happens when the AI *isn't* sure."

## 6. Risk register (2-day build)

| Risk | Mitigation |
|------|-----------|
| Groq free-tier rate limits during batch demo | Batch processes sequentially with small delay; Gemini fallback wired in `llm.py` |
| Streamlit Cloud deploy friction on Day 2 | Deploy a hello-world to Streamlit Cloud on Day 1 evening to burn the setup cost early |
| LLM classification flakiness in live demo | Screen-recorded backup demo captured before submission; sample set pre-tested |
| Scope creep | CLAUDE.md "What NOT to do" list; code freeze Day 2 midday |
