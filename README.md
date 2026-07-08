# Telecom Incoming Request Processing Workflow

An AI triage desk for a telecom customer-operations inbox. Every incoming message is
classified by an LLM (type + urgency + **calibrated confidence**), routed through a
LangGraph state machine into a type-specific remediation branch, and fully logged to an
auditable case history. Anything the classifier isn't confident about goes to a human —
**the system never acts on a guess.**

**Live demo:** <https://telecom-request.streamlit.app> · **Screenshots:** [`screenshots/`](screenshots/)

## How it works

```
message → INTAKE → CLASSIFY (Groq openai/gpt-oss-120b, JSON mode)
        → ROUTER: confidence < 0.75 OR type = ESCALATION → HUMAN_REVIEW
                  otherwise → branch by type
        → BRANCH: 4–5 remediation steps from a config-driven taxonomy
        → LOG_OUTCOME: case + every action → SQLite audit trail
```

The router reads exactly two fields — confidence and type. A deterministic guardrail
caps confidence below the gate whenever the model can't name a sub-topic, so the
safety property is structural, not model-dependent
(see [docs/calibration_note.md](docs/calibration_note.md)).

## The five branches

| Branch | Urgency | SLA | Remediation steps |
|---|---|---|---|
| Billing Dispute | HIGH | 24h | extract details → empathetic draft → route Billing Resolution Team* → priority case → follow-up 2h* |
| Network Complaint | HIGH | 8h | extract location/service → draft ack → route Network Ops Centre* → priority case → follow-up 4h* |
| Service Request | MEDIUM | 48h | extract request → route Provisioning* → confirmation draft with timeline → open case |
| General Enquiry | LOW | — | classify sub-topic → **KB-grounded answer** → resolve → log |
| Human Review | CRITICAL | — | pause automation → urgent draft → notify supervisor* → queue with AI reasoning |

\* simulated integration — rendered in the UI and written to the audit log, clearly badged; a webhook in production.

## Run it locally

```bash
pip install -r requirements.txt
cp .env.example .env          # add your GROQ_API_KEY (free: console.groq.com)
streamlit run app.py
```

Python 3.11+. Without a key the app still loads and explains what to set.
Tests (41, all offline — the LLM is mocked): `python -m pytest`

## One end-to-end example per branch (live outputs)

| Input | Classified | Outcome |
|---|---|---|
| "My bill shows ₹4,200 but my plan is ₹799…" | BILLING_DISPUTE · HIGH · 0.98 | draft citing ₹4,200 → Billing team* → 24h SLA → OPEN |
| "No signal at all in Baner, Pune since yesterday…" | NETWORK_COMPLAINT · HIGH · 0.98 | location extracted → NOC* → 8h SLA → OPEN |
| "Please upgrade my ₹299 plan to the ₹499 plan…" | SERVICE_REQUEST · MEDIUM · 0.98 | Provisioning* → confirmation with 48h timeline → OPEN |
| "What prepaid plans do you have under ₹500…" | GENERAL_ENQUIRY · LOW · 0.98 | answer grounded in the plans KB entry → RESOLVED |
| "Third complaint! Fix it today or I go to consumer court." | ESCALATION · CRITICAL · 0.85 | automation paused → supervisor alerted* → HUMAN_REVIEW |
| "hi sim not wrking plz do the needful…" (garbled) | BILLING_DISPUTE · **0.68** | below the 0.75 gate → HUMAN_REVIEW |

## What's real vs simulated

**Real:** LLM classification and drafting (Groq, Gemini fallback), routing logic, SQLite
audit trail (`cases` + `actions` tables), dashboard analytics, batch processing with
rate-limit backoff. **Simulated:** team hand-offs, supervisor alerts, follow-up timers —
each is a labelled artifact + audit row; production would swap one handler line per
step for a webhook, with no change to the graph or taxonomy.

## Design notes

- **Config-driven taxonomy** ([src/taxonomy.py](src/taxonomy.py)) — branches, SLAs, steps and
  classifier definitions are data; a sixth branch is one spec entry, zero routing changes.
- **Never crashes on bad model output** — JSON repair retry, then safe fallback to human review.
- **Audit spine** — every classification decision and every step is a row in `cases.db`;
  the Case Log and Dashboard tabs are just reads over it.

Full design rationale: [ARCHITECTURE.md](ARCHITECTURE.md)

## Next steps (given more time)

Real integrations behind the existing step handlers (ticketing, Slack, email), a
reviewer feedback loop to tune the confidence threshold from outcomes, per-agent
queues with SLA breach alerts, and multi-turn case threading.

## Local demo & batch run

Live app: https://telecom-request.streamlit.app/

To reproduce the batch demo locally (populates `cases.db` with the 16 sample requests):

```bash
pip install -r requirements.txt
cp .env.example .env   # set GROQ_API_KEY in .env if you want real LLM calls
python scripts/checkpoint_phase3.py
```

Alternatively, run the app and use the Batch tab: open `streamlit run app.py`, go to
the Batch tab, click "Load sample dataset" then "Process" (the UI will show the
summary: e.g. "16 processed · 14 auto-handled · 2 sent to human review").

If the Process Request tab errors on submit, confirm the `GROQ_API_KEY` secret in your
Streamlit Cloud app settings (quotes required) and restart the app.
