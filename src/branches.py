"""Generic remediation-step executor driven by the taxonomy step specs.

Deliberately contains no per-request-type code: every branch runs through
the same ``run_branch`` loop, and behavioural differences are expressed as
``StepSpec`` data (kind + params) in taxonomy.py. If a branch ever needs
special behaviour, it gets a new step kind or a new param -- never an
``if branch == X`` here. Handlers only *describe* what happened; all
persistence is deferred to the graph's single log_outcome node.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from . import kb
from .models import IST, CaseStatus, ClassificationResult, RemediationAction
from .taxonomy import BranchSpec, StepKind, StepSpec

# Sent instead of an LLM answer when no KB entry matches an enquiry: a fixed,
# safe pointer beats an ungrounded generation.
_KB_FALLBACK = (
    "Thank you for reaching out. We have noted your question; the self-care app "
    "has up-to-date details on plans, payments, roaming and coverage, and a care "
    "agent will follow up personally if anything more is needed."
)


@dataclass(frozen=True)
class StepContext:
    """Read-only facts every step handler may draw on."""

    case_id: str
    raw_text: str
    classification: ClassificationResult
    generate_fn: Callable[..., str]


@dataclass(frozen=True)
class StepOutcome:
    """What one executed step reports back to the executor loop."""

    detail: str
    artifact: dict[str, Any] | None = None
    status: CaseStatus | None = None
    sla_deadline: datetime | None = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _handle_extract(step: StepSpec, spec: BranchSpec, ctx: StepContext) -> StepOutcome:
    """Record the structured fields this branch cares about."""
    entities = dict(ctx.classification.extracted_entities)
    captured: dict[str, Any] = {}
    for field in step.params.get("fields", []):
        if field == "sub_topic":
            captured[field] = ctx.classification.sub_topic
        else:
            captured[field] = entities.pop(field, None)
    shown = "; ".join(
        f"{key}={'not stated' if value in (None, '') else value}"
        for key, value in captured.items()
    )
    artifact: dict[str, Any] = {"captured": captured}
    if entities:
        artifact["additional_entities"] = entities
    return StepOutcome(detail=f"Captured: {shown}", artifact=artifact)


def _handle_draft_reply(step: StepSpec, spec: BranchSpec, ctx: StepContext) -> StepOutcome:
    """LLM-draft a customer-facing reply shaped entirely by step params."""
    tone = step.params.get("tone", "professional")
    entities = ctx.classification.extracted_entities
    cited = {field: entities[field] for field in step.params.get("cite", []) if entities.get(field)}
    lines = [
        f"Write a {tone} acknowledgement to a telecom customer on behalf of the "
        f"{spec.label} desk.",
        f'Customer message: """{ctx.raw_text}"""',
    ]
    if cited:
        lines.append(
            "Refer explicitly to these details: "
            + "; ".join(f"{key} = {value}" for key, value in cited.items())
        )
    hours = step.params.get("timeline_hours")
    if hours:
        lines.append(f"State clearly that the request is expected to be completed within {hours} hours.")
    if step.params.get("no_promises"):
        lines.append(
            "Reassure the customer that a specialist is personally reviewing the "
            "case, but do not promise any specific outcome or resolution time."
        )
    draft = ctx.generate_fn("\n".join(lines))
    return StepOutcome(
        detail=f"Customer-facing draft prepared ({tone} tone) for ops review before sending",
        artifact={"draft_reply": draft, "tone": tone},
    )


def _handle_route(step: StepSpec, spec: BranchSpec, ctx: StepContext) -> StepOutcome:
    """Simulated hand-off to an internal team (a webhook in production)."""
    team = step.params["team"]
    notice = {
        "to": team,
        "case_id": ctx.case_id,
        "request_type": ctx.classification.request_type.value,
        "urgency": ctx.classification.urgency.value,
        "sub_topic": ctx.classification.sub_topic,
        "summary": ctx.classification.reasoning or ctx.raw_text[:140],
    }
    return StepOutcome(
        detail=f"Hand-off notice issued to {team} (simulated integration)",
        artifact={"routing_notice": notice},
    )


def _handle_open_case(step: StepSpec, spec: BranchSpec, ctx: StepContext) -> StepOutcome:
    """Open the tracked case and start the SLA clock from the branch spec."""
    priority = bool(step.params.get("priority", False))
    deadline = _now() + timedelta(hours=spec.sla_hours) if spec.sla_hours is not None else None
    parts = ["Case opened" + (" with priority flag" if priority else "")]
    if deadline is not None:
        parts.append(f"SLA {spec.sla_hours}h -> due {deadline.astimezone(IST):%d %b %Y %H:%M} IST")
    return StepOutcome(
        detail="; ".join(parts),
        artifact={
            "priority": priority,
            "sla_hours": spec.sla_hours,
            "sla_deadline": deadline.isoformat() if deadline else None,
        },
        status=CaseStatus.OPEN,
        sla_deadline=deadline,
    )


def _handle_follow_up(step: StepSpec, spec: BranchSpec, ctx: StepContext) -> StepOutcome:
    """Schedule a (simulated) follow-up reminder at params['hours']."""
    hours = step.params["hours"]
    due = _now() + timedelta(hours=hours)
    task = {
        "due_at": due.isoformat(),
        "action": "Verify progress with the assigned team and update the customer.",
    }
    return StepOutcome(
        detail=f"Follow-up reminder scheduled for +{hours}h ({due.astimezone(IST):%H:%M} IST, simulated)",
        artifact={"follow_up_task": task},
    )


def _handle_kb_answer(step: StepSpec, spec: BranchSpec, ctx: StepContext) -> StepOutcome:
    """Answer an enquiry grounded in the kb.py entry for its sub-topic."""
    hit = None
    if ctx.classification.sub_topic not in ("", "other"):
        hit = kb.retrieve(ctx.classification.sub_topic)
    if hit is None:
        hit = kb.retrieve(ctx.raw_text)
    if hit is None:
        return StepOutcome(
            detail="No knowledge-base entry matched; standard self-care pointer used "
            "instead of an ungrounded answer",
            artifact={"answer": _KB_FALLBACK, "grounded": False},
        )
    topic, entry = hit
    prompt = (
        "Answer the customer's question using ONLY the approved knowledge-base "
        "entry below. Do not add facts that are not in the entry. If the entry "
        "does not fully answer the question, share what it does say and point "
        "the customer to the self-care app for the rest.\n"
        f"Knowledge-base topic: {topic}\n"
        f'Entry: """{entry}"""\n'
        f'Customer question: """{ctx.raw_text}"""'
    )
    answer = ctx.generate_fn(prompt)
    return StepOutcome(
        detail=f"Grounded answer generated from knowledge-base topic '{topic}'",
        artifact={"kb_topic": topic, "kb_entry": entry, "answer": answer, "grounded": True},
    )


def _handle_resolve(step: StepSpec, spec: BranchSpec, ctx: StepContext) -> StepOutcome:
    return StepOutcome(
        detail="Case marked RESOLVED -- answered automatically, no human action required",
        status=CaseStatus.RESOLVED,
    )


def _handle_log(step: StepSpec, spec: BranchSpec, ctx: StepContext) -> StepOutcome:
    return StepOutcome(detail="Resolution note recorded in the audit trail")


def _handle_pause(step: StepSpec, spec: BranchSpec, ctx: StepContext) -> StepOutcome:
    return StepOutcome(
        detail="Automated processing paused -- no automatic reply will be sent for this case",
        status=CaseStatus.HUMAN_REVIEW,
    )


def _handle_notify(step: StepSpec, spec: BranchSpec, ctx: StepContext) -> StepOutcome:
    """Simulated alert to a person (Slack/email in production)."""
    who = step.params.get("who", "Duty Supervisor")
    alert = {
        "to": who,
        "case_id": ctx.case_id,
        "request_type": ctx.classification.request_type.value,
        "confidence": ctx.classification.confidence,
        "reasoning": ctx.classification.reasoning,
    }
    return StepOutcome(
        detail=f"Alert raised to {who} (simulated integration)",
        artifact={"supervisor_alert": alert},
    )


def _handle_queue_review(step: StepSpec, spec: BranchSpec, ctx: StepContext) -> StepOutcome:
    """Park the case for a human with everything they need to decide."""
    entry = {
        "queue": "human_review",
        "case_id": ctx.case_id,
        "classified_as": ctx.classification.request_type.value,
        "confidence": ctx.classification.confidence,
        "classifier_reasoning": ctx.classification.reasoning,
        "original_text": ctx.raw_text,
    }
    return StepOutcome(
        detail="Queued for human review with the classifier's full reasoning attached",
        artifact={"review_queue_entry": entry},
        status=CaseStatus.HUMAN_REVIEW,
    )


HANDLERS: dict[StepKind, Callable[[StepSpec, BranchSpec, StepContext], StepOutcome]] = {
    StepKind.EXTRACT: _handle_extract,
    StepKind.DRAFT_REPLY: _handle_draft_reply,
    StepKind.ROUTE: _handle_route,
    StepKind.OPEN_CASE: _handle_open_case,
    StepKind.FOLLOW_UP: _handle_follow_up,
    StepKind.KB_ANSWER: _handle_kb_answer,
    StepKind.RESOLVE: _handle_resolve,
    StepKind.LOG: _handle_log,
    StepKind.PAUSE_AUTOMATION: _handle_pause,
    StepKind.NOTIFY: _handle_notify,
    StepKind.QUEUE_REVIEW: _handle_queue_review,
}


def run_branch(
    spec: BranchSpec, ctx: StepContext
) -> tuple[list[RemediationAction], CaseStatus, datetime | None]:
    """Execute every step of a branch, in taxonomy order.

    Returns the audit actions plus the final status / SLA deadline the step
    kinds decided on (OPEN_CASE -> OPEN, RESOLVE -> RESOLVED,
    PAUSE_AUTOMATION / QUEUE_REVIEW -> HUMAN_REVIEW).
    """
    actions: list[RemediationAction] = []
    status: CaseStatus | None = None
    sla_deadline: datetime | None = None
    for step in spec.steps:
        outcome = HANDLERS[step.kind](step, spec, ctx)
        actions.append(
            RemediationAction(
                case_id=ctx.case_id,
                step_name=step.title,
                detail=outcome.detail,
                artifact=outcome.artifact,
                simulated=step.simulated,
            )
        )
        if outcome.status is not None:
            status = outcome.status
        if outcome.sla_deadline is not None:
            sla_deadline = outcome.sla_deadline
    return actions, status or CaseStatus.OPEN, sla_deadline
