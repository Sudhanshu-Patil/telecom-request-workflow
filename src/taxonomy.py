"""Single source of truth for the request taxonomy and remediation strategy.

Everything the rest of the system knows about a request type is data in
this file: what the type means (fed verbatim to the classifier prompt), its
default urgency, its SLA, and the ordered remediation steps. ``graph.py``
and ``branches.py`` are generic consumers -- adding a sixth branch is one
``BranchSpec`` here and nothing else.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .models import Branch, RequestType, Urgency

# Classifier results below this confidence are never auto-actioned; the
# router in graph.py diverts them to the HUMAN_REVIEW branch.
CONFIDENCE_THRESHOLD: float = 0.75

# Sub-topics the classifier may assign to a GENERAL_ENQUIRY. kb.py must hold
# a grounded answer for each one (enforced by the smoke tests).
ENQUIRY_SUBTOPICS: tuple[str, ...] = ("plans", "roaming", "payments", "coverage")


class StepKind(str, Enum):
    """Generic step verbs that branches.py knows how to execute."""

    EXTRACT = "EXTRACT"                    # pull structured fields into the case
    DRAFT_REPLY = "DRAFT_REPLY"            # LLM-drafted customer-facing text
    ROUTE = "ROUTE"                        # hand-off notice to an internal team
    OPEN_CASE = "OPEN_CASE"                # start the SLA clock on a tracked case
    FOLLOW_UP = "FOLLOW_UP"                # schedule a follow-up reminder
    KB_ANSWER = "KB_ANSWER"                # grounded answer from the kb.py FAQ
    RESOLVE = "RESOLVE"                    # close the case as resolved
    LOG = "LOG"                            # note-only audit row
    PAUSE_AUTOMATION = "PAUSE_AUTOMATION"  # stop auto-resolution entirely
    NOTIFY = "NOTIFY"                      # alert a person (supervisor)
    QUEUE_REVIEW = "QUEUE_REVIEW"          # place in the human-review queue


class StepSpec(BaseModel):
    """One remediation step as config; branches.py interprets it."""

    model_config = ConfigDict(frozen=True)

    kind: StepKind
    title: str                              # shown on the remediation timeline
    params: dict[str, Any] = Field(default_factory=dict)
    simulated: bool = False                 # no real side effect; badged in UI + log


class BranchSpec(BaseModel):
    """Full remediation strategy for one branch."""

    model_config = ConfigDict(frozen=True)

    key: Branch
    label: str                              # ops-friendly display name
    description: str                        # classifier-facing definition
    sub_topics: tuple[str, ...] = ()        # classifier picks one of these or "other"
    default_urgency: Urgency
    sla_hours: int | None = None            # None = no SLA clock for this branch
    steps: tuple[StepSpec, ...]


TAXONOMY: dict[Branch, BranchSpec] = {
    Branch.BILLING_DISPUTE: BranchSpec(
        key=Branch.BILLING_DISPUTE,
        label="Billing Dispute",
        description=(
            "The customer disputes money we charged: a bill higher than the "
            "plan price, double debit, unexpected deduction, a charge for "
            "something they did not use, or a refund demand. Cues: rupee "
            "amounts quoted with a grievance, 'overcharged', 'wrong bill', "
            "'money deducted'."
        ),
        sub_topics=(
            "overcharge", "double_charge", "unexpected_deduction",
            "refund_request", "wrong_plan_billed",
        ),
        default_urgency=Urgency.HIGH,
        sla_hours=24,
        steps=(
            StepSpec(
                kind=StepKind.EXTRACT,
                title="Extract billing details",
                params={"fields": ["account_ref", "disputed_amount", "billing_period"]},
            ),
            StepSpec(
                kind=StepKind.DRAFT_REPLY,
                title="Draft empathetic acknowledgement",
                params={"tone": "empathetic", "cite": ["disputed_amount", "billing_period"]},
            ),
            StepSpec(
                kind=StepKind.ROUTE,
                title="Route to Billing Resolution Team",
                params={"team": "Billing Resolution Team"},
                simulated=True,
            ),
            StepSpec(
                kind=StepKind.OPEN_CASE,
                title="Open priority case (24h SLA)",
                params={"priority": True},
            ),
            StepSpec(
                kind=StepKind.FOLLOW_UP,
                title="Set follow-up reminder (2h)",
                params={"hours": 2},
                simulated=True,
            ),
        ),
    ),
    Branch.NETWORK_COMPLAINT: BranchSpec(
        key=Branch.NETWORK_COMPLAINT,
        label="Network Complaint",
        description=(
            "Service is degraded or down: no signal, call drops, slow or dead "
            "mobile data or broadband, an outage affecting an area, poor "
            "indoor coverage. Cues: 'no network since', place names, speed "
            "complaints, 'calls keep dropping'."
        ),
        sub_topics=(
            "no_signal", "call_drops", "slow_data",
            "area_outage", "poor_indoor_coverage",
        ),
        default_urgency=Urgency.HIGH,
        sla_hours=8,
        steps=(
            StepSpec(
                kind=StepKind.EXTRACT,
                title="Extract location and affected service",
                params={"fields": ["location", "service_affected", "duration"]},
            ),
            StepSpec(
                kind=StepKind.DRAFT_REPLY,
                title="Draft acknowledgement",
                params={"tone": "reassuring", "cite": ["location", "service_affected"]},
            ),
            StepSpec(
                kind=StepKind.ROUTE,
                title="Route to Network Operations Centre",
                params={"team": "Network Operations Centre"},
                simulated=True,
            ),
            StepSpec(
                kind=StepKind.OPEN_CASE,
                title="Open priority case (8h SLA)",
                params={"priority": True},
            ),
            StepSpec(
                kind=StepKind.FOLLOW_UP,
                title="Set follow-up reminder (4h)",
                params={"hours": 4},
                simulated=True,
            ),
        ),
    ),
    Branch.SERVICE_REQUEST: BranchSpec(
        key=Branch.SERVICE_REQUEST,
        label="Service Request",
        description=(
            "The customer wants us to do or change something on their "
            "account: plan upgrade/downgrade, new SIM or eSIM, replacement "
            "for a lost SIM, number porting, relocating a connection, add-on "
            "activation. Cues: 'please change/activate/shift/port', a "
            "requested future state rather than a problem."
        ),
        sub_topics=(
            "plan_change", "new_sim", "sim_replacement",
            "number_porting", "relocation", "addon_activation",
        ),
        default_urgency=Urgency.MEDIUM,
        sla_hours=48,
        steps=(
            StepSpec(
                kind=StepKind.EXTRACT,
                title="Extract request details",
                params={"fields": ["requested_change", "current_plan", "target_plan_or_service"]},
            ),
            StepSpec(
                kind=StepKind.ROUTE,
                title="Route to Provisioning Team",
                params={"team": "Provisioning Team"},
                simulated=True,
            ),
            StepSpec(
                kind=StepKind.DRAFT_REPLY,
                title="Draft confirmation with expected timeline",
                params={"tone": "confirmatory", "timeline_hours": 48},
            ),
            StepSpec(
                kind=StepKind.OPEN_CASE,
                title="Open case (48h SLA)",
                params={"priority": False},
            ),
        ),
    ),
    Branch.GENERAL_ENQUIRY: BranchSpec(
        key=Branch.GENERAL_ENQUIRY,
        label="General Enquiry",
        description=(
            "An informational question with nothing broken and nothing to "
            "change: plan prices and validity, roaming packs, payment methods "
            "and due dates, coverage availability. Cues: question phrasing, "
            "no grievance."
        ),
        sub_topics=ENQUIRY_SUBTOPICS,
        default_urgency=Urgency.LOW,
        sla_hours=None,
        steps=(
            StepSpec(
                kind=StepKind.EXTRACT,
                title="Classify enquiry sub-topic",
                params={"fields": ["sub_topic"]},
            ),
            StepSpec(
                kind=StepKind.KB_ANSWER,
                title="Generate grounded answer from knowledge base",
            ),
            StepSpec(kind=StepKind.RESOLVE, title="Mark case resolved"),
            StepSpec(kind=StepKind.LOG, title="Log resolution"),
        ),
    ),
    Branch.HUMAN_REVIEW: BranchSpec(
        key=Branch.HUMAN_REVIEW,
        label="Human Review",
        description=(
            "Safety-net branch; the classifier never selects it directly. It "
            "receives explicit ESCALATION results and any classification "
            f"whose confidence is below {CONFIDENCE_THRESHOLD}. Automation "
            "pauses and a person decides the next step."
        ),
        default_urgency=Urgency.CRITICAL,
        sla_hours=None,
        steps=(
            StepSpec(kind=StepKind.PAUSE_AUTOMATION, title="Pause auto-resolution"),
            StepSpec(
                kind=StepKind.DRAFT_REPLY,
                title="Draft urgent acknowledgement",
                params={"tone": "urgent and personal", "no_promises": True},
            ),
            StepSpec(
                kind=StepKind.NOTIFY,
                title="Notify supervisor",
                params={"who": "Duty Supervisor"},
                simulated=True,
            ),
            StepSpec(
                kind=StepKind.QUEUE_REVIEW,
                title="Queue for human review with classifier reasoning",
            ),
        ),
    ),
}

# Guidance for the one classifier label that has no same-named branch.
ESCALATION_DESCRIPTION: str = (
    "The message needs a human immediately regardless of topic: legal or "
    "regulator threats (consumer court, TRAI), extreme anger or abuse, many "
    "failed contact attempts, churn threats ('cancel everything today'), "
    "safety issues -- or the text is garbled/contradictory and you cannot "
    "tell what the customer wants."
)

# Sub-topics for the ESCALATION label (which has no BranchSpec of its own).
ESCALATION_SUBTOPICS: tuple[str, ...] = (
    "legal_threat",
    "repeated_complaints",
    "churn_threat",
    "abusive_or_distressed",
    "unintelligible",
)


def classifier_type_definitions() -> dict[str, str]:
    """``request_type -> definition`` map, fed verbatim to the classifier."""
    definitions = {
        spec.key.value: spec.description
        for spec in TAXONOMY.values()
        if spec.key is not Branch.HUMAN_REVIEW
    }
    definitions[RequestType.ESCALATION.value] = ESCALATION_DESCRIPTION
    return definitions


def classifier_subtopic_options() -> dict[str, tuple[str, ...]]:
    """``request_type -> valid sub_topics`` map fed to the classifier prompt.

    Built from the branch specs so the prompt can never drift from the
    taxonomy; "other" is always permitted on top of these.
    """
    options = {
        spec.key.value: spec.sub_topics
        for spec in TAXONOMY.values()
        if spec.key is not Branch.HUMAN_REVIEW
    }
    options[RequestType.ESCALATION.value] = ESCALATION_SUBTOPICS
    return options
