"""Pydantic schemas and enums shared across the workflow.

These are the data contracts between the LLM layer, the LangGraph nodes,
the SQLite audit trail, and the Streamlit UI. Anything that crosses a
module boundary is one of these types.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

# Display timezone for the ops-facing UI; storage stays UTC so the audit
# trail is canonical. IST has no DST, so a fixed offset is exact and avoids
# a tz-database dependency on Windows.
IST = timezone(timedelta(hours=5, minutes=30), "IST")


class RequestType(str, Enum):
    """Labels the classifier may assign to an incoming request.

    ESCALATION never gets its own remediation branch: the router sends it,
    together with any low-confidence result, to the HUMAN_REVIEW branch.
    """

    BILLING_DISPUTE = "BILLING_DISPUTE"
    NETWORK_COMPLAINT = "NETWORK_COMPLAINT"
    SERVICE_REQUEST = "SERVICE_REQUEST"
    GENERAL_ENQUIRY = "GENERAL_ENQUIRY"
    ESCALATION = "ESCALATION"


class Urgency(str, Enum):
    """How fast operations must react."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class CaseStatus(str, Enum):
    """Lifecycle of a case in the audit trail."""

    OPEN = "OPEN"
    RESOLVED = "RESOLVED"
    HUMAN_REVIEW = "HUMAN_REVIEW"


class Branch(str, Enum):
    """Remediation branches implemented by the graph -- the taxonomy keys."""

    BILLING_DISPUTE = "BILLING_DISPUTE"
    NETWORK_COMPLAINT = "NETWORK_COMPLAINT"
    SERVICE_REQUEST = "SERVICE_REQUEST"
    GENERAL_ENQUIRY = "GENERAL_ENQUIRY"
    HUMAN_REVIEW = "HUMAN_REVIEW"


class ClassificationResult(BaseModel):
    """Structured classifier output -- the contract with the LLM.

    The ``before`` validators are deliberately forgiving (casing, None,
    slightly out-of-range confidence) because LLM JSON is only mostly
    well-behaved; anything truly unusable still fails validation and takes
    the repair-then-human-review path in llm.py.
    """

    request_type: RequestType
    urgency: Urgency
    confidence: float = Field(ge=0.0, le=1.0)
    sub_topic: str = "other"
    extracted_entities: dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""

    @field_validator("request_type", "urgency", mode="before")
    @classmethod
    def _normalise_label(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().upper().replace(" ", "_").replace("-", "_")
        return value

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, value: Any) -> Any:
        try:
            return min(1.0, max(0.0, float(value)))
        except (TypeError, ValueError):
            return value  # let pydantic report the real error

    @field_validator("reasoning", mode="before")
    @classmethod
    def _none_to_empty(cls, value: Any) -> Any:
        return "" if value is None else value

    @field_validator("sub_topic", mode="before")
    @classmethod
    def _subtopic_or_other(cls, value: Any) -> Any:
        """Blank/missing sub_topic must surface as an explicit 'other'."""
        if value is None:
            return "other"
        if isinstance(value, str):
            cleaned = value.strip().lower()
            return cleaned or "other"
        return value

    @field_validator("extracted_entities", mode="before")
    @classmethod
    def _none_to_dict(cls, value: Any) -> Any:
        return {} if value is None else value


class RemediationAction(BaseModel):
    """One executed remediation step -- one row in the audit trail."""

    case_id: str
    step_name: str
    detail: str
    artifact: dict[str, Any] | None = None
    simulated: bool = False
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CaseRecord(BaseModel):
    """A processed request -- one row in the case log."""

    case_id: str
    raw_text: str
    classification: ClassificationResult
    branch: Branch
    status: CaseStatus
    sla_deadline: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def generate_case_id() -> str:
    """Ops-scannable unique case id, e.g. ``TC-20260707-4F2A9C``."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"TC-{stamp}-{uuid.uuid4().hex[:6].upper()}"
