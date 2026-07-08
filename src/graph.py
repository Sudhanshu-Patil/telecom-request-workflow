"""LangGraph state machine: intake -> classify -> route -> branch -> log.

The conditional router is intentionally minimal: it reads exactly two
fields of the classification -- confidence and request_type -- and nothing
else. Branch behaviour lives in taxonomy.py step specs (executed by
branches.run_branch); every branch converges on the single log_outcome
node, which is the only place anything is written to SQLite.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from . import branches, db, llm
from .models import (
    Branch,
    CaseRecord,
    CaseStatus,
    ClassificationResult,
    RemediationAction,
    RequestType,
    generate_case_id,
)
from .taxonomy import CONFIDENCE_THRESHOLD, TAXONOMY, BranchSpec


class WorkflowState(TypedDict, total=False):
    """Everything a request accumulates on its way through the graph."""

    raw_text: str
    case_id: str
    created_at: datetime
    classification: ClassificationResult
    branch: Branch
    actions: list[RemediationAction]
    status: CaseStatus
    sla_deadline: datetime | None


def route_by_confidence(classification: ClassificationResult) -> Branch:
    """The routing rule from the brief, and nothing more.

    Decides on ``confidence`` and ``request_type`` ONLY; urgency, entities,
    wording etc. must never influence routing (the smoke tests feed
    adversarial classifications to enforce this).
    """
    if (
        classification.confidence < CONFIDENCE_THRESHOLD
        or classification.request_type is RequestType.ESCALATION
    ):
        return Branch.HUMAN_REVIEW
    return Branch(classification.request_type.value)


def build_workflow(
    classify_fn: Callable[[str], ClassificationResult] | None = None,
    generate_fn: Callable[..., str] | None = None,
    db_path: str | Path | None = None,
) -> Any:
    """Compile the request-processing graph.

    ``classify_fn``/``generate_fn`` default to the real LLM client but are
    injectable, so the tests and the Phase 2 checkpoint run fully offline
    against a throwaway database.
    """
    resolved_classify = classify_fn or llm.classify
    resolved_generate = generate_fn or llm.generate
    resolved_db = Path(db_path) if db_path is not None else db.DEFAULT_DB_PATH
    db.init_db(resolved_db)

    def intake(state: WorkflowState) -> dict[str, Any]:
        """Normalise the raw text and mint the case identity."""
        return {
            "raw_text": state["raw_text"].strip(),
            "case_id": generate_case_id(),
            "created_at": datetime.now(timezone.utc),
            "actions": [],
        }

    def classify(state: WorkflowState) -> dict[str, Any]:
        """Run the classifier; the decision itself is the first audit row."""
        result = resolved_classify(state["raw_text"])
        decision = RemediationAction(
            case_id=state["case_id"],
            step_name="Classification",
            detail=(
                f"Classified as {result.request_type.value} "
                f"(urgency {result.urgency.value}, confidence {result.confidence:.2f}). "
                f"{result.reasoning}"
            ).strip(),
            artifact={"classification": result.model_dump(mode="json")},
        )
        return {"classification": result, "actions": state["actions"] + [decision]}

    def make_branch_node(spec: BranchSpec) -> Callable[[WorkflowState], dict[str, Any]]:
        def branch_node(state: WorkflowState) -> dict[str, Any]:
            ctx = branches.StepContext(
                case_id=state["case_id"],
                raw_text=state["raw_text"],
                classification=state["classification"],
                generate_fn=resolved_generate,
            )
            actions, status, sla_deadline = branches.run_branch(spec, ctx)
            return {
                "branch": spec.key,
                "status": status,
                "sla_deadline": sla_deadline,
                "actions": state["actions"] + actions,
            }

        return branch_node

    def log_outcome(state: WorkflowState) -> dict[str, Any]:
        """Single persistence point: one case row + one row per action."""
        record = CaseRecord(
            case_id=state["case_id"],
            raw_text=state["raw_text"],
            classification=state["classification"],
            branch=state["branch"],
            status=state["status"],
            sla_deadline=state.get("sla_deadline"),
            created_at=state["created_at"],
        )
        db.insert_case(record, resolved_db)
        for action in state["actions"]:
            db.insert_action(action, resolved_db)
        return {}

    graph = StateGraph(WorkflowState)
    graph.add_node("intake", intake)
    graph.add_node("classify", classify)
    graph.add_node("log_outcome", log_outcome)
    for branch_key, spec in TAXONOMY.items():
        graph.add_node(branch_key.value, make_branch_node(spec))
        graph.add_edge(branch_key.value, "log_outcome")

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "classify")
    graph.add_conditional_edges(
        "classify",
        lambda state: route_by_confidence(state["classification"]).value,
        {branch_key.value: branch_key.value for branch_key in TAXONOMY},
    )
    graph.add_edge("log_outcome", END)
    return graph.compile()


def process_request(
    text: str,
    classify_fn: Callable[[str], ClassificationResult] | None = None,
    generate_fn: Callable[..., str] | None = None,
    db_path: str | Path | None = None,
) -> WorkflowState:
    """Convenience wrapper: run one request end-to-end, return final state."""
    workflow = build_workflow(classify_fn, generate_fn, db_path)
    return workflow.invoke({"raw_text": text})
