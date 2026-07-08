"""Streamlit UI for the telecom incoming-request processing workflow.

Presentation only: classification, routing, remediation and persistence all
live in src/ -- this file renders inputs and outcomes. Simulated
integrations are always labelled as such on every surface.
"""

from __future__ import annotations

import html
import json
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import db
from src.graph import build_workflow
from src.llm import LLMConfigError, LLMUnavailableError
from src.models import IST, Branch, CaseStatus, RequestType
from src.taxonomy import CONFIDENCE_THRESHOLD, TAXONOMY

st.set_page_config(page_title="Telecom Request Processor", page_icon="📡", layout="wide")

DATA_PATH = Path(__file__).parent / "data" / "sample_requests.csv"

# Palette: urgency chips follow the mandated grey/blue/orange/red; chart hues
# come from a CVD-validated set. LOW's grey deliberately recedes -- every place
# it appears also carries a text label (the "relief" rule for low-chroma marks).
URGENCY_COLORS = {"LOW": "#898781", "MEDIUM": "#2a78d6", "HIGH": "#eb6834", "CRITICAL": "#d03b3b"}
URGENCY_DOTS = {"LOW": "⚪", "MEDIUM": "🔵", "HIGH": "🟠", "CRITICAL": "🔴"}
URGENCY_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
STATUS_COLORS = {"OPEN": "#2a78d6", "RESOLVED": "#008300", "HUMAN_REVIEW": "#eda100"}
ACCENT = "#2a78d6"
GRID = "#e1e0d9"
INK_SOFT = "#52514e"

HAS_KEY = any(os.getenv(k) for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"))

db.init_db()


# --- small presentation helpers ------------------------------------------------


def urgency_badge(urgency: str) -> str:
    color = URGENCY_COLORS.get(urgency, "#898781")
    return (
        f'<span style="background:{color};color:#fff;padding:2px 10px;'
        f'border-radius:12px;font-size:0.78rem;font-weight:600;">{urgency}</span>'
    )


def simulated_tag() -> str:
    return (
        '<span style="background:#4a3aa7;color:#fff;padding:1px 8px;'
        'border-radius:10px;font-size:0.7rem;font-weight:700;'
        'letter-spacing:0.05em;">SIMULATED</span>'
    )


def confidence_bar(confidence: float) -> str:
    """Progress bar with the human-review threshold marked as a tick."""
    pct = round(confidence * 100)
    threshold_pct = round(CONFIDENCE_THRESHOLD * 100)
    fill = "#008300" if confidence >= CONFIDENCE_THRESHOLD else "#d03b3b"
    return f"""
<div style="position:relative;height:14px;background:#e9e8e3;border-radius:7px;max-width:420px;">
  <div style="width:{pct}%;height:100%;background:{fill};border-radius:7px;"></div>
  <div style="position:absolute;left:{threshold_pct}%;top:-3px;width:2px;height:20px;background:#0b0b0b;"></div>
</div>
<div style="font-size:0.75rem;color:{INK_SOFT};max-width:420px;">
confidence {confidence:.2f} &nbsp;·&nbsp; marker = {CONFIDENCE_THRESHOLD} human-review threshold</div>
"""


@st.cache_resource(show_spinner=False)
def get_workflow():
    """Compiled LangGraph app (real LLM client, default cases.db)."""
    return build_workflow()


@st.cache_data
def load_samples() -> pd.DataFrame | None:
    try:
        frame = pd.read_csv(DATA_PATH)
        return frame if "request_text" in frame.columns else None
    except Exception:  # noqa: BLE001 -- a missing sample file must not break the app
        return None


def run_one(text: str) -> dict | None:
    """Invoke the graph for one request; friendly errors, never a stack trace."""
    try:
        return get_workflow().invoke({"raw_text": text})
    except LLMConfigError as exc:
        st.error(f"LLM not configured: {exc}", icon="🔑")
    except LLMUnavailableError as exc:
        st.error(
            "The LLM provider is currently unavailable (rate limit or outage). "
            "Please try again in a minute.",
            icon="⏳",
        )
        st.caption(str(exc))
    except Exception as exc:  # noqa: BLE001 -- UI boundary
        st.error("Unexpected error while processing this request; nothing was saved.")
        st.caption(f"{type(exc).__name__}: {exc}")
    return None


# --- result rendering (Process Request tab) -------------------------------------


def render_result(state: dict) -> None:
    classification = state["classification"]
    branch: Branch = state["branch"]
    spec = TAXONOMY[branch]

    st.success(
        f"Processed as case **{state['case_id']}** — status **{state['status'].value}**. "
        "Full history in the Case Log tab.",
        icon="✅",
    )

    # a) classification card ------------------------------------------------
    st.subheader("Classification")
    with st.container(border=True):
        m1, m2, m3, m4 = st.columns([1.5, 0.9, 1.1, 1.2])
        with m1:
            st.caption("Request type")
            st.markdown(f"#### {classification.request_type.value.replace('_', ' ').title()}")
        with m2:
            st.caption("Urgency")
            st.markdown(urgency_badge(classification.urgency.value), unsafe_allow_html=True)
        with m3:
            st.caption("Sub-topic")
            st.markdown(f"`{classification.sub_topic}`")
        with m4:
            st.caption("Routed to")
            st.markdown(f"**{spec.label}**")
        st.markdown(confidence_bar(classification.confidence), unsafe_allow_html=True)
        if classification.extracted_entities:
            st.markdown("**Extracted entities**")
            entities = pd.DataFrame(
                [(k.replace("_", " "), str(v)) for k, v in classification.extracted_entities.items()],
                columns=["Entity", "Value"],
            )
            st.dataframe(entities, hide_index=True, width="stretch")
        st.info(f"**Model reasoning:** {classification.reasoning}", icon="🧠")

    if branch is Branch.HUMAN_REVIEW:
        if classification.request_type is RequestType.ESCALATION:
            why = (
                "the classifier flagged an **explicit escalation signal** "
                "(legal threat, extreme frustration, or an unintelligible message)"
            )
        else:
            why = (
                f"classifier confidence **{classification.confidence:.2f}** is below the "
                f"**{CONFIDENCE_THRESHOLD}** auto-routing threshold"
            )
        st.warning(
            f"**Routed to Human Review** because {why}. Automation is paused: no reply was "
            "auto-sent, a supervisor was notified (simulated), and the case is queued with "
            "the classifier's full reasoning attached.",
            icon="⚠️",
        )

    # b) remediation timeline -------------------------------------------------
    st.subheader(f"Remediation timeline — {spec.label}")
    if branch is Branch.HUMAN_REVIEW:
        st.markdown(
            "⏸ **Automation PAUSED** — the steps below prepared material for the human "
            "reviewer; nothing was sent automatically."
        )
    steps = [a for a in state["actions"] if a.step_name != "Classification"]
    for index, action in enumerate(steps, 1):
        prefix = "⏸ " if action.step_name.startswith("Pause") else ""
        label = f"Step {index} · {prefix}{action.step_name}"
        if action.simulated:
            label += " · 🧪 simulated"
        with st.status(label, state="complete", expanded=False):
            st.write(action.detail)
            if action.simulated:
                st.markdown(simulated_tag(), unsafe_allow_html=True)
            st.caption(f"{action.timestamp.astimezone(IST):%d %b %Y %H:%M:%S} IST")

    # c) artifacts -------------------------------------------------------------
    st.subheader("Generated artifacts")
    deadlines: list[str] = []
    any_artifact = False
    for action in state["actions"]:
        artifact = action.artifact or {}
        if "draft_reply" in artifact:
            any_artifact = True
            st.markdown(
                f"**✉️ Draft response** ({artifact.get('tone', 'professional')} tone) — "
                "copy-ready, for ops review before sending:"
            )
            st.code(artifact["draft_reply"], language=None)
        elif "answer" in artifact:
            any_artifact = True
            source = (
                f"grounded in KB topic **{artifact.get('kb_topic')}**"
                if artifact.get("grounded")
                else "standard pointer (no KB entry matched)"
            )
            st.markdown(f"**✉️ Enquiry answer** — {source}:")
            st.code(artifact["answer"], language=None)
        elif "routing_notice" in artifact:
            any_artifact = True
            notice = artifact["routing_notice"]
            with st.container(border=True):
                st.markdown(
                    f"**📨 Routing notice → {notice['to']}** &nbsp;{simulated_tag()}",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"Case `{notice['case_id']}` · {notice['request_type']} · "
                    f"urgency {notice['urgency']} · sub-topic `{notice['sub_topic']}`"
                )
                st.caption(str(notice["summary"]))
        elif "supervisor_alert" in artifact:
            any_artifact = True
            alert = artifact["supervisor_alert"]
            with st.container(border=True):
                st.markdown(
                    f"**🚨 Supervisor alert → {alert['to']}** &nbsp;{simulated_tag()}",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"`{alert['case_id']}` classified {alert['request_type']} at "
                    f"confidence {alert['confidence']:.2f}"
                )
                st.caption(str(alert["reasoning"]))
        elif "review_queue_entry" in artifact:
            any_artifact = True
            entry = artifact["review_queue_entry"]
            with st.container(border=True):
                st.markdown("**🧑‍⚖️ Human-review queue entry**")
                st.markdown(
                    f"Classified as `{entry['classified_as']}` at confidence "
                    f"`{entry['confidence']:.2f}` — awaiting a person's decision."
                )
                st.caption(str(entry["classifier_reasoning"]))
        if artifact.get("sla_deadline"):
            deadline = datetime.fromisoformat(artifact["sla_deadline"])
            priority = ", priority case" if artifact.get("priority") else ""
            deadlines.append(
                f"⏱ **SLA deadline:** {deadline.astimezone(IST):%a %d %b %Y, %H:%M} IST "
                f"({artifact.get('sla_hours')}h{priority})"
            )
        if "follow_up_task" in artifact:
            task = artifact["follow_up_task"]
            due = datetime.fromisoformat(task["due_at"])
            deadlines.append(
                f"🔔 **Follow-up due:** {due.astimezone(IST):%a %d %b %Y, %H:%M} IST — "
                f"{task['action']} *(simulated)*"
            )
    if deadlines:
        with st.container(border=True):
            st.markdown("**Deadlines & follow-ups**")
            for line in deadlines:
                st.markdown(line)
    if not any_artifact and not deadlines:
        st.caption("No artifacts were generated for this branch.")


# --- chrome ------------------------------------------------------------------


with st.sidebar:
    st.markdown("## 📡 Telecom Request Processor")
    st.markdown(
        "An AI triage desk for the customer-operations inbox: every message is "
        "classified, actioned through the right remediation path, and fully logged. "
        "When the AI isn't sure, a person decides — it never acts on a guess."
    )
    st.markdown("#### Remediation branches")
    for spec in TAXONOMY.values():
        st.markdown(
            f"{urgency_badge(spec.default_urgency.value)}&nbsp; {spec.label}",
            unsafe_allow_html=True,
        )
    st.caption(f"Confidence below {CONFIDENCE_THRESHOLD} always routes to Human Review.")
    st.info(
        "Team routing, supervisor notifications and follow-up timers are **simulated**: "
        "rendered here and written to the audit log, not sent to real systems.",
        icon="🧪",
    )

st.title("📡 Customer Request Triage")
st.caption(
    "Paste a customer message — the AI classifies it, runs the right remediation "
    "steps, and logs everything for audit. When it isn't sure, a person decides."
)
if not HAS_KEY:
    st.error(
        "**No LLM API key configured — processing is disabled.** Copy `.env.example` to "
        "`.env`, set `GROQ_API_KEY` (free key at console.groq.com) and restart the app. "
        "The Case Log and Dashboard tabs still work without a key.",
        icon="🔑",
    )

# Server-side navigation instead of st.tabs: tabs render every panel in the
# client and desync when a rerun reshapes the element tree (results appear
# stacked across tabs). A keyed segmented control keeps the selection in
# session state and only the active section ever renders.
SECTIONS = ["⚡ Process Request", "📥 Batch", "🗂 Case Log", "📊 Dashboard"]
st.session_state.setdefault("nav", SECTIONS[0])
nav = (
    st.segmented_control("Section", SECTIONS, key="nav", label_visibility="collapsed")
    or SECTIONS[0]
)


# --- tab 1 · process request ----------------------------------------------------


if nav == "⚡ Process Request":
    # Plain HTML, deliberately not markdown headers: headers get Streamlit's
    # auto-anchor links and read as interactive; this is guidance, not controls.
    _steps = [
        ("1", "Paste a message", "or pick a ready-made example below"),
        ("2", "The AI triages it", "type, urgency and confidence — uncertain cases go to a person"),
        ("3", "Review what it did", "draft reply, hand-offs, deadlines — every step logged for audit"),
    ]
    _cells = "".join(
        f'<div style="flex:1;min-width:200px;">'
        f'<span style="background:{ACCENT};color:#fff;border-radius:50%;'
        f'display:inline-block;width:22px;height:22px;text-align:center;'
        f'font-size:0.8rem;font-weight:700;line-height:22px;">{number}</span>'
        f"&nbsp; <b>{title}</b><br/>"
        f'<span style="color:{INK_SOFT};font-size:0.85rem;">{subtitle}</span></div>'
        for number, title, subtitle in _steps
    )
    st.markdown(
        f'<div style="display:flex;gap:24px;flex-wrap:wrap;">{_cells}</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    samples = load_samples()
    EXAMPLES = {
        "Billing dispute — overcharged bill": 0,
        "Network complaint — area outage": 4,
        "Service request — plan upgrade": 8,
        "General enquiry — international roaming": 12,
        "Ambiguous — should land in Human Review": 14,
    }
    options = ["(write your own)"]
    if samples is not None and len(samples) > max(EXAMPLES.values()):
        options += list(EXAMPLES)
    choice = st.selectbox(
        "Try an example (one per branch)", options, index=1 if len(options) > 1 else 0
    )
    default_text = (
        "" if choice == "(write your own)" else str(samples["request_text"].iloc[EXAMPLES[choice]])
    )
    text = st.text_area(
        "Incoming customer message",
        value=default_text,
        height=140,
        placeholder="Paste or type the customer's message here…",
    )
    if st.button("Process request", type="primary", disabled=not HAS_KEY):
        if not text.strip():
            st.warning("Enter a message first.")
        else:
            with st.spinner("Classifying and executing remediation steps…"):
                final_state = run_one(text)
            if final_state:
                st.session_state["last_result"] = final_state

    # Rendered from session state so the result survives reruns and section
    # switches instead of vanishing when the button state resets.
    if "last_result" in st.session_state:
        render_result(st.session_state["last_result"])


# --- tab 2 · batch ---------------------------------------------------------------


if nav == "📥 Batch":
    st.caption(
        "Each request runs through the full pipeline, one at a time, with a "
        "2-second pause between AI calls (free-tier rate limits)."
    )
    if st.button("Load sample dataset (16 requests)"):
        if load_samples() is None:
            st.error("Sample dataset not found at data/sample_requests.csv.")
        else:
            st.session_state["batch_df"] = load_samples()
    with st.expander("…or upload your own CSV (needs a request_text column)"):
        uploaded = st.file_uploader("Upload CSV", type="csv")
        if uploaded is not None:
            try:
                frame = pd.read_csv(uploaded)
                if "request_text" not in frame.columns:
                    st.error("The CSV must contain a `request_text` column.")
                else:
                    st.session_state["batch_df"] = frame[["request_text"]].dropna()
            except Exception as exc:  # noqa: BLE001 -- bad upload must not crash the tab
                st.error(f"Could not read that CSV: {exc}")

    batch_df = st.session_state.get("batch_df")
    if batch_df is not None:
        st.dataframe(batch_df, width="stretch", height=220)
        total = len(batch_df)
        if st.button(f"Process {total} requests", type="primary", disabled=not HAS_KEY):
            progress = st.progress(0.0, text="Starting…")
            rows: list[dict] = []
            errors = 0
            workflow = get_workflow()
            for i, item in enumerate(batch_df["request_text"].astype(str)):
                progress.progress(i / total, text=f"Processing request {i + 1} of {total}…")
                try:
                    state = workflow.invoke({"raw_text": item})
                    result = state["classification"]
                    rows.append(
                        {
                            "case_id": state["case_id"],
                            "request": item[:60] + ("…" if len(item) > 60 else ""),
                            "type": result.request_type.value,
                            "urgency": f"{URGENCY_DOTS[result.urgency.value]} {result.urgency.value}",
                            "confidence": f"{result.confidence:.2f}",
                            "branch": state["branch"].value,
                            "status": state["status"].value,
                        }
                    )
                except Exception as exc:  # noqa: BLE001 -- record and continue the batch
                    errors += 1
                    rows.append(
                        {
                            "case_id": "—",
                            "request": item[:60],
                            "type": "ERROR",
                            "urgency": "—",
                            "confidence": "—",
                            "branch": "—",
                            "status": type(exc).__name__,
                        }
                    )
                if i < total - 1:
                    time.sleep(2)
            progress.empty()
            st.session_state["batch_results"] = (rows, errors)

    if "batch_results" in st.session_state:
        rows, errors = st.session_state["batch_results"]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        processed = [r for r in rows if r["type"] != "ERROR"]
        review = sum(r["branch"] == Branch.HUMAN_REVIEW.value for r in processed)
        auto = len(processed) - review
        summary = (
            f"**{len(rows)} processed** · **{auto}** auto-handled · "
            f"**{review}** sent to human review"
        )
        if errors:
            st.error(summary + f" · **{errors} error(s)**")
        else:
            st.success(summary)


# --- tab 3 · case log -------------------------------------------------------------


if nav == "🗂 Case Log":
    f1, f2, _ = st.columns([1, 1, 2])
    type_choice = f1.selectbox("Request type", ["All"] + [t.value for t in RequestType])
    status_choice = f2.selectbox("Status", ["All"] + [s.value for s in CaseStatus])
    cases = db.list_cases(
        status=None if status_choice == "All" else status_choice,
        request_type=None if type_choice == "All" else type_choice,
        limit=200,
    )
    if not cases:
        st.info("No cases yet — triage a request on the first tab or run the sample batch.")
    else:
        st.caption(
            f"{len(cases)} case(s), newest first — expand a case for its full audit trail."
        )
    for row in cases:
        cls = json.loads(row["classification"])
        created = datetime.fromisoformat(row["created_at"])
        dot = URGENCY_DOTS.get(row["urgency"], "⚪")
        title = (
            f"{dot} {row['case_id']} · {row['request_type']} → {row['branch']} · "
            f"{row['status']} · {created.astimezone(IST):%d %b %Y %H:%M} IST"
        )
        with st.expander(title):
            st.markdown("**Original message**")
            st.markdown("> " + str(row["raw_text"]).replace("\n", "\n> "))
            st.markdown(
                f"{urgency_badge(row['urgency'])} &nbsp;confidence "
                f"**{cls.get('confidence', 0.0):.2f}** · sub-topic "
                f"`{html.escape(str(cls.get('sub_topic', '')))}`",
                unsafe_allow_html=True,
            )
            if cls.get("reasoning"):
                st.caption(f"Classifier reasoning: {cls['reasoning']}")
            if row["sla_deadline"]:
                sla = datetime.fromisoformat(row["sla_deadline"])
                st.caption(f"SLA deadline: {sla.astimezone(IST):%d %b %Y %H:%M} IST")
            st.markdown("---")
            st.markdown("**Audit trail**")
            for action in db.list_actions(row["case_id"]):
                stamp = datetime.fromisoformat(action["timestamp"]).astimezone(IST)
                tag = f" {simulated_tag()}" if action["simulated"] else ""
                st.markdown(
                    f"<code>{stamp:%H:%M:%S}</code> · **{html.escape(action['step_name'])}**{tag}",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div style='margin:0 0 0.6rem 3.2rem;color:{INK_SOFT};"
                    f"font-size:0.9rem;'>{html.escape(action['detail'])}</div>",
                    unsafe_allow_html=True,
                )
                artifact = json.loads(action["artifact"]) if action["artifact"] else None
                if artifact and ("draft_reply" in artifact or "answer" in artifact):
                    st.code(artifact.get("draft_reply") or artifact.get("answer"), language=None)


# --- tab 4 · dashboard --------------------------------------------------------------


if nav == "📊 Dashboard":
    all_cases = db.list_cases(limit=1000)
    if not all_cases:
        st.info("No cases yet — process a request or run a batch first.")
    else:
        confidences = [json.loads(r["classification"]).get("confidence", 0.0) for r in all_cases]
        resolved = sum(r["status"] == CaseStatus.RESOLVED.value for r in all_cases)
        in_review = sum(r["status"] == CaseStatus.HUMAN_REVIEW.value for r in all_cases)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total cases", len(all_cases))
        m2.metric("Auto-resolved", resolved)
        m3.metric("In human review", in_review)
        m4.metric("Avg confidence", f"{sum(confidences) / len(confidences):.2f}")

        stats = db.dashboard_stats()
        left, right = st.columns([3, 2])

        with left:
            by_type = sorted(stats["by_type"].items(), key=lambda kv: kv[1], reverse=True)
            fig_type = go.Figure(
                go.Bar(
                    x=[k for k, _ in by_type],
                    y=[v for _, v in by_type],
                    marker_color=ACCENT,
                    text=[v for _, v in by_type],
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate="%{x}: %{y} cases<extra></extra>",
                )
            )
            fig_type.update_layout(
                title="Requests by classified type",
                barcornerradius=4,
                showlegend=False,
                height=340,
                margin=dict(l=10, r=10, t=48, b=10),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, rangemode="tozero"),
                xaxis=dict(showgrid=False),
                font=dict(color=INK_SOFT),
            )
            st.plotly_chart(fig_type, width="stretch")

        with right:
            labels = list(stats["by_status"])
            fig_status = go.Figure(
                go.Pie(
                    labels=labels,
                    values=[stats["by_status"][k] for k in labels],
                    hole=0.5,
                    marker=dict(
                        colors=[STATUS_COLORS.get(k, "#898781") for k in labels],
                        line=dict(color="#fcfcfb", width=2),
                    ),
                    textinfo="label+percent",
                    hovertemplate="%{label}: %{value} cases<extra></extra>",
                )
            )
            fig_status.update_layout(
                title="Case status mix",
                showlegend=False,
                height=340,
                margin=dict(l=10, r=10, t=48, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color=INK_SOFT),
            )
            st.plotly_chart(fig_status, width="stretch")

        present = [u for u in URGENCY_ORDER if u in stats["by_urgency"]]
        fig_urgency = go.Figure(
            go.Bar(
                x=present,
                y=[stats["by_urgency"][u] for u in present],
                marker_color=[URGENCY_COLORS[u] for u in present],
                text=[stats["by_urgency"][u] for u in present],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{x}: %{y} cases<extra></extra>",
            )
        )
        fig_urgency.update_layout(
            title="Urgency distribution (badge colours)",
            barcornerradius=4,
            showlegend=False,
            height=300,
            margin=dict(l=10, r=10, t=48, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, rangemode="tozero"),
            xaxis=dict(showgrid=False),
            font=dict(color=INK_SOFT),
        )
        st.plotly_chart(fig_urgency, width="stretch")
