"""Mini telecom knowledge base backing General Enquiry answers.

One entry per enquiry sub-topic in the taxonomy (alignment enforced by the
smoke tests). The KB_ANSWER remediation step retrieves exactly one entry and
the LLM is instructed to answer *only* from it, so enquiry replies stay
grounded in approved copy rather than model memory. Figures are illustrative
for the prototype's fictional operator.
"""

from __future__ import annotations

import re
from typing import NamedTuple


class _Entry(NamedTuple):
    keywords: frozenset[str]
    content: str


_KB: dict[str, _Entry] = {
    "plans": _Entry(
        keywords=frozenset({
            "plan", "plans", "recharge", "tariff", "prepaid", "postpaid",
            "upgrade", "pack", "validity", "price", "ott", "unlimited",
        }),
        content=(
            "Popular prepaid plans: ₹299/28 days (1.5 GB/day + unlimited "
            "voice), ₹499/56 days (2 GB/day) and ₹799/84 days (2 GB/day + OTT "
            "bundle). Postpaid starts at ₹399/month with data rollover and "
            "family add-ons. Plans can be changed any time in the self-care "
            "app under Plans; a prepaid change applies from the next recharge "
            "and a postpaid change from the next bill cycle, with no fee."
        ),
    ),
    "roaming": _Entry(
        keywords=frozenset({
            "roaming", "international", "abroad", "travel", "overseas",
            "foreign", "flight",
        }),
        content=(
            "International roaming packs: ₹649/day (unlimited data + 100 "
            "voice minutes) or ₹2,999/10 days, valid in 38 countries. "
            "Activate in the app under Services → International Roaming at "
            "least 24 hours before departure. Domestic roaming anywhere in "
            "India is free on every plan."
        ),
    ),
    "payments": _Entry(
        keywords=frozenset({
            "payment", "payments", "pay", "bill", "upi", "autopay", "refund",
            "due", "late", "fee", "invoice", "receipt",
        }),
        content=(
            "Bills can be paid by UPI, debit/credit card, net banking, or "
            "auto-pay in the self-care app. Postpaid bills generate on the "
            "5th of each month and are due by the 20th; a ₹100 late fee "
            "applies after the due date. Refunds and reversals reach the "
            "original payment method within 5–7 working days."
        ),
    ),
    "coverage": _Entry(
        keywords=frozenset({
            "coverage", "signal", "network", "5g", "4g", "tower", "indoor",
            "wifi", "speed", "area", "city", "village",
        }),
        content=(
            "4G covers all of India; 5G is live in 140+ cities including "
            "Mumbai, Delhi, Bengaluru, Hyderabad, Pune and Nagpur at no extra "
            "cost on plans of ₹299 and above. Check street-level availability "
            "in the app under Network → Coverage Map. For weak indoor signal, "
            "enable Wi-Fi Calling in phone settings -- it works over any "
            "broadband connection."
        ),
    ),
}


def topics() -> list[str]:
    """All sub-topics the knowledge base can answer."""
    return list(_KB)


def retrieve(query: str) -> tuple[str, str] | None:
    """Best-matching ``(topic, content)`` for a sub-topic or question.

    Plain keyword overlap -- deliberately simple and deterministic; the
    brief rules out RAG/vector search, and four topics do not need it.
    Returns None when nothing matches, which the caller must treat as
    "no grounded answer available".
    """
    words = set(re.findall(r"[a-z0-9]+", query.lower()))
    if not words:
        return None
    best_topic: str | None = None
    best_score = 0
    for topic, entry in _KB.items():
        score = len(words & entry.keywords)
        if topic in words:
            score += 2  # an exact sub-topic mention wins ties
        if score > best_score:
            best_topic, best_score = topic, score
    if best_topic is None:
        return None
    return best_topic, _KB[best_topic].content
