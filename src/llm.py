"""LLM client: classification (JSON mode) and short free-text generation.

Primary provider is Groq (llama-3.3-70b-versatile); if a Gemini key is set,
it acts as an automatic fallback when Groq fails. Contract with the rest of
the system: classify() NEVER raises because of a malformed model response --
it retries once with a repair prompt, then returns a zero-confidence
ESCALATION result, which the router turns into HUMAN_REVIEW. Only
configuration/outage problems raise, so the UI can explain them.
"""

from __future__ import annotations

import json
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .models import ClassificationResult, RequestType, Urgency
from .taxonomy import (
    CONFIDENCE_THRESHOLD,
    classifier_subtopic_options,
    classifier_type_definitions,
)

# Load the repo-root .env regardless of the caller's working directory.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

GROQ_MODEL = "openai/gpt-oss-120b"
GEMINI_MODEL = "gemini-2.0-flash"


class LLMConfigError(RuntimeError):
    """No usable LLM credentials; the UI shows setup instructions."""


class LLMUnavailableError(RuntimeError):
    """Every configured provider failed (outage, rate limit, bad key)."""


@lru_cache(maxsize=1)
def _groq_client() -> Any:
    from groq import Groq  # lazy: a broken install only hurts its own path

    return Groq(api_key=os.environ["GROQ_API_KEY"])


def _provider_order() -> list[str]:
    order: list[str] = []
    if os.getenv("GROQ_API_KEY"):
        order.append("groq")
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        order.append("gemini")
    if not order:
        raise LLMConfigError(
            "No LLM API key configured. Copy .env.example to .env and set "
            "GROQ_API_KEY (free key: https://console.groq.com)."
        )
    return order


def _call_provider(provider: str, system: str, user: str, *, json_mode: bool, temperature: float) -> str:
    if provider == "groq":
        kwargs: dict[str, Any] = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": 1024,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = _groq_client().chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    import google.generativeai as genai  # lazy fallback import

    genai.configure(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=system)
    config: dict[str, Any] = {"temperature": temperature}
    if json_mode:
        config["response_mime_type"] = "application/json"
    return model.generate_content(user, generation_config=config).text or ""


def _is_rate_limit(exc: Exception) -> bool:
    text = str(exc).lower()
    return getattr(exc, "status_code", None) == 429 or "429" in text or "rate limit" in text


# Two bounded waits per provider before giving up on a rate-limited call --
# free-tier limits during sequential batch runs are a named risk in the
# architecture doc, and a short wait is the correct remediation.
_RATE_LIMIT_BACKOFF_S = (6, 15)


def _complete(system: str, user: str, *, json_mode: bool, temperature: float) -> str:
    """One chat completion via the first configured provider that answers.

    Rate-limit responses get a bounded backoff-and-retry on the same
    provider; any other failure falls through to the next provider.
    """
    failures: list[str] = []
    for provider in _provider_order():
        for delay in (*_RATE_LIMIT_BACKOFF_S, None):
            try:
                return _call_provider(provider, system, user, json_mode=json_mode, temperature=temperature)
            except Exception as exc:  # noqa: BLE001 -- classified below
                if delay is not None and _is_rate_limit(exc):
                    time.sleep(delay)
                    continue
                failures.append(f"{provider}: {exc}")
                break
    raise LLMUnavailableError("All LLM providers failed -- " + " | ".join(failures))


def _classification_prompt() -> str:
    subtopic_options = classifier_subtopic_options()
    definitions = "\n".join(
        f"- {name}: {text}\n"
        f"  Valid sub_topic values for {name}: {', '.join(subtopic_options[name])}, other"
        for name, text in classifier_type_definitions().items()
    )
    return f"""You are the triage classifier for an Indian telecom operator's customer-operations inbox.

Assign the customer message exactly one request_type:
{definitions}

Urgency rubric: CRITICAL only for escalations/safety/legal; HIGH when money is disputed or service is down; MEDIUM for account changes wanted soon; LOW for informational questions.

extracted_entities: include only values literally present in the text (account or mobile numbers, rupee amounts, plan names, locations, dates, requested changes). Never invent a value.

sub_topic is required and must be one of the "Valid sub_topic values" listed for your chosen request_type; if none clearly applies, return exactly "other".

confidence is the probability that your request_type is correct -- calibrate it honestly. If the message is ambiguous, garbled, mixes several intents, or lacks the information needed to be sure, set confidence below {CONFIDENCE_THRESHOLD:.2f} and explain why in reasoning; low-confidence requests go to a human reviewer, which is the safe outcome.

Hard calibration rule: a mixed, garbled, or incoherent message -- especially one where no listed sub_topic clearly applies and you would return "other" -- MUST receive confidence below {CONFIDENCE_THRESHOLD:.2f}.

Reply with a single JSON object and nothing else:
{{
  "request_type": "BILLING_DISPUTE | NETWORK_COMPLAINT | SERVICE_REQUEST | GENERAL_ENQUIRY | ESCALATION",
  "urgency": "LOW | MEDIUM | HIGH | CRITICAL",
  "confidence": 0.0,
  "sub_topic": "one of the valid sub_topic values for the chosen request_type, or other",
  "extracted_entities": {{}},
  "reasoning": "1-2 sentences for the operations reviewer"
}}"""


def _parse_result(raw: str) -> ClassificationResult:
    """Pull a JSON object out of raw model text and validate it."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))
    return ClassificationResult.model_validate(payload)


# Deterministic calibration floor -- deliberately below CONFIDENCE_THRESHOLD
# so a capped result can never clear the auto-routing gate.
_SUBTOPIC_GUARDRAIL_CAP = 0.70


def _apply_consistency_guardrail(result: ClassificationResult) -> ClassificationResult:
    """Structural safety net applied after validation, never in the router.

    A model that reports high confidence while unable to name a sub-topic is
    miscalibrated by definition (observed live with llama-3.3-70b; see
    docs/calibration_note.md). If sub_topic is "other" on a non-ESCALATION
    type, confidence is capped below the routing threshold so the case lands
    with a human regardless of which model produced it.
    """
    if (
        result.sub_topic == "other"
        and result.request_type is not RequestType.ESCALATION
        and result.confidence > _SUBTOPIC_GUARDRAIL_CAP
    ):
        return result.model_copy(
            update={
                "confidence": _SUBTOPIC_GUARDRAIL_CAP,
                "reasoning": (
                    f"{result.reasoning} [confidence capped: no clear sub-topic identified]"
                ).strip(),
            }
        )
    return result


def classify(text: str) -> ClassificationResult:
    """Classify one raw customer message.

    Malformed model output triggers one repair round-trip; if that also
    fails, the request is handed to human review instead of crashing.
    """
    system = _classification_prompt()
    user = f'Customer message:\n"""\n{text.strip()}\n"""'
    raw = _complete(system, user, json_mode=True, temperature=0.1)
    for attempt in (1, 2):
        try:
            return _apply_consistency_guardrail(_parse_result(raw))
        except Exception as exc:  # noqa: BLE001 -- malformed JSON / enum values
            if attempt == 1:
                raw = _complete(
                    system,
                    f"{user}\n\nYour previous reply could not be parsed ({exc}). "
                    "Reply again with ONLY the corrected JSON object.",
                    json_mode=True,
                    temperature=0.0,
                )
    return ClassificationResult(
        request_type=RequestType.ESCALATION,
        urgency=Urgency.CRITICAL,
        confidence=0.0,
        sub_topic="classification_failure",
        reasoning=(
            "Classifier output could not be parsed after a repair attempt; "
            "routing to human review."
        ),
    )


def generate(prompt: str, system: str | None = None, temperature: float = 0.4) -> str:
    """Short free-text generation used by remediation steps for drafts."""
    default_system = (
        "You write brief, warm, professional replies for an Indian telecom "
        "operator's customer-care team. Plain text only, no markdown, no "
        "placeholders like [Name] unless unavoidable, maximum 120 words."
    )
    return _complete(
        default_system if system is None else system,
        prompt,
        json_mode=False,
        temperature=temperature,
    ).strip()
