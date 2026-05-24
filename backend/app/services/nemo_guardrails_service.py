"""AI guardrails service backed by deterministic checks and optional NeMo rails."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

from ..config import settings

logger = logging.getLogger(__name__)

GuardrailAction = Literal["allow", "block", "rewrite", "redact", "log_only"]
SAFE_GUARDRAIL_RESPONSE = "Mình không thể hỗ trợ yêu cầu đó."

_JAILBREAK_PATTERN = re.compile(
    r"\b(ignore|disregard|override)\b.{0,80}\b(instructions?|rules?|policy|guardrails?)\b|"
    r"\b(system prompt|hidden context|developer message|reveal.*(?:prompt|context|config)|"
    r"print.*(?:prompt|context|config)|bypass guardrails?|jailbreak|"
    r"retrieved documents? as instructions?)\b",
    re.IGNORECASE | re.DOTALL,
)
_SECRET_PATTERN = re.compile(
    r"\b(api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret|credential|"
    r"connection[_-]?string|private[_-]?key|bearer)\b\s*[:=]\s*\S+|"
    r"\b(sk-[A-Za-z0-9_-]{8,}|AIza[0-9A-Za-z_-]{20,}|tvly-[A-Za-z0-9_-]{8,})\b",
    re.IGNORECASE,
)
_SECRET_REQUEST_PATTERN = re.compile(
    r"\b(reveal|show|print|dump|expose|leak)\b.{0,80}"
    r"\b(api[_-]?key|token|password|secret|credential|config|env|\.env)\b",
    re.IGNORECASE | re.DOTALL,
)
_OUTPUT_LEAK_PATTERN = re.compile(
    r"\b(system prompt|developer message|hidden context|api[_-]?key|access[_-]?token|"
    r"connection string|private key)\b",
    re.IGNORECASE,
)
_STOPWORDS = {
    "about",
    "are",
    "can",
    "does",
    "for",
    "from",
    "how",
    "latest",
    "please",
    "the",
    "this",
    "that",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


@dataclass(frozen=True)
class GuardrailDecision:
    allowed: bool
    action: GuardrailAction
    reason: str
    safe_response: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class NemoGuardrailsService:
    def _enabled(self) -> bool:
        return bool(settings.ai_guardrails_enabled)

    def _mode(self) -> str:
        return str(settings.ai_guardrails_mode).strip().lower()

    def _fail_mode(self) -> str:
        return str(settings.ai_guardrails_fail_mode).strip().lower()

    def _config_path(self) -> Path:
        configured = Path(str(settings.ai_guardrails_config_path))
        if configured.is_absolute():
            return configured
        repo_root = Path(__file__).resolve().parents[3]
        backend_root = Path(__file__).resolve().parents[2]
        repo_relative = repo_root / configured
        if repo_relative.exists():
            return repo_relative
        return backend_root / configured

    @cached_property
    def _rails(self) -> Any | None:
        try:
            from nemoguardrails import LLMRails, RailsConfig
        except ImportError:
            logger.warning("nemoguardrails package is not installed")
            return None

        config_path = self._config_path()
        if not config_path.exists():
            logger.warning(
                "NeMo guardrails config path does not exist: %s", config_path
            )
            return None

        config = RailsConfig.from_path(str(config_path))
        return LLMRails(config)

    def _allow(
        self, reason: str, metadata: dict[str, Any] | None = None
    ) -> GuardrailDecision:
        return GuardrailDecision(
            allowed=True,
            action="allow",
            reason=reason,
            metadata=metadata or {},
        )

    def _deny(
        self, reason: str, metadata: dict[str, Any] | None = None
    ) -> GuardrailDecision:
        decision_metadata = metadata or {}
        if self._mode() == "shadow":
            decision = GuardrailDecision(
                allowed=True,
                action="log_only",
                reason=reason,
                metadata={**decision_metadata, "would_block": True},
            )
        else:
            decision = GuardrailDecision(
                allowed=False,
                action="block",
                reason=reason,
                safe_response=SAFE_GUARDRAIL_RESPONSE,
                metadata=decision_metadata,
            )
        self._log_decision(decision)
        return decision

    def _redact(self, reason: str, safe_response: str) -> GuardrailDecision:
        decision = GuardrailDecision(
            allowed=True,
            action="redact",
            reason=reason,
            safe_response=safe_response,
        )
        self._log_decision(decision)
        return decision

    def _failure_decision(self, reason: str) -> GuardrailDecision:
        if self._fail_mode() == "closed":
            return GuardrailDecision(
                allowed=False,
                action="block",
                reason=reason,
                safe_response=SAFE_GUARDRAIL_RESPONSE,
            )
        return self._allow(reason)

    def _log_decision(self, decision: GuardrailDecision) -> None:
        if not bool(settings.ai_guardrails_log_decisions):
            return
        logger.info(
            "AI guardrail decision: allowed=%s action=%s reason=%s metadata=%s",
            decision.allowed,
            decision.action,
            decision.reason,
            decision.metadata,
        )

    async def _run_nemo(self, text: str, kind: str) -> GuardrailDecision:
        if self._rails is None:
            return self._failure_decision("nemo_unavailable")

        rail = "output" if kind == "output" else "input"
        messages = self._build_nemo_messages(text, rail)
        response = await asyncio.wait_for(
            self._rails.generate_async(
                messages=messages,
                options={
                    "rails": [rail],
                    "output_vars": True,
                    "log": {"llm_calls": True},
                },
            ),
            timeout=float(settings.ai_guardrails_timeout_seconds),
        )
        allowed = self._extract_nemo_allowed(response)
        if allowed is False:
            return self._deny("nemo_blocked", {"kind": kind})
        if allowed is True:
            return self._allow("nemo_allowed", {"kind": kind})

        content = self._extract_nemo_content(response)
        verdict = self._parse_nemo_verdict(content)
        if verdict == "yes":
            return self._deny("nemo_blocked", {"kind": kind})
        if verdict == "no":
            return self._allow("nemo_allowed", {"kind": kind})
        return self._failure_decision("nemo_malformed_verdict")

    def _build_nemo_messages(self, text: str, rail: str) -> list[dict[str, str]]:
        if rail == "output":
            return [
                {"role": "user", "content": "Check assistant response."},
                {"role": "assistant", "content": text},
            ]
        return [{"role": "user", "content": text}]

    def _extract_nemo_allowed(self, response: Any) -> bool | None:
        output_data = getattr(response, "output_data", None)
        if isinstance(output_data, dict) and isinstance(output_data.get("allowed"), bool):
            return output_data["allowed"]
        return None

    def _extract_nemo_content(self, response: Any) -> str:
        if isinstance(response, dict):
            return str(response.get("content", ""))
        log = getattr(response, "log", None)
        llm_calls = getattr(log, "llm_calls", None)
        if llm_calls:
            completion = getattr(llm_calls[-1], "completion", "")
            return str(completion)
        response_content = getattr(response, "response", None)
        if isinstance(response_content, list) and response_content:
            last_message = response_content[-1]
            if isinstance(last_message, dict):
                return str(last_message.get("content", ""))
        return str(response)

    def _parse_nemo_verdict(self, content: str) -> Literal["yes", "no"] | None:
        first_token = re.match(r"\s*(yes|no)\b", content, re.IGNORECASE)
        if first_token:
            return first_token.group(1).lower()
        if SAFE_GUARDRAIL_RESPONSE in content or "can't assist" in content.lower():
            return "yes"
        return None

    async def _run_optional_nemo(
        self,
        text: str,
        kind: str,
        context: dict[str, Any] | None,
    ) -> GuardrailDecision:
        if context is not None and context.get("skip_nemo") is True:
            return self._allow("nemo_skipped", {"kind": kind})
        if not text.strip():
            return self._allow("empty_text", {"kind": kind})
        try:
            return await self._run_nemo(text, kind)
        except Exception as exc:
            logger.warning("NeMo guardrail %s check failed: %s", kind, exc)
            return self._failure_decision(f"nemo_{kind}_failed")

    async def check_input(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> GuardrailDecision:
        if not self._enabled():
            return self._allow("guardrails_disabled")
        if _SECRET_PATTERN.search(query) or _SECRET_REQUEST_PATTERN.search(query):
            return self._deny("sensitive_input")
        if _JAILBREAK_PATTERN.search(query):
            return self._deny("prompt_injection")
        return await self._run_optional_nemo(query, "input", context)

    async def check_rewrite(
        self,
        original_query: str,
        rewritten_query: str,
        context: dict[str, Any] | None = None,
    ) -> GuardrailDecision:
        if not self._enabled():
            return self._allow("guardrails_disabled")
        if (
            not rewritten_query.strip()
            or original_query.strip() == rewritten_query.strip()
        ):
            return self._allow("rewrite_unchanged")

        original_tokens = self._meaningful_tokens(original_query)
        rewritten_tokens = self._meaningful_tokens(rewritten_query)
        overlap = len(original_tokens & rewritten_tokens) / max(1, len(original_tokens))
        similarity = SequenceMatcher(
            None, original_query.lower(), rewritten_query.lower()
        ).ratio()
        if original_tokens and overlap < 0.25 and similarity < 0.55:
            return self._deny(
                "rewrite_intent_changed",
                {"overlap": round(overlap, 3), "similarity": round(similarity, 3)},
            )
        return await self._run_optional_nemo(
            f"Original query: {original_query}\nRewritten query: {rewritten_query}",
            "rewrite",
            context,
        )

    async def check_output(
        self,
        answer: str,
        context: dict[str, Any] | None = None,
    ) -> GuardrailDecision:
        if not self._enabled():
            return self._allow("guardrails_disabled")
        if _SECRET_PATTERN.search(answer) or _OUTPUT_LEAK_PATTERN.search(answer):
            return self._deny("unsafe_output")
        return await self._run_optional_nemo(answer, "output", context)

    async def check_web_query(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> GuardrailDecision:
        if not self._enabled():
            return self._allow("guardrails_disabled")
        if bool(settings.ai_guardrails_block_web_on_sensitive_query) and (
            _SECRET_PATTERN.search(query) or _SECRET_REQUEST_PATTERN.search(query)
        ):
            return self._deny("sensitive_web_query")
        return await self._run_optional_nemo(query, "web", context)

    async def sanitize_summary(
        self,
        summary: str,
        context: dict[str, Any] | None = None,
    ) -> GuardrailDecision:
        if not self._enabled():
            return self._allow("guardrails_disabled")
        sanitized = self._sanitize_text(summary)
        if sanitized != summary:
            return self._redact("summary_sanitized", sanitized)
        return await self._run_optional_nemo(summary, "summary", context)

    def _sanitize_text(self, text: str) -> str:
        sanitized = _JAILBREAK_PATTERN.sub("", text)
        sanitized = _SECRET_PATTERN.sub("[REDACTED]", sanitized)
        sanitized = re.sub(r"\s{2,}", " ", sanitized).strip()
        return sanitized

    def _meaningful_tokens(self, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[A-Za-z0-9_]+", text.lower())
            if len(token) > 2 and token not in _STOPWORDS
        }


nemo_guardrails_service = NemoGuardrailsService()
