"""LLM-based sufficiency judge for GraphRAG responses."""

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from litellm import acompletion
from litellm.exceptions import RateLimitError

from ..config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InsufficiencyDecision:
    """Decision on whether GraphRAG answer needs web fallback."""

    is_sufficient: bool
    needs_web_fallback: bool
    confidence: float
    reason: str
    missing_information: list[str]
    risk: str


class InsufficiencyJudge:
    """Judge whether GraphRAG response is sufficient to answer user query."""

    def __init__(self) -> None:
        self.prompt_template = self._load_prompt()

    def _provider_from_model(self, model_name: str) -> str:
        if "/" in model_name:
            return model_name.split("/", 1)[0].strip().lower()
        return settings.query_chat_model_provider

    def _temperature_for_model(self, model_name: str) -> float:
        normalized = model_name.split("/", 1)[-1].strip().lower()
        if normalized.startswith("gpt-5"):
            return 1.0
        return settings.insufficiency_judge_temperature

    def _load_prompt(self) -> str:
        prompt_path = (
            Path(__file__).parent.parent.parent
            / "prompts"
            / "insufficiency_judge_prompt.txt"
        )
        if prompt_path.exists():
            return prompt_path.read_text()
        return self._default_prompt()

    def _default_prompt(self) -> str:
        return (
            "Return JSON with is_sufficient, needs_web_fallback, confidence, reason, "
            "missing_information, risk. Query: {original_query}. "
            "GraphRAG answer: {graphrag_response}"
        )

    async def _call_llm(self, prompt: str):
        max_retries = 2
        model_name = settings.insufficiency_judge_model or settings.query_chat_model_litellm
        provider = self._provider_from_model(model_name)
        api_key = settings.api_key_for_provider(provider)
        for attempt in range(max_retries + 1):
            try:
                return await asyncio.wait_for(
                    acompletion(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self._temperature_for_model(model_name),
                        max_tokens=settings.insufficiency_judge_max_tokens,
                        api_key=api_key,
                        response_format={"type": "json_object"},
                    ),
                    timeout=settings.insufficiency_judge_timeout_seconds,
                )
            except RateLimitError:
                if attempt == max_retries:
                    raise
                await asyncio.sleep(1.0 * (2**attempt))
            except asyncio.TimeoutError:
                raise
            except Exception as e:
                if "response_format" in str(e):
                    return await asyncio.wait_for(
                        acompletion(
                            model=model_name,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=self._temperature_for_model(model_name),
                            max_tokens=settings.insufficiency_judge_max_tokens,
                            api_key=api_key,
                        ),
                        timeout=settings.insufficiency_judge_timeout_seconds,
                    )
                raise

    def _coerce_bool(self, value, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized == "true":
                return True
            if normalized == "false":
                return False
        return default

    def _parse_decision(self, content: str) -> InsufficiencyDecision | None:
        try:
            data = json.loads(content)
            is_sufficient = self._coerce_bool(data.get("is_sufficient"), True)
            needs_web_fallback = self._coerce_bool(
                data.get("needs_web_fallback"),
                not is_sufficient,
            )
            confidence = float(data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
            reason = str(data.get("reason", ""))[:240]
            missing_information_raw = data.get("missing_information", [])
            if isinstance(missing_information_raw, list):
                missing_information = [str(x) for x in missing_information_raw][:8]
            else:
                missing_information = []
            risk = str(data.get("risk", "medium")).lower()
            if risk not in ("low", "medium", "high"):
                risk = "medium"

            if is_sufficient == needs_web_fallback:
                return None

            return InsufficiencyDecision(
                is_sufficient=is_sufficient,
                needs_web_fallback=needs_web_fallback,
                confidence=confidence,
                reason=reason,
                missing_information=missing_information,
                risk=risk,
            )
        except Exception:
            return None

    async def judge(
        self,
        *,
        original_query: str,
        search_query: str,
        method_used: str,
        graphrag_response: str,
        context_metadata: str,
    ) -> InsufficiencyDecision | None:
        if not settings.insufficiency_judge_enabled:
            return None

        prompt = self.prompt_template.format(
            original_query=original_query,
            search_query=search_query,
            method_used=method_used,
            graphrag_response=graphrag_response[: settings.insufficiency_judge_max_response_chars],
            context_metadata=context_metadata,
        )

        try:
            response = await self._call_llm(prompt)
            content = response.choices[0].message.content or ""
            decision = self._parse_decision(content)
            if decision is None:
                logger.warning("Insufficiency judge returned invalid decision")
            return decision
        except Exception as e:
            logger.warning("Insufficiency judge failed: %s", e)
            return None


insufficiency_judge = InsufficiencyJudge()
