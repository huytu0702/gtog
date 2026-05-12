# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""LiteLLM completion/embedding logging wrapper."""

import logging
import time
from typing import TYPE_CHECKING, Any, Literal

from graphrag.language_model.providers.litellm.request_wrappers.usage_logging import (
    UsageLogContext,
    calculate_cost_usd,
    extract_token_usage,
    write_usage_record,
)
from graphrag.language_model.providers.litellm.types import (
    AsyncLitellmRequestFunc,
    LitellmRequestFunc,
)

if TYPE_CHECKING:
    from graphrag.config.models.language_model_config import LanguageModelConfig

logger = logging.getLogger(__name__)


def with_logging(
    *,
    sync_fn: LitellmRequestFunc,
    async_fn: AsyncLitellmRequestFunc,
    model_config: "LanguageModelConfig",
    request_type: Literal["chat", "embedding"],
) -> tuple[LitellmRequestFunc, AsyncLitellmRequestFunc]:
    """
    Wrap the synchronous and asynchronous request functions with usage logging.

    Args
    ----
        sync_fn: The synchronous chat/embedding request function to wrap.
        async_fn: The asynchronous chat/embedding request function to wrap.
        model_config: The configuration for the language model.
        request_type: The type of request being made.

    Returns
    -------
        A tuple containing the wrapped synchronous and asynchronous chat/embedding request functions.
    """

    def _wrapped_with_logging(**kwargs: Any) -> Any:
        start_time = time.perf_counter()
        try:
            response = sync_fn(**kwargs)
        except Exception as e:
            _write_error_record(model_config, request_type, start_time, e)
            logger.warning(
                "with_logging: Request failed with exception type %s",
                type(e).__name__,
            )
            raise

        _write_success_record(model_config, request_type, start_time, response)
        return response

    async def _wrapped_with_logging_async(
        **kwargs: Any,
    ) -> Any:
        start_time = time.perf_counter()
        try:
            response = await async_fn(**kwargs)
        except Exception as e:
            _write_error_record(model_config, request_type, start_time, e)
            logger.warning(
                "with_logging: Async request failed with exception type %s",
                type(e).__name__,
            )
            raise

        _write_success_record(model_config, request_type, start_time, response)
        return response

    return (_wrapped_with_logging, _wrapped_with_logging_async)


def _write_success_record(
    model_config: "LanguageModelConfig",
    request_type: Literal["chat", "embedding"],
    start_time: float,
    response: Any,
) -> None:
    usage = extract_token_usage(response, request_type)
    model = model_config.deployment_name or model_config.model
    model_provider = model_config.model_provider or ""
    write_usage_record(
        UsageLogContext(
            model_provider=model_provider,
            model=model,
            request_type=request_type,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            prompt_tokens=usage.prompt_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            success=True,
            cache_hit=getattr(response, "_graphrag_cache_hit", None),
            cost_usd=calculate_cost_usd(
                model_provider,
                model,
                usage.prompt_tokens,
                usage.output_tokens,
            ),
        )
    )


def _write_error_record(
    model_config: "LanguageModelConfig",
    request_type: Literal["chat", "embedding"],
    start_time: float,
    error: Exception,
) -> None:
    model = model_config.deployment_name or model_config.model
    model_provider = model_config.model_provider or ""
    write_usage_record(
        UsageLogContext(
            model_provider=model_provider,
            model=model,
            request_type=request_type,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            prompt_tokens=None,
            output_tokens=None,
            total_tokens=None,
            success=False,
            cache_hit=None,
            cost_usd=None,
            error_type=type(error).__name__,
            error_message=None,
        )
    )
