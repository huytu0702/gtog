"""Services package."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_MODULE_BY_EXPORT = {
    "ConversationService": ".conversation_service",
    "conversation_service": ".conversation_service",
    "IndexingService": ".indexing_service",
    "indexing_service": ".indexing_service",
    "InsufficiencyDecision": ".insufficiency_judge",
    "InsufficiencyJudge": ".insufficiency_judge",
    "insufficiency_judge": ".insufficiency_judge",
    "GuardrailDecision": ".nemo_guardrails_service",
    "NemoGuardrailsService": ".nemo_guardrails_service",
    "nemo_guardrails_service": ".nemo_guardrails_service",
    "QueryService": ".query_service",
    "query_service": ".query_service",
    "QueueService": ".queue_service",
    "queue_service": ".queue_service",
    "RouteDecision": ".router_agent",
    "RouterAgent": ".router_agent",
    "router_agent": ".router_agent",
    "StorageService": ".storage_service",
    "storage_service": ".storage_service",
    "SummarizationService": ".summarization_service",
    "summarization_service": ".summarization_service",
    "WebSearchResult": ".web_search",
    "WebSearchService": ".web_search",
    "web_search_service": ".web_search",
}

_SINGLETON_EXPORTS = {
    "conversation_service",
    "indexing_service",
    "insufficiency_judge",
    "nemo_guardrails_service",
    "query_service",
    "queue_service",
    "router_agent",
    "storage_service",
    "summarization_service",
    "web_search_service",
}


class _LazyExportProxy:
    def __init__(self, export_name: str) -> None:
        self._export_name = export_name

    def _resolve(self) -> Any:
        module_name = _MODULE_BY_EXPORT[self._export_name]
        module = import_module(module_name, __name__)
        return getattr(module, self._export_name)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._resolve(), name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._resolve()(*args, **kwargs)


for _export_name in _SINGLETON_EXPORTS:
    globals()[_export_name] = _LazyExportProxy(_export_name)

__all__ = list(_MODULE_BY_EXPORT.keys())


def __getattr__(name: str) -> Any:
    module_name = _MODULE_BY_EXPORT.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    return getattr(module, name)
