"""Shared utilities, type aliases, and data-normalisation helpers for query services."""

import json
import logging
import re
from numbers import Integral, Real
from typing import Any

from collections.abc import Iterable

import pandas as pd
from pandas import DataFrame

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column name mappings: what to use as the "name" and "description" per dataset
# ---------------------------------------------------------------------------
_CONTEXT_COLS: dict[str, tuple[str, str]] = {
    "entities": ("entity", "description"),
    "relationships": ("source", "description"),
    "reports": ("title", "summary"),
    "sources": ("text", "text"),
    "covariates": ("subject_id", "covariate_type"),
}


# ---------------------------------------------------------------------------
# Primitive value helpers
# ---------------------------------------------------------------------------


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        is_na = pd.isna(value)
        if isinstance(is_na, bool):
            return is_na
    except Exception:
        pass
    return False


def _non_empty_text(value: Any) -> str:
    if _is_missing_value(value):
        return ""
    return str(value).strip()


def _coerce_findings(value: Any) -> list[dict[str, Any]]:
    if _is_missing_value(value):
        return []

    parsed = value
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except json.JSONDecodeError:
            return []
    elif hasattr(parsed, "tolist") and not isinstance(parsed, (bytes, bytearray)):
        try:
            parsed = parsed.tolist()
        except Exception:
            return []

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []

    return [item for item in parsed if isinstance(item, dict)]


def _preferred_entity_name_column(entities: pd.DataFrame) -> str:
    for col in ("title", "name", "entity", "id"):
        if col in entities.columns:
            return col
    return entities.columns[0] if len(entities.columns) > 0 else "id"


# ---------------------------------------------------------------------------
# Context serialisation
# ---------------------------------------------------------------------------


def _serialize_context_records(
    context_data: str | list[DataFrame] | dict[str, DataFrame] | dict | None,
) -> dict[str, dict[str, dict]] | None:
    """Convert context_records DataFrames into a JSON-serializable lookup dict.

    Returns: {dataset_name: {short_id: {name, description}}}

    Non-dict inputs (e.g. plain strings or lists returned by some search
    methods) are ignored and ``None`` is returned.
    """
    if not context_data or not isinstance(context_data, dict):
        return None
    result: dict[str, dict[str, dict]] = {}
    for key, df in context_data.items():
        if df is None or df.empty:
            continue
        key_lower = key.lower()
        name_col, desc_col = _CONTEXT_COLS.get(key_lower, ("id", ""))
        lookup: dict[str, dict] = {}
        for _, row in df.iterrows():
            short_id = str(row.get("id", ""))
            if (
                key_lower == "relationships"
                and "source" in df.columns
                and "target" in df.columns
            ):
                source = str(row.get("source", ""))
                target = str(row.get("target", ""))
                name = f"{source} → {target}" if source and target else source or target
            else:
                name = (
                    str(row.get(name_col, "")) if name_col in df.columns else short_id
                )
            raw_desc = (
                row.get(desc_col, "") if desc_col and desc_col in df.columns else ""
            )
            desc = _non_empty_text(raw_desc)
            if key_lower == "reports" and not desc:
                desc = _non_empty_text(row.get("full_content"))
            lookup[short_id] = {"name": name, "description": desc}
        result[key] = lookup
    return result or None


def _serialize_json_safe_context(value: Any) -> Any:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        numeric = float(value)
        if not pd.notna(numeric) or numeric in (float("inf"), float("-inf")):
            return None
        return numeric
    if isinstance(value, DataFrame):
        normalized = value.astype(object).where(pd.notna(value), None)
        return normalized.to_dict(orient="records")
    if isinstance(value, pd.Series):
        normalized = value.astype(object).where(pd.notna(value), None)
        return {
            str(key): _serialize_json_safe_context(item)
            for key, item in normalized.items()
        }
    if isinstance(value, dict):
        return {
            str(key): _serialize_json_safe_context(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_serialize_json_safe_context(item) for item in value]
    return str(value)


def _serialize_tog_context_data(context_data: Any) -> dict[str, Any] | None:
    if not isinstance(context_data, dict) or not context_data:
        return None

    serialized = _serialize_json_safe_context(context_data)
    return serialized if isinstance(serialized, dict) else None


def _extract_tog_entity_names_from_context(
    context_data: dict[str, Any] | None,
) -> set[str]:
    if not context_data:
        return set()

    paths = context_data.get("exploration_paths", [])
    if not isinstance(paths, Iterable) or isinstance(paths, (str, bytes)):
        return set()

    entity_names: set[str] = set()
    for path in paths:
        if not isinstance(path, str):
            continue
        for segment in [part.strip() for part in path.split(" | ") if part.strip()]:
            match = re.match(r"^(.+?)\s+--\[(.+?)\]-->\s+(.+)$", segment)
            if match is None:
                match = re.match(r"^(.+?)\s+<--\[(.+?)\]--\s+(.+)$", segment)
            if match is None:
                entity_names.add(segment)
                continue
            entity_names.add(match.group(1).strip())
            entity_names.add(match.group(3).strip())

    return {name for name in entity_names if name}


def _build_tog_relationships_context(
    *,
    context_data: dict[str, Any] | None,
    relationships: pd.DataFrame,
) -> dict[str, dict[str, str]]:
    if not context_data:
        return {}

    paths = context_data.get("exploration_paths", [])
    if not isinstance(paths, Iterable) or isinstance(paths, (str, bytes)):
        return {}

    relationship_lookup: dict[str, dict[str, str]] = {}
    for path in paths:
        if not isinstance(path, str):
            continue
        for segment in [part.strip() for part in path.split(" | ") if part.strip()]:
            match = re.match(r"^(.+?)\s+--\[(.+?)\]-->\s+(.+)$", segment)
            if match is not None:
                source = match.group(1).strip()
                relation = match.group(2).strip()
                target = match.group(3).strip()
            else:
                match = re.match(r"^(.+?)\s+<--\[(.+?)\]--\s+(.+)$", segment)
                if match is None:
                    continue
                source = match.group(3).strip()
                relation = match.group(2).strip()
                target = match.group(1).strip()

            if not source or not target or not relation:
                continue
            relationship_key = f"{source}|{relation}|{target}"
            relationship_lookup[relationship_key] = {
                "name": f"{source} → {target}",
                "description": _tog_relationship_description(
                    relationships=relationships,
                    source=source,
                    target=target,
                    relation=relation,
                ),
            }

    return relationship_lookup


def _tog_relationship_description(
    *,
    relationships: pd.DataFrame,
    source: str,
    target: str,
    relation: str,
) -> str:
    if (
        relationships.empty
        or "source" not in relationships.columns
        or "target" not in relationships.columns
    ):
        return relation

    source_key = source.casefold()
    target_key = target.casefold()
    relation_key = relation.casefold()
    candidate_descriptions: list[str] = []
    for _, row in relationships.iterrows():
        row_source = _non_empty_text(row.get("source"))
        row_target = _non_empty_text(row.get("target"))
        if row_source.casefold() != source_key or row_target.casefold() != target_key:
            continue
        description = _non_empty_text(row.get("description"))
        if not description:
            continue
        if description.casefold() == relation_key:
            return description
        candidate_descriptions.append(description)

    if len(candidate_descriptions) == 1:
        return candidate_descriptions[0]
    return relation


def _coerce_text_unit_ids(value: Any) -> list[str]:
    if _is_missing_value(value):
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [stripped]
        return _coerce_text_unit_ids(parsed)
    if hasattr(value, "tolist") and not isinstance(value, (bytes, bytearray)):
        try:
            return _coerce_text_unit_ids(value.tolist())
        except Exception:
            return []
    if isinstance(value, Iterable) and not isinstance(
        value, (str, bytes, bytearray, dict)
    ):
        return [_non_empty_text(item) for item in value if _non_empty_text(item)]
    return [_non_empty_text(value)]


def _build_tog_sources_context(
    *,
    entity_names: set[str],
    entities: pd.DataFrame,
    text_units: pd.DataFrame,
) -> dict[str, dict[str, str]]:
    if not entity_names or entities.empty or text_units.empty:
        return {}
    if "text_unit_ids" not in entities.columns or "id" not in text_units.columns:
        return {}

    entity_name_lookup = {name.casefold() for name in entity_names}
    name_column = _preferred_entity_name_column(entities)
    candidate_columns = [
        col
        for col in (name_column, "title", "name", "entity")
        if col in entities.columns
    ]
    text_unit_ids: list[str] = []

    for _, row in entities.iterrows():
        row_names = {
            _non_empty_text(row.get(column)).casefold() for column in candidate_columns
        }
        if not row_names.intersection(entity_name_lookup):
            continue
        text_unit_ids.extend(_coerce_text_unit_ids(row.get("text_unit_ids")))

    wanted_ids = set(dict.fromkeys(text_unit_ids))
    sources: dict[str, dict[str, str]] = {}
    for _, row in text_units.iterrows():
        text_unit_id = _non_empty_text(row.get("id"))
        if text_unit_id not in wanted_ids:
            continue
        text = _non_empty_text(row.get("text")) or _non_empty_text(row.get("content"))
        sources[text_unit_id] = {
            "name": text_unit_id,
            "description": text,
        }

    return sources


# ---------------------------------------------------------------------------
# Community report normalisation
# ---------------------------------------------------------------------------


def _community_report_payload(row: pd.Series) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    raw_payload = row.get("full_content_json")
    if isinstance(raw_payload, str) and raw_payload.strip():
        try:
            parsed = json.loads(raw_payload)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            payload = parsed
    elif isinstance(raw_payload, dict):
        payload = raw_payload
    return payload


def _render_community_report_full_content(row: pd.Series) -> str:
    payload = _community_report_payload(row)
    title = (
        _non_empty_text(payload.get("title") or row.get("title")) or "Community Report"
    )
    summary = _non_empty_text(payload.get("summary") or row.get("summary"))
    findings = _coerce_findings(payload.get("findings") or row.get("findings"))
    rating = payload.get("rating")
    if _is_missing_value(rating):
        rating = row.get("rank")
    rating_text = _non_empty_text(rating)
    rating_explanation = _non_empty_text(
        payload.get("rating_explanation") or row.get("rating_explanation")
    )

    sections = [f"# {title}"]
    if summary:
        sections.append(summary)

    for finding in findings:
        finding_title = _non_empty_text(finding.get("summary")) or "Finding"
        explanation = _non_empty_text(finding.get("explanation"))
        if explanation:
            sections.append(f"## {finding_title}\n\n{explanation}")

    if rating_text or rating_explanation:
        rating_body = rating_text
        if rating_explanation:
            rating_body = (
                f"{rating_body}\n\n{rating_explanation}"
                if rating_body
                else rating_explanation
            )
        sections.append(f"## Impact Severity Rating\n\n{rating_body}")

    rendered = "\n\n".join(section for section in sections if section)
    return rendered.strip() or title


def _normalize_community_reports_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    has_full_content = "full_content" in frame.columns
    if has_full_content:
        missing_mask = frame["full_content"].apply(_is_missing_value)
        if not missing_mask.any():
            return frame
    else:
        missing_mask = None

    normalized = frame.copy()
    if not has_full_content:
        normalized["full_content"] = ""
        missing_mask = normalized["full_content"].apply(_is_missing_value)

    if missing_mask is not None and missing_mask.any():
        normalized.loc[missing_mask, "full_content"] = normalized.loc[
            missing_mask
        ].apply(
            _render_community_report_full_content,
            axis=1,
        )

    return normalized


# ---------------------------------------------------------------------------
# ToG citation normalisation
# ---------------------------------------------------------------------------


def _normalize_tog_citations(text: str, entity_names: set[str]) -> str:
    """Normalize ToG LLM citations to the frontend-expected [Data: Entities (...)] format.

    The LLM often emits [Data: NAME1, NAME2] instead of [Data: Entities (NAME1, NAME2)].
    This detects bare [Data: ...] blocks that contain known entity names and rewrites them.
    """
    name_map = {n.lower(): n for n in entity_names}

    def _rewrite(match: re.Match) -> str:
        inner = match.group(1).strip()
        if re.match(
            r"^(Entities|Relationships|Sources|Reports)\s*\(", inner, re.IGNORECASE
        ):
            return match.group(0)
        raw_names = [n.strip() for n in inner.split(",")]
        matched = [name_map.get(n.lower(), "") for n in raw_names if n.strip()]
        if matched and all(matched):
            return f"[Data: Entities ({', '.join(matched)})]"
        return match.group(0)

    return re.sub(r"\[Data:\s*([^\]]+)\]", _rewrite, text)


_CITATION_BLOCK_RE = re.compile(r"\[Data:\s*([^\]]+)\]")
_CITATION_ENTRY_RE = re.compile(r"([A-Za-z][A-Za-z\s]*?)\s*\(([^)]+)\)")
_CITATION_DATASET_ALIASES: dict[str, tuple[str, ...]] = {
    "sources": ("sources", "text units", "source chunks"),
    "reports": ("reports",),
    "entities": ("entities",),
    "relationships": ("relationships",),
    "claims": ("claims",),
}
_CITATION_DATASET_LABELS: dict[str, str] = {
    "sources": "Sources",
    "reports": "Reports",
    "entities": "Entities",
    "relationships": "Relationships",
    "claims": "Claims",
}


def _collect_context_citation_ids(
    context_data: dict[str, Any] | None,
    dataset: str,
) -> list[str]:
    if not context_data:
        return []

    aliases = _CITATION_DATASET_ALIASES.get(dataset, (dataset,))
    dataset_key = next(
        (
            key
            for key in context_data
            if key.casefold() in {alias.casefold() for alias in aliases}
        ),
        None,
    )
    if dataset_key is None:
        return []

    dataset_value = context_data.get(dataset_key)
    if isinstance(dataset_value, dict):
        return [identifier for identifier in dataset_value if _non_empty_text(identifier)]
    if isinstance(dataset_value, list):
        identifiers: list[str] = []
        for item in dataset_value:
            if not isinstance(item, dict):
                continue
            identifier = _non_empty_text(item.get("id"))
            if identifier:
                identifiers.append(identifier)
        return identifiers
    return []



def _format_citation_identifier_list(values: list[str]) -> str:
    unique_values = [value for value in dict.fromkeys(values) if value]
    if len(unique_values) <= 5:
        return ", ".join(unique_values)
    return ", ".join([*unique_values[:5], "+more"])



def _parse_structured_citation_entries(inner: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    for match in _CITATION_ENTRY_RE.finditer(inner):
        label = _non_empty_text(match.group(1))
        values = _non_empty_text(match.group(2))
        if not label or not values:
            continue
        entries[label] = values
    return entries



def _normalize_context_citations(
    text: str,
    *,
    context_data: dict[str, Any] | None,
    dataset_order: list[str],
    entity_names: set[str] | None = None,
) -> str:
    if not text or not context_data:
        return text

    del dataset_order
    name_map = {name.casefold(): name for name in entity_names or set()}
    if not name_map:
        return text

    def _rewrite(match: re.Match) -> str:
        inner = _non_empty_text(match.group(1))
        entries = _parse_structured_citation_entries(inner)
        if entries:
            return match.group(0)

        raw_names = [_non_empty_text(value) for value in inner.split(",")]
        matched_names = [
            name_map.get(raw_name.casefold(), "") for raw_name in raw_names if raw_name
        ]
        if matched_names and all(matched_names):
            return f"[Data: Entities ({', '.join(matched_names)})]"
        return match.group(0)

    return _CITATION_BLOCK_RE.sub(_rewrite, text)


# ---------------------------------------------------------------------------
# Query log no-ops (cloud mode)
# ---------------------------------------------------------------------------


def _attach_query_log(collection_id: str) -> None:
    """No-op query logger in cloud mode (avoid local file writes)."""
    return


def _detach_query_log(handler: Any) -> None:
    """No-op query logger in cloud mode."""
    return
