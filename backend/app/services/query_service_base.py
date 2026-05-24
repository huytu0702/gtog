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
            if key_lower == "relationships" and "source" in df.columns and "target" in df.columns:
                source = str(row.get("source", ""))
                target = str(row.get("target", ""))
                name = f"{source} → {target}" if source and target else source or target
            else:
                name = str(row.get(name_col, "")) if name_col in df.columns else short_id
            raw_desc = row.get(desc_col, "") if desc_col and desc_col in df.columns else ""
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
        return {str(key): _serialize_json_safe_context(item) for key, item in normalized.items()}
    if isinstance(value, dict):
        return {
            str(key): _serialize_json_safe_context(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_serialize_json_safe_context(item) for item in value]
    return str(value)


def _match_entity_record_by_name(
    entities: pd.DataFrame,
    raw_name: str,
) -> dict[str, str] | None:
    if entities.empty:
        return None

    normalized_name = raw_name.strip()
    if not normalized_name:
        return None

    name_column = _preferred_entity_name_column(entities)
    candidate_columns = [
        col for col in (name_column, "title", "name", "entity") if col in entities.columns
    ]
    normalized_target = normalized_name.casefold()

    for _, row in entities.iterrows():
        for column in candidate_columns:
            value = _non_empty_text(row.get(column))
            if value and value.casefold() == normalized_target:
                description = _non_empty_text(row.get("description"))
                if not description:
                    description = _non_empty_text(row.get("text"))
                if not description:
                    return None
                return {"name": value, "description": description}
    return None


def _build_tog_serialized_context(
    context_data: dict[str, Any],
    *,
    entities: pd.DataFrame,
    relationships: pd.DataFrame,
) -> tuple[dict[str, dict[str, dict[str, str]]] | None, set[str]]:
    paths = context_data.get("exploration_paths", [])
    if not isinstance(paths, Iterable) or isinstance(paths, (str, bytes)):
        return None, set()

    entity_paths: dict[str, list[str]] = {}
    rel_lookup: dict[str, dict[str, str]] = {}
    known_entity_names: set[str] = set()
    saw_edge_path = False

    for path in paths:
        path_text = _non_empty_text(path)
        if not path_text:
            continue
        segments = [segment.strip() for segment in path_text.split(" | ") if segment.strip()]
        for segment in segments:
            match = re.match(r"^(.+?)\s+--\[(.+?)\]-->\s+(.+)$", segment)
            if match is None:
                match = re.match(r"^(.+?)\s+<--\[(.+?)\]--\s+(.+)$", segment)
                if match is not None:
                    src = match.group(3).strip()
                    rel = match.group(2).strip()
                    tgt = match.group(1).strip()
                else:
                    continue
            else:
                src = match.group(1).strip()
                rel = match.group(2).strip()
                tgt = match.group(3).strip()

            saw_edge_path = True
            entity_paths.setdefault(src, []).append(segment)
            entity_paths.setdefault(tgt, []).append(segment)
            known_entity_names.add(src)
            known_entity_names.add(tgt)
            rel_lookup[rel] = {"name": rel, "description": ""}

    if saw_edge_path:
        entity_lookup = {
            name: {
                "name": name,
                "description": " | ".join(dict.fromkeys(path_list)),
            }
            for name, path_list in entity_paths.items()
        }
        serialized: dict[str, dict[str, dict[str, str]]] = {}
        if entity_lookup:
            serialized["Entities"] = entity_lookup
        if rel_lookup:
            serialized["Relationships"] = rel_lookup
        return serialized or None, known_entity_names

    entity_lookup: dict[str, dict[str, str]] = {}
    for path in paths:
        entity_name = _non_empty_text(path)
        if not entity_name:
            continue
        entity_record = _match_entity_record_by_name(entities, entity_name)
        if entity_record is None:
            continue
        entity_lookup[entity_record["name"]] = entity_record
        known_entity_names.add(entity_record["name"])

    if not entity_lookup:
        return None, set()

    return {"Entities": entity_lookup}, known_entity_names


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
        matched = [
            name_map[n.lower()] if n.lower() in name_map else n
            for n in raw_names
            if n.strip()
        ]
        if matched:
            return f"[Data: Entities ({', '.join(matched)})]"
        return match.group(0)

    return re.sub(r"\[Data:\s*([^\]]+)\]", _rewrite, text)


# ---------------------------------------------------------------------------
# Query log no-ops (cloud mode)
# ---------------------------------------------------------------------------


def _attach_query_log(collection_id: str) -> None:
    """No-op query logger in cloud mode (avoid local file writes)."""
    return


def _detach_query_log(handler: Any) -> None:
    """No-op query logger in cloud mode."""
    return
