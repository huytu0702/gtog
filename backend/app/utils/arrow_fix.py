"""Monkey-patch to fix ArrowStringArray conversion issues in GraphRAG."""

import logging

import pandas as pd
from graphrag.index.workflows import create_final_text_units

logger = logging.getLogger(__name__)


def _patched_entities(df: pd.DataFrame) -> pd.DataFrame:
    """Patched version that converts ArrowStringArray to Python lists."""
    selected = df.loc[:, ["id", "text_unit_ids"]]
    unrolled = selected.explode(["text_unit_ids"]).reset_index(drop=True)

    result = (
        unrolled.groupby("text_unit_ids", sort=False)
        .agg(entity_ids=("id", "unique"))
        .reset_index()
        .rename(columns={"text_unit_ids": "id"})
    )

    # Convert ArrowStringArray to Python lists
    result["entity_ids"] = result["entity_ids"].apply(
        lambda x: x.tolist() if hasattr(x, "tolist") else list(x)
    )

    return result


def _patched_relationships(df: pd.DataFrame) -> pd.DataFrame:
    """Patched version that converts ArrowStringArray to Python lists."""
    selected = df.loc[:, ["id", "text_unit_ids"]]
    unrolled = selected.explode(["text_unit_ids"]).reset_index(drop=True)

    result = (
        unrolled.groupby("text_unit_ids", sort=False)
        .agg(relationship_ids=("id", "unique"))
        .reset_index()
        .rename(columns={"text_unit_ids": "id"})
    )

    # Convert ArrowStringArray to Python lists
    result["relationship_ids"] = result["relationship_ids"].apply(
        lambda x: x.tolist() if hasattr(x, "tolist") else list(x)
    )

    return result


def _patched_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """Patched version that converts ArrowStringArray to Python lists."""
    selected = df.loc[:, ["id", "text_unit_id"]]

    result = (
        selected.groupby("text_unit_id", sort=False)
        .agg(covariate_ids=("id", "unique"))
        .reset_index()
        .rename(columns={"text_unit_id": "id"})
    )

    # Convert ArrowStringArray to Python lists
    result["covariate_ids"] = result["covariate_ids"].apply(
        lambda x: x.tolist() if hasattr(x, "tolist") else list(x)
    )

    return result


def apply_arrow_fix():
    """Apply monkey-patches to fix ArrowStringArray conversion issues."""
    logger.info("Applying ArrowStringArray conversion fix...")

    # Store original functions
    create_final_text_units._original_entities = create_final_text_units._entities
    create_final_text_units._original_relationships = (
        create_final_text_units._relationships
    )
    create_final_text_units._original_covariates = create_final_text_units._covariates

    # Apply patches
    create_final_text_units._entities = _patched_entities
    create_final_text_units._relationships = _patched_relationships
    create_final_text_units._covariates = _patched_covariates

    logger.info("ArrowStringArray conversion fix applied successfully")


def remove_arrow_fix():
    """Remove monkey-patches and restore original functions."""
    logger.info("Removing ArrowStringArray conversion fix...")

    if hasattr(create_final_text_units, "_original_entities"):
        create_final_text_units._entities = create_final_text_units._original_entities
        delattr(create_final_text_units, "_original_entities")

    if hasattr(create_final_text_units, "_original_relationships"):
        create_final_text_units._relationships = (
            create_final_text_units._original_relationships
        )
        delattr(create_final_text_units, "_original_relationships")

    if hasattr(create_final_text_units, "_original_covariates"):
        create_final_text_units._covariates = create_final_text_units._original_covariates
        delattr(create_final_text_units, "_original_covariates")

    logger.info("ArrowStringArray conversion fix removed")
