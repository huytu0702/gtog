# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Storage functions for the GraphRAG run module."""

import asyncio
import logging
from io import BytesIO

import pandas as pd

from graphrag.storage.pipeline_storage import PipelineStorage

logger = logging.getLogger(__name__)


async def load_table_from_storage(name: str, storage: PipelineStorage) -> pd.DataFrame:
    """Load a parquet from the storage instance."""
    filename = f"{name}.parquet"
    if not await storage.has(filename):
        msg = f"Could not find {filename} in storage!"
        raise ValueError(msg)

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            logger.info("reading table from storage: %s", filename)
            payload = await storage.get(filename, as_bytes=True)
            if not payload:
                raise ValueError(f"Storage returned empty payload for {filename}")
            return pd.read_parquet(BytesIO(payload))
        except Exception:
            if attempt < max_attempts:
                logger.warning(
                    "retrying table read from storage: %s (attempt %d/%d)",
                    filename,
                    attempt,
                    max_attempts,
                )
                await asyncio.sleep(1)
                continue

            logger.exception("error loading table from storage: %s", filename)
            raise


async def write_table_to_storage(
    table: pd.DataFrame, name: str, storage: PipelineStorage
) -> None:
    """Write a table to storage."""
    await storage.set(f"{name}.parquet", table.to_parquet())


async def delete_table_from_storage(name: str, storage: PipelineStorage) -> None:
    """Delete a table to storage."""
    await storage.delete(f"{name}.parquet")


async def storage_has_table(name: str, storage: PipelineStorage) -> bool:
    """Check if a table exists in storage."""
    return await storage.has(f"{name}.parquet")
