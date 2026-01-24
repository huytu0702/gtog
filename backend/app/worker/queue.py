"""Redis queue configuration."""

from functools import lru_cache
from uuid import UUID

from redis import Redis
from rq import Queue

from app.config import settings


@lru_cache(maxsize=1)
def get_redis_connection() -> Redis:
    """Get cached Redis connection."""
    return Redis.from_url(settings.redis_url)


@lru_cache(maxsize=1)
def get_queue() -> Queue:
    """Get the indexing job queue."""
    return Queue("graphrag-indexing", connection=get_redis_connection())


def enqueue_indexing_job(collection_id: UUID, index_run_id: UUID) -> str:
    """
    Enqueue an indexing job.

    Args:
        collection_id: Collection to index
        index_run_id: Index run record ID

    Returns:
        Job ID
    """
    from app.worker.tasks import run_indexing_task

    queue = get_queue()
    job = queue.enqueue(
        run_indexing_task,
        str(collection_id),
        str(index_run_id),
        job_timeout="2h",
        result_ttl=86400,  # 24 hours
    )
    return job.id
