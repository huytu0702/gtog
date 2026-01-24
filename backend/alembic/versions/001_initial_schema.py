"""Initial schema with all GraphRAG tables.

Revision ID: 001
Revises:
Create Date: 2026-01-24

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Collections table
    op.create_table(
        "collections",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(100), unique=True, nullable=False, index=True),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # Index runs table
    op.create_table(
        "index_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("status", sa.String(20), nullable=False, default="queued"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error", sa.Text, nullable=True),
    )

    # Documents table
    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("human_readable_id", sa.Integer, nullable=True, index=True),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("index_run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("index_runs.id", ondelete="SET NULL"), nullable=True, index=True),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("text", sa.Text, nullable=True),
        sa.Column("doc_metadata", postgresql.JSONB, nullable=True),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("content_type", sa.String(100), nullable=True),
        sa.Column("bytes_content", sa.LargeBinary, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # Entities table
    op.create_table(
        "entities",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("human_readable_id", sa.Integer, nullable=True, index=True),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("index_run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("index_runs.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("title", sa.String(500), nullable=False, index=True),
        sa.Column("type", sa.String(100), nullable=True),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("frequency", sa.Integer, nullable=True),
        sa.Column("degree", sa.Integer, nullable=True),
        sa.Column("x", sa.Float, nullable=True),
        sa.Column("y", sa.Float, nullable=True),
    )

    # Relationships table
    op.create_table(
        "relationships",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("human_readable_id", sa.Integer, nullable=True, index=True),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("index_run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("index_runs.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("source", sa.String(500), nullable=False, index=True),
        sa.Column("target", sa.String(500), nullable=False, index=True),
        sa.Column("source_entity_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("entities.id", ondelete="SET NULL"), nullable=True),
        sa.Column("target_entity_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("entities.id", ondelete="SET NULL"), nullable=True),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("weight", sa.Float, nullable=True),
        sa.Column("combined_degree", sa.Integer, nullable=True),
    )

    # Text units table
    op.create_table(
        "text_units",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("human_readable_id", sa.Integer, nullable=True, index=True),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("index_run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("index_runs.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("n_tokens", sa.Integer, nullable=True),
    )

    # Communities table
    op.create_table(
        "communities",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("human_readable_id", sa.Integer, nullable=True, index=True),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("index_run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("index_runs.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("community", sa.Integer, nullable=False, index=True),
        sa.Column("parent", sa.Integer, nullable=True),
        sa.Column("level", sa.Integer, nullable=False, index=True),
        sa.Column("title", sa.String(500), nullable=True),
        sa.Column("period", sa.String(50), nullable=True),
        sa.Column("size", sa.Integer, nullable=True),
    )

    # Community reports table
    op.create_table(
        "community_reports",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("human_readable_id", sa.Integer, nullable=True, index=True),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("index_run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("index_runs.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("community", sa.Integer, nullable=False, index=True),
        sa.Column("parent", sa.Integer, nullable=True),
        sa.Column("level", sa.Integer, nullable=False),
        sa.Column("title", sa.String(500), nullable=True),
        sa.Column("summary", sa.Text, nullable=True),
        sa.Column("full_content", sa.Text, nullable=True),
        sa.Column("rank", sa.Float, nullable=True),
        sa.Column("rating_explanation", sa.Text, nullable=True),
        sa.Column("findings", postgresql.JSONB, nullable=True),
        sa.Column("full_content_json", postgresql.JSONB, nullable=True),
        sa.Column("period", sa.String(50), nullable=True),
        sa.Column("size", sa.Integer, nullable=True),
    )

    # Covariates table
    op.create_table(
        "covariates",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("human_readable_id", sa.Integer, nullable=True, index=True),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("index_run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("index_runs.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("covariate_type", sa.String(50), nullable=False),
        sa.Column("type", sa.String(100), nullable=True),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("subject_id", sa.String(500), nullable=True),
        sa.Column("object_id", sa.String(500), nullable=True),
        sa.Column("status", sa.String(20), nullable=True),
        sa.Column("start_date", sa.String(50), nullable=True),
        sa.Column("end_date", sa.String(50), nullable=True),
        sa.Column("source_text", sa.Text, nullable=True),
        sa.Column("text_unit_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("text_units.id", ondelete="SET NULL"), nullable=True),
    )

    # Embeddings table with pgvector
    op.create_table(
        "embeddings",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("index_run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("index_runs.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("embedding_type", sa.String(30), nullable=False, index=True),
        sa.Column("ref_id", postgresql.UUID(as_uuid=True), nullable=False, index=True),
    )
    # Add vector column with pgvector
    op.execute("ALTER TABLE embeddings ADD COLUMN vector vector(1536)")

    # Create HNSW index for fast similarity search
    op.execute("""
        CREATE INDEX embeddings_vector_idx ON embeddings
        USING hnsw (vector vector_cosine_ops)
    """)

    # Association tables
    op.create_table(
        "document_text_units",
        sa.Column("document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("text_unit_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("text_units.id", ondelete="CASCADE"), primary_key=True),
    )

    op.create_table(
        "text_unit_entities",
        sa.Column("text_unit_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("text_units.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("entity_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("entities.id", ondelete="CASCADE"), primary_key=True),
    )

    op.create_table(
        "text_unit_relationships",
        sa.Column("text_unit_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("text_units.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("relationship_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("relationships.id", ondelete="CASCADE"), primary_key=True),
    )

    op.create_table(
        "community_entities",
        sa.Column("community_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("communities.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("entity_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("entities.id", ondelete="CASCADE"), primary_key=True),
    )

    op.create_table(
        "community_relationships",
        sa.Column("community_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("communities.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("relationship_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("relationships.id", ondelete="CASCADE"), primary_key=True),
    )

    op.create_table(
        "community_text_units",
        sa.Column("community_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("communities.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("text_unit_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("text_units.id", ondelete="CASCADE"), primary_key=True),
    )

    op.create_table(
        "community_hierarchy",
        sa.Column("parent_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("communities.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("child_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("communities.id", ondelete="CASCADE"), primary_key=True),
    )


def downgrade() -> None:
    # Drop association tables
    op.drop_table("community_hierarchy")
    op.drop_table("community_text_units")
    op.drop_table("community_relationships")
    op.drop_table("community_entities")
    op.drop_table("text_unit_relationships")
    op.drop_table("text_unit_entities")
    op.drop_table("document_text_units")

    # Drop main tables
    op.drop_table("embeddings")
    op.drop_table("covariates")
    op.drop_table("community_reports")
    op.drop_table("communities")
    op.drop_table("text_units")
    op.drop_table("relationships")
    op.drop_table("entities")
    op.drop_table("documents")
    op.drop_table("index_runs")
    op.drop_table("collections")

    # Drop extension
    op.execute("DROP EXTENSION IF EXISTS vector")
