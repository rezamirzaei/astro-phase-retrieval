"""002 — add crystallography_jobs table.

Revision ID: 002
Revises:     001
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "crystallography_jobs",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("algorithm", sa.String(50), nullable=False),
        sa.Column("status", sa.String(20), server_default="pending"),
        sa.Column("cif_filename", sa.String(500), nullable=False),
        sa.Column("cod_id", sa.String(50), server_default=""),
        sa.Column("formula", sa.String(200), server_default=""),
        sa.Column("config_json", sa.Text, server_default="{}"),
        sa.Column("r_factor", sa.Float, nullable=True),
        sa.Column("n_iterations", sa.Integer, nullable=True),
        sa.Column("elapsed_seconds", sa.Float, nullable=True),
        sa.Column("converged", sa.Boolean, nullable=True),
        sa.Column("cost_history_json", sa.Text, nullable=True),
        sa.Column("output_dir", sa.String(500), nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    # Index for fast per-user history lookup
    op.create_index(
        "ix_crystallography_jobs_user_id",
        "crystallography_jobs",
        ["user_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_crystallography_jobs_user_id", "crystallography_jobs")
    op.drop_table("crystallography_jobs")

