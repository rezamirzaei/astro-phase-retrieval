"""001 — initial tables (users + jobs).

Revision ID: 001
"""

import sqlalchemy as sa
from alembic import op

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("email", sa.String(255), unique=True, nullable=False),
        sa.Column("username", sa.String(100), unique=True, nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("is_active", sa.Boolean, default=True),
        sa.Column("created_at", sa.DateTime(timezone=True)),
    )
    op.create_table(
        "jobs",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("algorithm", sa.String(50), nullable=False),
        sa.Column("status", sa.String(20), server_default="pending"),
        sa.Column("fits_filename", sa.String(500), nullable=False),
        sa.Column("config_json", sa.Text, server_default="{}"),
        sa.Column("strehl_ratio", sa.Float, nullable=True),
        sa.Column("rms_phase_rad", sa.Float, nullable=True),
        sa.Column("n_iterations", sa.Integer, nullable=True),
        sa.Column("elapsed_seconds", sa.Float, nullable=True),
        sa.Column("converged", sa.Boolean, nullable=True),
        sa.Column("cost_history_json", sa.Text, nullable=True),
        sa.Column("output_dir", sa.String(500), nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("jobs")
    op.drop_table("users")
