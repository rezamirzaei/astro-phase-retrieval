"""003 — add token_version column to users table.

Supports instant JWT revocation: when a user changes their password,
``token_version`` is incremented and all previously issued JWTs (which
carry the old version) are rejected.

Revision ID: 003
Revises:     002
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("token_version", sa.Integer, nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("users", "token_version")
