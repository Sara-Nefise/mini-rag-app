"""Add new models (bootstrap for empty DB)

Revision ID: d77b5366226c
Revises:
Create Date: 2025-04-17 08:41:42.969698

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'd77b5366226c'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Legacy vector table from older deployments; ignore if missing
    op.execute(sa.text("DROP TABLE IF EXISTS collection_1024_1 CASCADE"))

    op.create_table(
        "projects",
        sa.Column("project_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("project_uuid", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("project_id"),
        sa.UniqueConstraint("project_uuid"),
    )

    op.create_table(
        "users",
        sa.Column("user_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user__uuid", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("firebase_uid", sa.String(length=128), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("user_id"),
        sa.UniqueConstraint("email"),
        sa.UniqueConstraint("firebase_uid"),
        sa.UniqueConstraint("user__uuid"),
    )
    op.create_index("ix_user_email", "users", ["email"], unique=False)
    op.create_index("ix_user_firebase_uid", "users", ["firebase_uid"], unique=False)

    op.create_table(
        "assets",
        sa.Column("asset_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("asset_uuid", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("asset_type", sa.String(), nullable=False),
        sa.Column("asset_name", sa.String(), nullable=False),
        sa.Column("asset_size", sa.Integer(), nullable=False),
        sa.Column("asset_config", postgresql.JSONB(), nullable=True),
        sa.Column("asset_project_id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["asset_project_id"], ["projects.project_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("asset_id"),
        sa.UniqueConstraint("asset_uuid"),
    )
    op.create_index("ix_asset_project_id", "assets", ["asset_project_id"], unique=False)
    op.create_index("ix_asset_type", "assets", ["asset_type"], unique=False)

    op.create_table(
        "chunks",
        sa.Column("chunk_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("chunk_uuid", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_text", sa.String(), nullable=False),
        sa.Column("chunk_metadata", postgresql.JSONB(), nullable=True),
        sa.Column("chunk_order", sa.Integer(), nullable=False),
        sa.Column("chunk_project_id", sa.Integer(), nullable=False),
        sa.Column("chunk_asset_id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["chunk_project_id"], ["projects.project_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["chunk_asset_id"], ["assets.asset_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("chunk_id"),
        sa.UniqueConstraint("chunk_uuid"),
    )
    op.create_index("ix_chunk_project_id", "chunks", ["chunk_project_id"], unique=False)
    op.create_index("ix_chunk_asset_id", "chunks", ["chunk_asset_id"], unique=False)

    op.create_table(
        "chats",
        sa.Column("chat_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("chat_uuid", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.project_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("chat_id"),
        sa.UniqueConstraint("chat_uuid"),
    )
    op.create_index("ix_chat_project_id", "chats", ["project_id"], unique=False)
    op.create_index("ix_chat_user_id", "chats", ["user_id"], unique=False)

    op.create_table(
        "messages",
        sa.Column("message_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("message_uuid", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chat_id", sa.Integer(), nullable=False),
        sa.Column("is_user", sa.Boolean(), nullable=False),
        sa.Column("content", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["chat_id"], ["chats.chat_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("message_id"),
        sa.UniqueConstraint("message_uuid"),
    )
    op.create_index("ix_message_chat_id", "messages", ["chat_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_message_chat_id", table_name="messages")
    op.drop_table("messages")
    op.drop_index("ix_chat_user_id", table_name="chats")
    op.drop_index("ix_chat_project_id", table_name="chats")
    op.drop_table("chats")
    op.drop_index("ix_chunk_asset_id", table_name="chunks")
    op.drop_index("ix_chunk_project_id", table_name="chunks")
    op.drop_table("chunks")
    op.drop_index("ix_asset_type", table_name="assets")
    op.drop_index("ix_asset_project_id", table_name="assets")
    op.drop_table("assets")
    op.drop_index("ix_user_firebase_uid", table_name="users")
    op.drop_index("ix_user_email", table_name="users")
    op.drop_table("users")
    op.drop_table("projects")
