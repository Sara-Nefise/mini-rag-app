"""Add new models

Revision ID: 3cbcf900c866
Revises: d86ea6eb2211
Create Date: 2025-04-18 20:08:54.415026

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3cbcf900c866'
down_revision: Union[str, None] = 'd86ea6eb2211'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Foreign keys already match models in d77b5366226c (fresh DB bootstrap).
    pass


def downgrade() -> None:
    pass
