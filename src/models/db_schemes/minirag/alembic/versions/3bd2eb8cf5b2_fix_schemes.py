"""fix schemes

Revision ID: 3bd2eb8cf5b2
Revises: adc365b3c119
Create Date: 2025-04-18 20:06:14.102500

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3bd2eb8cf5b2'
down_revision: Union[str, None] = 'adc365b3c119'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # FKs already match models in d77b5366226c (fresh DB bootstrap).
    pass


def downgrade() -> None:
    pass
