"""chat table edited

Revision ID: adc365b3c119
Revises: d77b5366226c
Create Date: 2025-04-18 16:35:12.072378

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'adc365b3c119'
down_revision: Union[str, None] = 'd77b5366226c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Schema already created with chat_uuid in d77b5366226c (fresh DB bootstrap).
    pass


def downgrade() -> None:
    pass
