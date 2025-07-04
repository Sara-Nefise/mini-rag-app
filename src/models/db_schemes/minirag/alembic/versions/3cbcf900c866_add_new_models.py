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
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('chunks', schema=None) as batch_op:
        batch_op.drop_constraint('chunks_chunk_asset_id_fkey', type_='foreignkey')
        batch_op.drop_constraint('chunks_chunk_project_id_fkey', type_='foreignkey')
        batch_op.create_foreign_key(None, 'projects', ['chunk_project_id'], ['project_id'])
        batch_op.create_foreign_key(None, 'assets', ['chunk_asset_id'], ['asset_id'])

    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('chunks', schema=None) as batch_op:
        batch_op.drop_constraint(None, type_='foreignkey')
        batch_op.drop_constraint(None, type_='foreignkey')
        batch_op.create_foreign_key('chunks_chunk_project_id_fkey', 'projects', ['chunk_project_id'], ['project_id'], ondelete='CASCADE')
        batch_op.create_foreign_key('chunks_chunk_asset_id_fkey', 'assets', ['chunk_asset_id'], ['asset_id'], ondelete='CASCADE')

    # ### end Alembic commands ###
