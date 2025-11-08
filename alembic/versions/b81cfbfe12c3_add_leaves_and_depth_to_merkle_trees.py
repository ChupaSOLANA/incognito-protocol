"""add_leaves_and_depth_to_merkle_trees

Revision ID: b81cfbfe12c3
Revises: 776f21bbdb65
Create Date: 2025-11-07 11:00:16.572082

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'b81cfbfe12c3'
down_revision: Union[str, None] = '776f21bbdb65'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add leaves column (JSONB array) to merkle_trees table
    op.add_column('merkle_trees', sa.Column('leaves', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='[]'))

    # Add depth column to merkle_trees table
    op.add_column('merkle_trees', sa.Column('depth', sa.Integer(), nullable=False, server_default='20'))


def downgrade() -> None:
    # Remove the added columns
    op.drop_column('merkle_trees', 'depth')
    op.drop_column('merkle_trees', 'leaves')
