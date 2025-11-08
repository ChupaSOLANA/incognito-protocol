"""Add payment_note field to escrows table

Revision ID: 776f21bbdb65
Revises: 30ca2435d0cc
Create Date: 2025-11-07 00:29:45.113524

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '776f21bbdb65'
down_revision: Union[str, None] = '30ca2435d0cc'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add payment_note JSONB column
    op.add_column('escrows', sa.Column('payment_note', sa.dialects.postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    # Remove payment_note column
    op.drop_column('escrows', 'payment_note')
