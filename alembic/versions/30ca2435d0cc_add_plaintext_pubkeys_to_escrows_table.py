"""Add plaintext pubkeys to escrows table

Revision ID: 30ca2435d0cc
Revises: 4690bb1528d0
Create Date: 2025-11-07 00:06:20.375810

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '30ca2435d0cc'
down_revision: Union[str, None] = '4690bb1528d0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add plaintext pubkey columns (nullable for existing rows)
    op.add_column('escrows', sa.Column('buyer_pubkey', sa.String(length=44), nullable=True))
    op.add_column('escrows', sa.Column('seller_pubkey', sa.String(length=44), nullable=True))


def downgrade() -> None:
    # Remove plaintext pubkey columns
    op.drop_column('escrows', 'seller_pubkey')
    op.drop_column('escrows', 'buyer_pubkey')
