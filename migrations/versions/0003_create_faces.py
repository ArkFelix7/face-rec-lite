"""create faces table

Revision ID: c3d4e5f6a1b2
Revises: b2c3d4e5f6a1
Create Date: 2024-01-01 00:02:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c3d4e5f6a1b2"
down_revision: Union[str, None] = "b2c3d4e5f6a1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "faces",
        sa.Column("id", sa.UUID(), nullable=False, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("embedding", sa.Text(), nullable=False),  # stored via pgvector type
        sa.Column("image_hash", sa.Text(), nullable=False),
        sa.Column("quality_score", sa.Float(), nullable=False),
        sa.Column("blur_score", sa.Float(), nullable=False),
        sa.Column("brightness", sa.Float(), nullable=False),
        sa.Column("face_confidence", sa.Float(), nullable=False),
        sa.Column("face_width_px", sa.Integer(), nullable=False),
        sa.Column("face_height_px", sa.Integer(), nullable=False),
        sa.Column("pitch_deg", sa.Float(), nullable=False),
        sa.Column("yaw_deg", sa.Float(), nullable=False),
        sa.Column("roll_deg", sa.Float(), nullable=False),
        sa.Column("label", sa.Text(), nullable=True),
        sa.Column(
            "enrolled_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    # Alter the embedding column to the proper vector(512) type now that the extension is loaded
    op.execute("ALTER TABLE faces ALTER COLUMN embedding TYPE vector(512) USING embedding::vector(512)")

    op.create_index("idx_faces_user_id", "faces", ["user_id"])
    op.create_index("idx_faces_image_hash", "faces", ["image_hash"])

    # IVFFlat index for approximate nearest-neighbour cosine similarity search
    op.execute(
        "CREATE INDEX idx_faces_embedding ON faces "
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_faces_embedding")
    op.drop_index("idx_faces_image_hash", table_name="faces")
    op.drop_index("idx_faces_user_id", table_name="faces")
    op.drop_table("faces")
