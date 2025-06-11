from nanoid import generate
from sqlalchemy import Boolean, Column

from src.core.db.mixins.timestamp_mixin import TimestampMixin
from src.core.db.session import Base


def generate_uuid(entity: str, size: int = 10) -> str:
    """Generate a unique identifier prefixed with entity name.

    Args:
        entity: Entity prefix (e.g., "model", "task")
        size: Size of random part

    Returns:
        Unique identifier in format "{entity}_{random_id}"
    """
    nano_id = generate(size=size)
    return f"{entity}_{nano_id}"


class BaseModel(Base, TimestampMixin):
    __abstract__ = True

    is_deleted = Column(Boolean, nullable=False, default=False)
