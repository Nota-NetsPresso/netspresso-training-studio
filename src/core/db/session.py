from contextlib import contextmanager
from typing import Generator

from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy_utils import create_database, database_exists

from src.configs.settings import settings

engine = create_engine(
    f"{settings.DATABASE_URL}",
    pool_pre_ping=True,
    pool_use_lifo=True,
    pool_recycle=3600,
)
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False,
)

Base = declarative_base()


def init_db() -> None:
    """Initialize database if it doesn't exist."""
    if not database_exists(engine.url):
        logger.info("Create database...")
        create_database(engine.url)


def get_db() -> Generator:
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()


@contextmanager
def get_db_session():
    db = None
    try:
        db = SessionLocal()
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        raise e
    finally:
        db.close()
