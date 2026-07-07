"""
Database engine + session setup.
SQLite file lives at backend/database/pulmo.db (created automatically on first run).
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "sqlite:///./pulmo.db"

# check_same_thread=False needed since FastAPI/Streamlit may access from multiple threads
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI dependency — yields a session, always closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Creates all tables if they don't exist yet. Call once on API startup."""
    Base.metadata.create_all(bind=engine)