"""
SQLAlchemy ORM models for User and UploadedFile.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base


class User(Base):
    """Registered user account."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationship to uploaded files
    files = relationship("UploadedFile", back_populates="owner", cascade="all, delete-orphan")


class UploadedFile(Base):
    """Metadata for a CSV file uploaded by a user."""

    __tablename__ = "uploaded_files"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    original_filename = Column(String(255), nullable=False)
    stored_filename = Column(String(255), nullable=False)
    upload_time = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationship back to user
    owner = relationship("User", back_populates="files")
