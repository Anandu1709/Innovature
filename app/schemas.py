"""
Pydantic schemas for request validation and response serialization.
"""

from pydantic import BaseModel, EmailStr, Field
from typing import Dict, List
from datetime import datetime


# ──────────────────────────────────────────────
# Auth Schemas
# ──────────────────────────────────────────────

class UserCreate(BaseModel):
    """Request body for user registration."""

    username: str = Field(..., min_length=3, max_length=50, examples=["john_doe"])
    email: EmailStr = Field(..., examples=["john@example.com"])
    password: str = Field(..., min_length=6, max_length=128, examples=["strongP@ss123"])


class UserResponse(BaseModel):
    """Response after successful registration."""

    id: int
    username: str
    email: str
    created_at: datetime

    class Config:
        from_attributes = True


class LoginRequest(BaseModel):
    """Request body for user login."""

    username: str = Field(..., examples=["john_doe"])
    password: str = Field(..., examples=["strongP@ss123"])


class Token(BaseModel):
    """JWT access token response."""

    access_token: str
    token_type: str = "bearer"


# ──────────────────────────────────────────────
# CSV Upload Schemas
# ──────────────────────────────────────────────

class FileUploadResponse(BaseModel):
    """Response after successful CSV upload."""

    file_name: str
    total_rows: int
    total_columns: int
    column_names: List[str]
    data_types: Dict[str, str]
    missing_values: Dict[str, int]


# ──────────────────────────────────────────────
# CSV Cleaning Schemas
# ──────────────────────────────────────────────

class CleanRequest(BaseModel):
    """Request body for CSV cleaning operation."""

    file_name: str = Field(..., description="Name of the previously uploaded CSV file")
    clean: int = Field(
        ...,
        ge=1,
        le=5,
        description=(
            "Cleaning operation: "
            "1=Drop rows with missing values, "
            "2=Fill missing with column mean, "
            "3=Forward fill, "
            "4=Backward fill, "
            "5=Remove duplicates"
        ),
    )


class CleanResponse(BaseModel):
    """Response after successful CSV cleaning."""

    original_file_name: str
    cleaned_file_name: str
    total_rows_before: int
    total_rows_after: int
    records_removed_or_modified: int
