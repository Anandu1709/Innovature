"""
CSV Upload router — authenticated file upload with metadata extraction.
"""

import os
import uuid

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.config import get_settings
from app.database import get_db
from app.models import User, UploadedFile
from app.schemas import FileUploadResponse

settings = get_settings()
router = APIRouter(prefix="/csv", tags=["CSV Operations"])

UPLOAD_DIR = "uploads"


@router.post(
    "/upload",
    response_model=FileUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a CSV file",
)
async def upload_csv(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a CSV file for the authenticated user.

    **Validations:**
    - Only `.csv` files are accepted.
    - Maximum file size is configurable (default 10 MB).

    **Response includes:**
    - File name, total rows/columns, column names, data types, missing values per column.
    """
    # ── Validate file extension ──────────────────
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files are allowed. Please upload a file with .csv extension.",
        )

    # ── Validate content type (additional check) ──
    if file.content_type and file.content_type not in [
        "text/csv",
        "application/vnd.ms-excel",
        "application/octet-stream",
    ]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid content type: {file.content_type}. Expected text/csv.",
        )

    # ── Read file content and validate size ──────
    content = await file.read()
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024

    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds the {settings.MAX_FILE_SIZE_MB} MB limit.",
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # ── Create user-specific upload directory ────
    user_dir = os.path.join(UPLOAD_DIR, str(current_user.id))
    os.makedirs(user_dir, exist_ok=True)

    # ── Save file with UUID prefix to avoid collisions ──
    unique_prefix = uuid.uuid4().hex[:8]
    stored_filename = f"{unique_prefix}_{file.filename}"
    file_path = os.path.join(user_dir, stored_filename)

    with open(file_path, "wb") as f:
        f.write(content)

    # ── Parse CSV with pandas ────────────────────
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        # Clean up the invalid file
        os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse CSV file: {str(e)}",
        )

    # ── Record file metadata in database ─────────
    uploaded_file = UploadedFile(
        user_id=current_user.id,
        original_filename=file.filename,
        stored_filename=stored_filename,
    )
    db.add(uploaded_file)
    await db.flush()

    # ── Build response ───────────────────────────
    return FileUploadResponse(
        file_name=file.filename,
        total_rows=len(df),
        total_columns=len(df.columns),
        column_names=df.columns.tolist(),
        data_types={col: str(dtype) for col, dtype in df.dtypes.items()},
        missing_values={col: int(df[col].isnull().sum()) for col in df.columns},
    )
