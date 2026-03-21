"""
CSV Cleaning router — perform predefined data cleaning operations
on previously uploaded CSV files.
"""

import os

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.database import get_db
from app.models import User, UploadedFile
from app.schemas import CleanRequest, CleanResponse

router = APIRouter(prefix="/csv", tags=["CSV Operations"])

UPLOAD_DIR = "uploads"

# Human-readable names for each cleaning operation
CLEAN_OPERATION_NAMES = {
    1: "drop_missing",
    2: "fill_mean",
    3: "forward_fill",
    4: "backward_fill",
    5: "remove_duplicates",
}


@router.post(
    "/clean",
    response_model=CleanResponse,
    summary="Clean a previously uploaded CSV file",
)
async def clean_csv(
    request: CleanRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Perform a data cleaning operation on a previously uploaded CSV file.

    **Operations:**
    - `clean=1` → Drop all rows with any missing values
    - `clean=2` → Fill missing values with column mean (numeric) / mode (non-numeric)
    - `clean=3` → Forward fill missing values
    - `clean=4` → Backward fill missing values
    - `clean=5` → Remove duplicate rows

    **Security:** Only files belonging to the authenticated user can be cleaned.
    """
    # ── Find the file in database (must belong to current user) ──
    result = await db.execute(
        select(UploadedFile)
        .where(
            UploadedFile.original_filename == request.file_name,
            UploadedFile.user_id == current_user.id,
        )
        .order_by(UploadedFile.upload_time.desc())
        .limit(1)
    )
    file_record = result.scalar_one_or_none()

    if not file_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{request.file_name}' not found. Make sure you have uploaded this file.",
        )

    # ── Construct file path and verify it exists on disk ──
    user_dir = os.path.join(UPLOAD_DIR, str(current_user.id))
    file_path = os.path.join(user_dir, file_record.stored_filename)

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File exists in records but not on disk. It may have been deleted.",
        )

    # ── Load CSV into DataFrame ──────────────────
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read CSV file: {str(e)}",
        )

    rows_before = len(df)

    # ── Apply cleaning operation ─────────────────
    if request.clean == 1:
        # Drop all rows containing any missing values
        df_cleaned = df.dropna()

    elif request.clean == 2:
        # Fill missing values: mean for numeric columns, mode for non-numeric
        df_cleaned = df.copy()
        for col in df_cleaned.columns:
            if df_cleaned[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
                else:
                    mode_val = df_cleaned[col].mode()
                    if not mode_val.empty:
                        df_cleaned[col] = df_cleaned[col].fillna(mode_val[0])

    elif request.clean == 3:
        # Forward fill
        df_cleaned = df.ffill()

    elif request.clean == 4:
        # Backward fill
        df_cleaned = df.bfill()

    elif request.clean == 5:
        # Remove duplicate rows
        df_cleaned = df.drop_duplicates()

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid clean parameter. Must be between 1 and 5.",
        )

    rows_after = len(df_cleaned)

    # ── Save cleaned file ────────────────────────
    operation_name = CLEAN_OPERATION_NAMES[request.clean]
    base_name, ext = os.path.splitext(file_record.stored_filename)
    cleaned_filename = f"cleaned_{operation_name}_{file_record.original_filename}"
    cleaned_stored_name = f"cleaned_{operation_name}_{file_record.stored_filename}"
    cleaned_path = os.path.join(user_dir, cleaned_stored_name)

    df_cleaned.to_csv(cleaned_path, index=False)

    # ── Record cleaned file in database ──────────
    cleaned_file_record = UploadedFile(
        user_id=current_user.id,
        original_filename=cleaned_filename,
        stored_filename=cleaned_stored_name,
    )
    db.add(cleaned_file_record)
    await db.flush()

    # ── Build response ───────────────────────────
    records_changed = abs(rows_before - rows_after)

    # For fill operations (2, 3, 4), rows stay the same but values change
    if request.clean in (2, 3, 4):
        # Count cells that were modified (originally null, now filled)
        original_nulls = df.isnull().sum().sum()
        remaining_nulls = df_cleaned.isnull().sum().sum()
        records_changed = int(original_nulls - remaining_nulls)

    return CleanResponse(
        original_file_name=request.file_name,
        cleaned_file_name=cleaned_filename,
        total_rows_before=rows_before,
        total_rows_after=rows_after,
        records_removed_or_modified=records_changed,
    )
