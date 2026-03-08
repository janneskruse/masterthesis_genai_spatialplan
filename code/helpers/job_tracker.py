"""
=================================
Job tracking helper for SLURM pipeline.

Provides functions to record job start/complete/fail events
to per-script CSV files under each region's processed folder,
and to query job completion status for dependency management.
=================================
"""

###### import libraries ######
# Standard libraries
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

# Data handling
import pandas as pd

# Constants
JOB_CSV_COLUMNS = [
    "job_id",
    "script_name",
    "start_time",
    "end_time",
    "duration_seconds",
    "status",
    "error_message",
]

STATUS_IN_PROGRESS = "in_progress"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


def get_job_csv_path(
    big_data_storage_path: str,
    region: str,
    script_name: str,
) -> Path:
    """Return the path to the job tracking CSV for a given region and script.

    The CSV lives at:
        ``{big_data_storage_path}/processed/{region_lower}/jobs/{script_name}.csv``

    Args:
        big_data_storage_path: Root data storage path (e.g. ``/work/zt75vipu-master/data``).
        region: Metropolitan region name (e.g. ``Berlin``).  Will be lower-cased.
        script_name: Logical script name without extension (e.g. ``landsat_to_xarray``).

    Returns:
        Path object pointing to the CSV file.
    """
    jobs_dir = Path(big_data_storage_path) / "processed" / region.lower() / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    return jobs_dir / f"{script_name}.csv"


def _read_or_create_csv(csv_path: Path) -> pd.DataFrame:
    """Read an existing job CSV or create an empty DataFrame with the correct schema."""
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, dtype={"job_id": str})
            # Ensure all expected columns are present
            for col in JOB_CSV_COLUMNS:
                if col not in df.columns:
                    df[col] = pd.NA
            return df
        except Exception:
            # Corrupted file – start fresh
            return pd.DataFrame(columns=JOB_CSV_COLUMNS)
    return pd.DataFrame(columns=JOB_CSV_COLUMNS)


def record_job_start(
    csv_path: Path | str,
    job_id: str,
    script_name: str,
) -> None:
    """Append a new row marking a job as *in_progress*.

    Args:
        csv_path: Path to the job tracking CSV.
        job_id: SLURM job ID or ``"local"`` when running outside SLURM.
        script_name: Logical script name (e.g. ``landsat_to_xarray``).
    """
    csv_path = Path(csv_path)
    df = _read_or_create_csv(csv_path)

    new_row = pd.DataFrame(
        [
            {
                "job_id": str(job_id),
                "script_name": script_name,
                "start_time": datetime.now().isoformat(),
                "end_time": pd.NA,
                "duration_seconds": pd.NA,
                "status": STATUS_IN_PROGRESS,
                "error_message": pd.NA,
            }
        ]
    )
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(csv_path, index=False)


def _update_latest_job(
    csv_path: Path | str,
    job_id: str,
    status: str,
    error_message: Optional[str] = None,
) -> None:
    """Update the most recent row for *job_id* with end time and status.

    If no matching row is found the update is silently skipped (defensive).
    """
    csv_path = Path(csv_path)
    df = _read_or_create_csv(csv_path)

    mask = df["job_id"] == str(job_id)
    if not mask.any():
        # Fallback: update the last row regardless of job_id
        if df.empty:
            return
        mask = pd.Series([False] * len(df))
        mask.iloc[-1] = True

    # Take the *last* matching row (most recent start)
    idx = df.index[mask][-1]
    end_time = datetime.now()
    start_time_str = df.at[idx, "start_time"]

    try:
        start_time = datetime.fromisoformat(start_time_str)
        duration = (end_time - start_time).total_seconds()
    except Exception:
        duration = pd.NA

    df.at[idx, "end_time"] = end_time.isoformat()
    df.at[idx, "duration_seconds"] = duration
    df.at[idx, "status"] = status
    if error_message is not None:
        df.at[idx, "error_message"] = str(error_message)

    df.to_csv(csv_path, index=False)


def record_job_complete(csv_path: Path | str, job_id: str) -> None:
    """Mark the latest run for *job_id* as **completed**."""
    _update_latest_job(csv_path, job_id, STATUS_COMPLETED)


def record_job_failure(
    csv_path: Path | str,
    job_id: str,
    error_message: str,
) -> None:
    """Mark the latest run for *job_id* as **failed** with an error message."""
    _update_latest_job(csv_path, job_id, STATUS_FAILED, error_message=error_message)


def is_script_completed(csv_path: Path | str) -> bool:
    """Return ``True`` if the **latest** job row has status *completed*.

    Returns ``False`` when the CSV does not exist, is empty, or the latest
    row has any other status.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return False
    df = _read_or_create_csv(csv_path)
    if df.empty:
        return False
    return df.iloc[-1]["status"] == STATUS_COMPLETED


def get_latest_job_status(csv_path: Path | str) -> Optional[str]:
    """Return the status string of the most recent job, or ``None`` if no rows exist."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return None
    df = _read_or_create_csv(csv_path)
    if df.empty:
        return None
    return str(df.iloc[-1]["status"])
