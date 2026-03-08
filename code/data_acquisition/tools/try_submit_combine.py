"""
================================================================================
Check whether all upstream jobs (landsat, osm, process_planetscope) are
completed for a given region. If so, and if combine_datasets is not already
completed or in-progress, submit the combine job via SLURM.
================================================================================
"""
import argparse
import os
import sys

from helpers.load_configs import load_configs
from helpers.job_tracker import (
    get_job_csv_path,
    is_script_completed,
    get_latest_job_status,
    STATUS_IN_PROGRESS,
)
from helpers.submit_job import submit_job_with_dependency, check_existing_job


UPSTREAM_SCRIPTS = [
    "landsat_to_xarray",
    "osm_to_xarray",
    "process_planetscope",
]


def main():
    parser = argparse.ArgumentParser(
        description="Conditionally submit combine_datasets job if upstream steps are done."
    )
    parser.add_argument("--REGION", type=str, required=True)
    args = parser.parse_args()

    region = args.REGION.title()

    config = load_configs()
    data_config = config.get("data_config", {})
    big_data_storage_path = data_config.get("big_data_storage_path", "/work/zt75vipu-master/data")

    # Check upstream completion
    all_done = True
    for script in UPSTREAM_SCRIPTS:
        csv_path = get_job_csv_path(big_data_storage_path, region, script)
        done = is_script_completed(csv_path)
        print(f"  {script}: {'DONE' if done else 'PENDING'}")
        if not done:
            all_done = False

    if not all_done:
        print(f"Not all upstream steps completed for {region}. Skipping combine submission.")
        sys.exit(0)

    # Check combine status
    combine_csv = get_job_csv_path(big_data_storage_path, region, "combine_datasets")
    if is_script_completed(combine_csv):
        print(f"combine_datasets already completed for {region}. Skipping.")
        sys.exit(0)

    combine_status = get_latest_job_status(combine_csv)
    if combine_status == STATUS_IN_PROGRESS:
        print(f"combine_datasets is already in-progress for {region}. Skipping.")
        sys.exit(0)

    # Check SLURM queue for a running combine job
    existing_job = check_existing_job("Combine_Datasets")
    if existing_job:
        print(f"Combine job already queued/running (Job ID: {existing_job}). Skipping.")
        sys.exit(0)

    # Submit the combine job
    print(f"All upstream steps done for {region}. Submitting combine_datasets job.")
    job_id = submit_job_with_dependency(
        "./combine_datasets.sh",
        region=region,
    )
    print(f"Submitted combine job for {region}: {job_id}")


if __name__ == "__main__":
    main()
