import os

def submit_processing_job(job_name, script_path, **kwargs):
    cmd = f"sbatch --export={','.join([f'{k}={v}' for k, v in kwargs.items()])} {script_path}"
    print(f"Submitting job: {job_name}")
    print(f"Command: {cmd}")
    job_id = os.popen(cmd).read().strip()
    print(f"Job ID: {job_id}")
    return job_id