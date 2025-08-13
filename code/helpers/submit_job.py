import subprocess
import shlex

def run_command(command, capture_output=True):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(command, shell=True, capture_output=capture_output, text=True, check=True)
        return result.stdout.strip() if capture_output else None
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Return code: {e.returncode}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return None

def check_existing_job(job_name_pattern):
    """Check if a job with the given name pattern is already running."""
    command = f"squeue -u $USER --name=\"{job_name_pattern}\" --noheader --format=\"%i\""
    result = run_command(command)
    return result if result else None

# def submit_job_with_dependency(script_path, dependency_job_id=None, **kwargs):
#     """Submit a SLURM job with optional dependency."""
#     escaped_kwargs = {}
#     for k, v in kwargs.items():
#         k = k.upper()  # uppercase the key
#         if isinstance(v, list):
#             # List to space-separated string for SLURM
#             # list_str = ' '.join(str(item) for item in v)
#             # escaped_kwargs[k] = f'"{list_str}"'
#             list_str = ":".join(str(item) for item in v)
#             escaped_kwargs[k] = f"'{list_str}'"
#         elif isinstance(v, str) and (v.startswith('{') or '"' in v):
#             # Single quotes to wrap region names etc.
#             escaped_v = v.replace("'", "'\"'\"'")
#             escaped_kwargs[k] = f"'{escaped_v}'"
#         else:
#             escaped_kwargs[k] = v
    
#     export_str = ",".join([f"{k}={v}" for k, v in escaped_kwargs.items()])
    
#     #Extract job id
#     if dependency_job_id:
#         if "Submitted batch job" in str(dependency_job_id):
#             dependency_job_id = dependency_job_id.split()[-1]
#         cmd = f"sbatch --parsable --dependency=afterok:{dependency_job_id} --export={export_str} {script_path}"
#     else:
#         cmd = f"sbatch --parsable --export={export_str} {script_path}"
    
#     print(f"Command: {cmd}")
#     result = run_command(cmd)
#     if result and "Submitted batch job" in result:
#         job_id = result.split()[-1]  # Extract job number
#     else:
#         job_id = result
#     return job_id

def submit_job_with_dependency(script_path, dependency_job_id=None, **kwargs):
    """
    Submit a SLURM job with optional dependency.

    - Lists are joined with ':' (parse in bash with IFS=':' or readarray trick).
    - All variables are passed via env prefix to avoid --export comma parsing.
    - Sets --export=ALL so the prefixed env is inherited by sbatch.
    """
    env_assignments = []
    for k, v in kwargs.items():
        key = k.upper()
        if isinstance(v, list):
            v = ":".join(str(item) for item in v)
        else:
            v = str(v)
        env_assignments.append(f"{key}={shlex.quote(v)}")

    env_prefix = " ".join(env_assignments)

    # Normalize dependency id (handles "Submitted batch job 12345" or "12345")
    dep_part = ""
    if dependency_job_id:
        dep_id = str(dependency_job_id).split()[-1]
        dep_part = f"--dependency=afterok:{dep_id} "

    cmd = f"{env_prefix} sbatch --parsable {dep_part}--export=ALL {shlex.quote(script_path)}"
    print(f"Command: {cmd}")

    # Run the command
    result = run_command(cmd)
    
    # Return the job ID if successful
    if not result:
        return result
    result = result.strip()
    if "Submitted batch job" in result:
        return result.split()[-1]
    return result
