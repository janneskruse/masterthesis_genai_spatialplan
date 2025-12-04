import os
from typing import Dict
# from pathlib import Path
import yaml
# import json

def load_configs() -> Dict:
    """Load configuration from config.yml."""
    repo_name = 'masterthesis_genai_spatialplan'
    if repo_name not in os.getcwd():
        try:
            os.chdir(repo_name)
        except FileNotFoundError:
            pass

    if os.path.exists('/.dockerenv'):
        # Running in Docker - use /app as repo_dir
        repo_dir = '/app'
    else:
        p = os.popen('git rev-parse --show-toplevel')
        repo_dir = p.read().strip()
        p.close()
    
    with open(f"{repo_dir}/code/model/config/class_cond.yml", 'r') as stream:
        config = yaml.safe_load(stream)
        # config = json.loads(json.dumps(config))  # Convert None to null
        
    with open(f"{repo_dir}/code/data_acquisition/config.yml", 'r') as stream:
        data_config = yaml.safe_load(stream)        
    
    config['repo_dir'] = repo_dir
    config['data_config'] = data_config
    return config