import os
from typing import Dict
import argparse
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
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='code/model/config/diffusion_1.yml', help='Path to the experiment config file')
    parser.add_argument('--data_config', type=str, default='code/data_acquisition/config.yml', help='Path to the data config file')
    args = parser.parse_args()
    
    print(f"Loading config from: {args.config}")
    print(f"Loading data config from: {args.data_config}")

    if os.path.exists('/.dockerenv'):
        # Running in Docker - use /app as repo_dir
        repo_dir = '/app'
    else:
        p = os.popen('git rev-parse --show-toplevel')
        repo_dir = p.read().strip()
        p.close()
    
    with open(f"{repo_dir}/{args.config}", 'r') as stream:
        config = yaml.safe_load(stream)
        # config = json.loads(json.dumps(config))  # Convert None to null
        
    with open(f"{repo_dir}/{args.data_config}", 'r') as stream:
        data_config = yaml.safe_load(stream)        
    
    config['repo_dir'] = repo_dir
    config['data_config'] = data_config
    return config