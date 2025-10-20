import sys
import yaml

def get_nested(d, keys, default=None):
    """
    Get nested value from dictionary using list of keys.
    
    :param d: dictionary to search
    :param keys: list of keys to traverse
    :param default: default value to return if key path not found
    :return: value at nested location or default
    """
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k)
            if d is None:
                return default
        else:
            return default
    return d

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: read_yaml.py <yaml_file> <dot.separated.key> [default_value]")
        sys.exit(1)
    
    yaml_file = sys.argv[1]
    key_path = sys.argv[2].split('.')
    default_value = sys.argv[3] if len(sys.argv) > 3 else None
    
    with open(yaml_file) as f:
        data = yaml.safe_load(f)
    
    value = get_nested(data, key_path, default=default_value)
    
    if isinstance(value, list):
        print(' '.join(map(str, value)))
    else:
        print(value)