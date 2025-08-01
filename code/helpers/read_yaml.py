import sys
import yaml

def get_nested(d, keys):
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k)
        else:
            return None
    return d

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: read_yaml.py <yaml_file> <dot.separated.key>")
        sys.exit(1)
    yaml_file = sys.argv[1]
    key_path = sys.argv[2].split('.')
    with open(yaml_file) as f:
        data = yaml.safe_load(f)
    value = get_nested(data, key_path)
    if isinstance(value, list):
        print(' '.join(map(str, value)))
    else:
        print(value)