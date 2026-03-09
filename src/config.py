import yaml

def load_params(params_path='params.yaml'):
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)
    return params

# Usage: params = load_params()