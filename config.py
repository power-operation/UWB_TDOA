import numpy as np
import argparse
import yaml

args = None
config = None

def load_yaml_config(yaml_file):
    """Load configuration from a YAML file."""
    try:
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
            if config is None:
                raise ValueError(f"YAML file '{yaml_file}' is empty or invalid")
            return config
    except FileNotFoundError:
        raise FileNotFoundError(f"YAML file '{yaml_file}' not found")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file '{yaml_file}': {str(e)}")

def parse_args():
    """Parse command-line arguments, or provide defaults when under test."""
    import sys
    parser = argparse.ArgumentParser(description="UWB TDOA Simulation")
    parser.add_argument('--cfg', default='configs/model.yaml', help='Path to model configuration YAML file')
    parser.add_argument('--model', default='least_squares', help='Model to use (e.g., least_squares, taylor)')
    parser.add_argument('--dimension', type=int, choices=[2, 3], default=2, help='Dimension (2 or 3)')
    parser.add_argument('--nlos', type=lambda x: x.lower() == 'true', default=False, help='Enable NLOS (true/false)')
    parser.add_argument('--multipath', type=lambda x: x.lower() == 'true', default=False, help='Enable multipath (true/false)')
    parser.add_argument('--blockage', type=lambda x: x.lower() == 'true', default=False, help='Enable blockage (true/false)')
    parser.add_argument('--process', type=lambda x: x.lower() == 'true', default=True, help='Enable preprocessing (true/false)')
    parser.add_argument('--trajectory', default='line', choices=['line', 'circle', 'sinusoid', 'random'], help='Trajectory type for target movement')

    # If running in pytest, instead of parsing the command line, use the default parameter
    if "pytest" in sys.modules:
        return parser.parse_args([])
    else:
        return parser.parse_args()


def get_config():
    """Load args and config once"""
    global args, config
    if args is None or config is None:
        args = parse_args()
        config = load_yaml_config(args.cfg)
    return args, config

def get_dimension():
    args, config = get_config()
    return args.dimension if args.dimension is not None else config['defaults']['dimension']

def get_space():
    _, config = get_config()
    return config['defaults']['space_x'], config['defaults']['space_y'], config['defaults']['space_z']

def get_num_targets():
    _, config = get_config()
    return config['defaults']['num_targets']

def get_tdoa_noise_std():
    _, config = get_config()
    return config['defaults']['tdoa_noise_std']

def get_interference_flags():
    args, _ = get_config()
    return (
        args.nlos if args.nlos is not None else False,
        args.multipath if args.multipath is not None else False,
        args.blockage if args.blockage is not None else False
    )

def get_anchors():
    dim = get_dimension()
    space_x, space_y, space_z = get_space()
    if dim == 2:
        return np.array([
            [0, 0],
            [0, space_y],
            [space_x, 0],
            [space_x, space_y],
        ])
    else:
        return np.array([
            [0, 0, 1],
            [0, space_y, 4],
            [space_x, 0, 3],
            [space_x, space_y, 2],
        ])

def get_nlos_params():
    _, config = get_config()
    return config['defaults']['nlos_bias_mean'], config['defaults']['nlos_bias_std']

def get_multipath_params():
    _, config = get_config()
    return config['defaults']['multipath_delay_mean'], config['defaults']['multipath_delay_std']

def get_blockage_prob():
    _, config = get_config()
    return config['defaults']['blockage_drop_prob']
