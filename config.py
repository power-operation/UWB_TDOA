import numpy as np
import argparse
import yaml

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
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="UWB TDOA Simulation")
    parser.add_argument('--cfg', default='configs/model.yaml', help='Path to model configuration YAML file')
    parser.add_argument('--model', default='least_squares', help='Model to use (e.g., least_squares, taylor)')
    parser.add_argument('--dimension', type=int, choices=[2, 3], default=None, help='Dimension (2 or 3)')
    parser.add_argument('--nlos', type=lambda x: x.lower() == 'true', default=None, help='Enable NLOS (true/false)')
    parser.add_argument('--multipath', type=lambda x: x.lower() == 'true', default=None, help='Enable multipath (true/false)')
    parser.add_argument('--blockage', type=lambda x: x.lower() == 'true', default=None, help='Enable blockage (true/false)')
    parser.add_argument('--process', type=lambda x: x.lower() == 'true', default=True, help='Enable preprocessing (true/false)')
    return parser.parse_args()

# Load YAML configuration
args = parse_args()
config = load_yaml_config(args.cfg)

# Configuration parameters with defaults from YAML
DIMENSION = args.dimension if args.dimension is not None else config['defaults']['dimension']
SPACE_X = config['defaults']['space_x']
SPACE_Y = config['defaults']['space_y']
SPACE_Z = config['defaults']['space_z']
NUM_TARGETS = config['defaults']['num_targets']
TDOA_NOISE_STD = config['defaults']['tdoa_noise_std']
NLOS_BIAS_MEAN = config['defaults']['nlos_bias_mean']
NLOS_BIAS_STD = config['defaults']['nlos_bias_std']
MULTIPATH_DELAY_MEAN = config['defaults']['multipath_delay_mean']
MULTIPATH_DELAY_STD = config['defaults']['multipath_delay_std']
BLOCKAGE_DROP_PROB = config['defaults']['blockage_drop_prob']

# Interference flags with command-line overrides
ENABLE_NLOS = args.nlos if args.nlos is not None else False
ENABLE_MULTIPATH = args.multipath if args.multipath is not None else False
ENABLE_BLOCKAGE = args.blockage if args.blockage is not None else False

# Anchor positions based on dimension
if DIMENSION == 2:
    ANCHORS = np.array([
        [0, 0],
        [0, SPACE_Y],
        [SPACE_X, 0],
        [SPACE_X, SPACE_Y],
    ])
else:  # DIMENSION == 3 UWB needs to be non-coplanar
    ANCHORS = np.array([
        [0, 0, 1],
        [0, SPACE_Y, 4],
        [SPACE_X, 0, 3],
        [SPACE_X, SPACE_Y, 2],
    ])