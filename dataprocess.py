import numpy as np
# from config import NLOS_BIAS_MEAN, NLOS_BIAS_STD, MULTIPATH_DELAY_MEAN, MULTIPATH_DELAY_STD
from config import (
    get_space, get_num_targets, get_anchors,
    get_tdoa_noise_std, get_nlos_params,
    get_multipath_params, get_blockage_prob
)

SPACE_X, SPACE_Y, SPACE_Z = get_space()
NUM_TARGETS = get_num_targets()
ANCHORS = get_anchors()
TDOA_NOISE_STD = get_tdoa_noise_std()
NLOS_BIAS_MEAN, NLOS_BIAS_STD = get_nlos_params()
MULTIPATH_DELAY_MEAN, MULTIPATH_DELAY_STD = get_multipath_params()
BLOCKAGE_DROP_PROB = get_blockage_prob()

def preprocess_tdoa(tdoa_measurements, problem_mask, strategy="adaptive"):
    tdoa = tdoa_measurements.copy()

    if strategy == "mask":
        tdoa[problem_mask > 0] = np.nan

    elif strategy == "adaptive":
        # 1. NLOS processing: direct discard
        nlos_mask = (problem_mask == 1)
        tdoa[nlos_mask] = np.nan

        # 2. Multipath processing: filling with medians
        for i in range(tdoa.shape[0]):
            multipath_idx = (problem_mask[i] == 2)
            valid = tdoa[i][(problem_mask[i] == 0) & ~np.isnan(tdoa[i])]
            if valid.size > 0:
                replacement = np.median(valid)
                tdoa[i][multipath_idx] = replacement
            else:
                tdoa[i][multipath_idx] = np.nan

        # 3. Shading: mean fill if RMS is sufficient
        for i in range(tdoa.shape[0]):
            blockage_idx = (problem_mask[i] == 3)
            valid = tdoa[i][(problem_mask[i] == 0) & ~np.isnan(tdoa[i])]
            if valid.size >= tdoa.shape[1] // 2:
                tdoa[i][blockage_idx] = np.mean(valid)
            else:
                tdoa[i][blockage_idx] = np.nan

    return tdoa
