import numpy as np
from config import NLOS_BIAS_MEAN, NLOS_BIAS_STD, MULTIPATH_DELAY_MEAN, MULTIPATH_DELAY_STD

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
