import numpy as np
from config import (
    SPACE_X, SPACE_Y, SPACE_Z, NUM_TARGETS, ANCHORS, TDOA_NOISE_STD,
    NLOS_BIAS_MEAN, NLOS_BIAS_STD,
    MULTIPATH_DELAY_MEAN, MULTIPATH_DELAY_STD,
    BLOCKAGE_DROP_PROB
)

def generate_targets(num_targets=NUM_TARGETS):
    if ANCHORS.shape[1] == 2:
        return np.random.uniform([0, 0], [SPACE_X, SPACE_Y], size=(num_targets, 2))
    else:
        return np.random.uniform([0, 0, 0], [SPACE_X, SPACE_Y, SPACE_Z], size=(num_targets, 3))

def compute_distances(points, anchors):
    return np.linalg.norm(points[:, np.newaxis, :] - anchors[np.newaxis, :, :], axis=2)

def simulate_tdoa_measurements(distances, enable_nlos, enable_multipath, enable_blockage, reference_index=0):
    N, M = distances.shape
    ref_dist = distances[:, reference_index][:, np.newaxis]
    tdoa = distances - ref_dist

    noise = np.random.normal(0, TDOA_NOISE_STD, size=tdoa.shape)
    tdoa_noisy = tdoa + noise
    problem_mask = np.zeros_like(tdoa_noisy, dtype=int)

    if enable_nlos:
        nlos_mask = np.random.rand(*tdoa.shape) < 0.2
        bias = np.random.normal(NLOS_BIAS_MEAN, NLOS_BIAS_STD, size=tdoa.shape)
        tdoa_noisy += nlos_mask * bias
        problem_mask[nlos_mask] = 1

    if enable_multipath:
        multipath_mask = np.random.rand(*tdoa.shape) < 0.2
        delay = np.abs(np.random.normal(MULTIPATH_DELAY_MEAN, MULTIPATH_DELAY_STD, size=tdoa.shape))
        tdoa_noisy += multipath_mask * delay
        problem_mask[multipath_mask] = 2

    if enable_blockage:
        blockage_mask = np.random.rand(*tdoa.shape) < BLOCKAGE_DROP_PROB
        tdoa_noisy[blockage_mask] = np.nan
        problem_mask[blockage_mask] = 3

    tdoa_noisy = np.delete(tdoa_noisy, reference_index, axis=1)
    problem_mask = np.delete(problem_mask, reference_index, axis=1)

    return tdoa_noisy, problem_mask

def generate_simulated_data(enable_nlos=False, enable_multipath=False, enable_blockage=False):
    targets = generate_targets()
    distances = compute_distances(targets, ANCHORS)
    tdoa_measurements, problem_mask = simulate_tdoa_measurements(
        distances, enable_nlos, enable_multipath, enable_blockage
    )
    return targets, tdoa_measurements, problem_mask
