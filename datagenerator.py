import numpy as np
from scipy.interpolate import CubicSpline
from config import (
    SPACE_X, SPACE_Y, SPACE_Z, NUM_TARGETS, ANCHORS, TDOA_NOISE_STD,
    NLOS_BIAS_MEAN, NLOS_BIAS_STD,
    MULTIPATH_DELAY_MEAN, MULTIPATH_DELAY_STD,
    BLOCKAGE_DROP_PROB
)

# def generate_targets(num_targets=NUM_TARGETS):
#     if ANCHORS.shape[1] == 2:
#         return np.random.uniform([0, 0], [SPACE_X, SPACE_Y], size=(num_targets, 2))
#     else:
#         return np.random.uniform([0, 0, 0], [SPACE_X, SPACE_Y, SPACE_Z], size=(num_targets, 3))

def generate_trajectory(num_points=NUM_TARGETS, trajectory_type='line', dimension=2, speed=1.0):
    """
    Generates a continuous motion trajectory.
    Parameters:
        num_points: number of trajectory points
        trajectory_type: type of trajectory ('line', 'circle', 'sinusoid', 'random')
        dimension: 2D or 3D
        speed: movement speed (m/s)
    Returns:
        ndarray, shape=(num_points, dimension): coordinates of track points
    """
    t = np.linspace(0, 10, num_points)  # timeline, 10 secs
    if dimension == 2:
        if trajectory_type == 'line':
            x = speed * t
            y = np.full_like(x, SPACE_Y / 2)
            return np.vstack((x, y)).T
        elif trajectory_type == 'circle':
            radius = min(SPACE_X, SPACE_Y) / 4
            x = SPACE_X / 2 + radius * np.cos(t)
            y = SPACE_Y / 2 + radius * np.sin(t)
            return np.vstack((x, y)).T
        elif trajectory_type == 'sinusoid':
            x = speed * t
            y = SPACE_Y / 2 + (SPACE_Y / 4) * np.sin(t)
            return np.vstack((x, y)).T
        elif trajectory_type == 'random':
            # Generate random control points and use spline interpolation
            num_control = min(5, num_points)
            control_t = np.linspace(0, 10, num_control)
            control_x = np.random.uniform(0, SPACE_X, num_control)
            control_y = np.random.uniform(0, SPACE_Y, num_control)
            spline_x = CubicSpline(control_t, control_x, bc_type='natural')
            spline_y = CubicSpline(control_t, control_y, bc_type='natural')
            x = spline_x(t)
            y = spline_y(t)
            return np.vstack((x, y)).T
    else:  # 3D
        if trajectory_type == 'line':
            x = speed * t
            y = np.full_like(x, SPACE_Y / 2)
            z = np.full_like(x, SPACE_Z / 2)
            return np.vstack((x, y, z)).T
        elif trajectory_type == 'circle':
            radius = min(SPACE_X, SPACE_Y) / 4
            x = SPACE_X / 2 + radius * np.cos(t)
            y = SPACE_Y / 2 + radius * np.sin(t)
            z = np.full_like(x, SPACE_Z / 2)
            return np.vstack((x, y, z)).T
        elif trajectory_type == 'sinusoid':
            x = speed * t
            y = SPACE_Y / 2 + (SPACE_Y / 4) * np.sin(t)
            z = np.full_like(x, SPACE_Z / 2)
            return np.vstack((x, y, z)).T
        elif trajectory_type == 'random':
            num_control = min(5, num_points)
            control_t = np.linspace(0, 10, num_control)
            control_x = np.random.uniform(0, SPACE_X, num_control)
            control_y = np.random.uniform(0, SPACE_Y, num_control)
            control_z = np.random.uniform(0, SPACE_Z, num_control)
            spline_x = CubicSpline(control_t, control_x, bc_type='natural')
            spline_y = CubicSpline(control_t, control_y, bc_type='natural')
            spline_z = CubicSpline(control_t, control_z, bc_type='natural')
            x = spline_x(t)
            y = spline_y(t)
            z = spline_z(t)
            return np.vstack((x, y, z)).T
    return np.zeros((num_points, dimension))

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

# def generate_simulated_data(enable_nlos=False, enable_multipath=False, enable_blockage=False):
#     targets = generate_targets()
#     distances = compute_distances(targets, ANCHORS)
#     tdoa_measurements, problem_mask = simulate_tdoa_measurements(
#         distances, enable_nlos, enable_multipath, enable_blockage
#     )
#     return targets, tdoa_measurements, problem_mask

def generate_simulated_data(enable_nlos=False, enable_multipath=False, enable_blockage=False, 
                            trajectory_type='line'):
    targets = generate_trajectory(NUM_TARGETS, trajectory_type, dimension=ANCHORS.shape[1])
    distances = compute_distances(targets, ANCHORS)
    tdoa_measurements, problem_mask = simulate_tdoa_measurements(
        distances, enable_nlos, enable_multipath, enable_blockage
    )
    return targets, tdoa_measurements, problem_mask