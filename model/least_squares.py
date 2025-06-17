import numpy as np
from config import get_anchors

ANCHORS = get_anchors()

def compute_distance(x, anchor):
    return np.linalg.norm(x - anchor)

#   least_squares_without_clock

def compute_jacobian(x, anchors, reference_index):
    
    # Construct the matrix Ha(0) (Eq. 4.18)
    
    H = []
    x_ref = anchors[reference_index]
    for i, anchor in enumerate(anchors):
        if i == reference_index:
            continue
        diff = x - anchor
        dist = np.linalg.norm(x - anchor)
        ref_dist = np.linalg.norm(x - x_ref)
        if dist == 0 or ref_dist == 0:
            H.append(np.zeros_like(x))
        else:
            row = diff / dist - (x - x_ref) / ref_dist
            H.append(row)
    return np.vstack(H)

def compute_residual(x, anchors, tdoa_measurement, reference_index):
    """
    Construct the residual vector δza (Eq. 4.19)
    """
    ref_dist = np.linalg.norm(x - anchors[reference_index])
    predicted = []
    for i, anchor in enumerate(anchors):
        if i == reference_index:
            continue
        dist = np.linalg.norm(x - anchor)
        predicted.append(dist - ref_dist)
    predicted = np.array(predicted)
    return tdoa_measurement - predicted  # the δz_a in Eq. 4.19

def estimate_positions_least_squares(tdoa_measurements, reference_index=0, max_iter=100, tol=1e-4):
    estimated_positions = []
    dim = ANCHORS.shape[1]

    for tdoa in tdoa_measurements:
        # Preprocess observations and anchors for NaN data
        if np.any(np.isnan(tdoa)):
            valid_idx = ~np.isnan(tdoa)
            reduced_tdoa = tdoa[valid_idx]
            reduced_anchors = np.delete(ANCHORS, reference_index, axis=0)[valid_idx]
            anchors_used = np.insert(reduced_anchors, reference_index, ANCHORS[reference_index], axis=0)
        else:
            reduced_tdoa = tdoa
            anchors_used = ANCHORS

        if anchors_used.shape[0] < dim + 1:
            estimated_positions.append(np.full((dim,), np.nan))
            continue

        # Initialise estimate x_a(0): it can be averaged by anchor
        x_est = np.mean(anchors_used, axis=0)

        for _ in range(max_iter):
            # Construct the residuals δz and the Jacobi matrix H
            delta_z = compute_residual(x_est, anchors_used, reduced_tdoa, reference_index)
            H = compute_jacobian(x_est, anchors_used, reference_index)

            # Least squares incremental solution δx (Eq. 4.20)
            try:
                delta_x = np.linalg.pinv(H.T @ H) @ H.T @ delta_z
            except np.linalg.LinAlgError:
                delta_x = np.zeros_like(x_est)

            # Update the position estimate x(k+1) = x(k) + δx (Eq. 4.21)
            x_est_new = x_est + delta_x

            # Termination condition ||x_k+1 - x_k|| < threshold (Eq. 4.22)
            if np.linalg.norm(x_est_new - x_est) <= tol:
                break

            x_est = x_est_new

        estimated_positions.append(x_est)

    return np.array(estimated_positions)