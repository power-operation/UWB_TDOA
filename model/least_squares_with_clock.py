import numpy as np
from config import get_anchors, get_tdoa_noise_std

ANCHORS = get_anchors()
TDOA_NOISE_STD = get_tdoa_noise_std()


def compute_distance_with_clock(x, anchor):
    return np.linalg.norm(x[:-1] - anchor) + x[-1]  # + cΔt_R

def compute_jacobian_with_clock(x, anchors, reference_index):
    """
    Construct the matrix H, containing the clock bias, corresponding to Eq. 4.26
    x is [x, y, z, cΔt_R]
    """
    H = []
    x_ref = anchors[reference_index]
    for i, anchor in enumerate(anchors):
        if i == reference_index:
            continue
        d_i = np.linalg.norm(x[:-1] - anchor)
        d_ref = np.linalg.norm(x[:-1] - x_ref)

        if d_i == 0 or d_ref == 0:
            grad = np.zeros_like(x)
        else:
            grad_i = (x[:-1] - anchor) / d_i
            grad_ref = (x[:-1] - x_ref) / d_ref
            grad_spatial = grad_i - grad_ref
            grad = np.concatenate([grad_spatial, [1]])  # ∂/∂cΔt_R = 1
        H.append(grad)
    return np.vstack(H)

def compute_residual_with_clock(x, anchors, tdoa_measurement, reference_index):
    """
    Construct the vector δz, containing the clock deviation
    """
    d_ref = np.linalg.norm(x[:-1] - anchors[reference_index]) + x[-1]
    predicted = []
    for i, anchor in enumerate(anchors):
        if i == reference_index:
            continue
        d_i = np.linalg.norm(x[:-1] - anchor) + x[-1]
        predicted.append(d_i - d_ref)
    predicted = np.array(predicted)
    return tdoa_measurement - predicted

def estimate_positions_least_squares_with_clock(tdoa_measurements, reference_index=0, max_iter=100, tol=1e-4):
    estimated_positions = []
    dim = ANCHORS.shape[1]  # 2 or 3
    std = TDOA_NOISE_STD

    for tdoa in tdoa_measurements:
        # Samples with NaN need to be excluded from the corresponding anchor.
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

        # Initial estimate: centre of space + initial clock deviation set to 0
        x_est = np.zeros(dim + 1)
        x_est[:-1] = np.mean(anchors_used, axis=0)
        x_est[-1] = 0.0

        # Construct covariance weight matrix Wa (diagonal, assuming homoskedasticity）
        N = len(reduced_tdoa)
        Wa = np.eye(N) / (std ** 2)

        for _ in range(max_iter):
            delta_z = compute_residual_with_clock(x_est, anchors_used, reduced_tdoa, reference_index)
            H = compute_jacobian_with_clock(x_est, anchors_used, reference_index)

            try:
                HtW = H.T @ Wa
                delta_x = np.linalg.pinv(HtW @ H) @ HtW @ delta_z
            except np.linalg.LinAlgError:
                delta_x = np.zeros_like(x_est)

            x_new = x_est + delta_x
            if np.linalg.norm(delta_x) < tol:
                break
            x_est = x_new

        estimated_positions.append(x_est[:-1])  # Only retain the spatial coordinates portion

    return np.array(estimated_positions)
