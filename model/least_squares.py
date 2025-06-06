import numpy as np
from scipy.optimize import least_squares
from config import ANCHORS

def tdoa_residual(x, anchors, tdoa_measurement, reference_index=0):
    distances = np.linalg.norm(anchors - x, axis=1)
    ref_distance = distances[reference_index]
    pred_tdoa = distances - ref_distance
    pred_tdoa = np.delete(pred_tdoa, reference_index)
    return pred_tdoa - tdoa_measurement

def estimate_positions_least_squares(tdoa_measurements, reference_index=0):
    estimated_positions = []

    for tdoa in tdoa_measurements:
        # If there is NaN in this sample, exclude relevant observations and base stations
        if np.any(np.isnan(tdoa)):
            valid_idx = ~np.isnan(tdoa)
            reduced_tdoa = tdoa[valid_idx]
            reduced_anchors = np.delete(ANCHORS, reference_index, axis=0)[valid_idx]
            all_anchors = np.insert(reduced_anchors, reference_index, ANCHORS[reference_index], axis=0)
        else:
            reduced_tdoa = tdoa
            all_anchors = ANCHORS

        # Check if there's enough anchor solving
        if all_anchors.shape[0] < all_anchors.shape[1] + 1:
            estimated_positions.append(np.full((ANCHORS.shape[1],), np.nan))
            continue

        x0 = np.mean(all_anchors, axis=0)

        try:
            res = least_squares(
                tdoa_residual, x0,
                args=(all_anchors, reduced_tdoa, reference_index),
                bounds=([0]*ANCHORS.shape[1], [20]*ANCHORS.shape[1])
            )
            estimated_positions.append(res.x)
        except:
            estimated_positions.append(np.full((ANCHORS.shape[1],), np.nan))

    return np.array(estimated_positions)
