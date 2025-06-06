import numpy as np
from config import ANCHORS

def estimate_positions_chan(tdoa_measurements, reference_index=0):
    """
    Chan algorithm is used for closed-form solution of TDOA localisation with good robustness. 
    It is suitable for 2D scenarios where the number of base stations is greater than 3.
    """
    estimated_positions = []
    anchors = ANCHORS
    ref_anchor = anchors[reference_index]
    other_anchors = np.delete(anchors, reference_index, axis=0)

    for tdoa in tdoa_measurements:
        if np.any(np.isnan(tdoa)):
            estimated_positions.append(np.array([np.nan, np.nan]))
            continue

        M = len(tdoa)
        A = np.zeros((M, 2))
        b = np.zeros(M)
        for i in range(M):
            xi, yi = other_anchors[i]
            x0, y0 = ref_anchor
            ri = tdoa[i]
            A[i] = [xi - x0, yi - y0]
            b[i] = 0.5 * (xi**2 + yi**2 - x0**2 - y0**2 - ri**2)

        try:
            x = np.linalg.lstsq(A, b, rcond=None)[0]
            estimated_positions.append(x)
        except:
            estimated_positions.append(np.array([np.nan, np.nan]))
    return np.array(estimated_positions)
