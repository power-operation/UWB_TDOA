import numpy as np
from config import ANCHORS

def estimate_positions_fang(tdoa_measurements, reference_index=0):
    """
    Fang algorithm is an analytical TDOA localisation method for the 2D plane. 
    It converts the TDOA into a set of linear equations to be solved analytically.
    """
    estimated_positions = []
    anchors = ANCHORS
    ref_anchor = anchors[reference_index]
    other_anchors = np.delete(anchors, reference_index, axis=0)

    for tdoa in tdoa_measurements:
        if np.any(np.isnan(tdoa)):
            estimated_positions.append(np.array([np.nan, np.nan]))
            continue

        A = []
        b = []
        for i in range(len(tdoa)):
            ai = other_anchors[i] - ref_anchor
            ti = tdoa[i]
            bi = 0.5 * (np.dot(ai, ai) - ti ** 2)
            A.append(ai)
            b.append(bi - ti * np.linalg.norm(ai))
        try:
            A = np.array(A)
            b = np.array(b)
            x = np.linalg.lstsq(A, b, rcond=None)[0] + ref_anchor
            estimated_positions.append(x)
        except:
            estimated_positions.append(np.array([np.nan, np.nan]))
    return np.array(estimated_positions)
