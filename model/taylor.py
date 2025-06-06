import numpy as np
from config import ANCHORS, SPACE_X, SPACE_Y, SPACE_Z

def estimate_positions_taylor(tdoa_measurements, reference_index=0, max_iter=20, tol=1e-4):
    estimated_positions = []
    dim = ANCHORS.shape[1]
    anchors = ANCHORS
    ref_anchor = anchors[reference_index]
    other_anchors = np.delete(anchors, reference_index, axis=0)

    upper_bounds = np.array([SPACE_X, SPACE_Y] + ([SPACE_Z] if dim == 3 else []))

    for tdoa in tdoa_measurements:
        if np.any(np.isnan(tdoa)):
            estimated_positions.append(np.full((dim,), np.nan))
            continue

        # Initial point: centre of target range + random perturbation
        x = np.mean(anchors, axis=0) + np.random.normal(0, 0.5, size=dim)

        for _ in range(max_iter):
            dists = np.linalg.norm(other_anchors - x, axis=1)
            d0 = np.linalg.norm(ref_anchor - x)

            if d0 < 1e-6 or np.any(dists < 1e-6):
                x[:] = np.nan
                break

            H = []
            delta = []
            for i in range(len(tdoa)):
                ai = other_anchors[i]
                ri = dists[i]
                gi = (x - ai) / ri - (x - ref_anchor) / d0
                H.append(gi)
                delta.append(tdoa[i] - (ri - d0))

            H = np.vstack(H)
            delta = np.array(delta)

            try:
                dx = np.linalg.lstsq(H, delta, rcond=None)[0]
            except:
                dx = np.zeros_like(x)

            # Update Location
            x_new = x + dx

            # Diffuse protection: direct abort if far from space boundary
            if np.any(np.abs(x_new) > 10 * upper_bounds):
                x[:] = np.nan
                break

            if np.linalg.norm(dx) < tol:
                break
            x = x_new

        estimated_positions.append(x)
    return np.array(estimated_positions)
