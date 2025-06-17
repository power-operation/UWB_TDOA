import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.least_squares import compute_distance, compute_residual


def pytest_target_matches(request, name):
    return request.config.getoption("--target") in (None, name)

# === Test 1: compute_distance ===
def test_compute_distance_known_points():
    p1 = np.array([0, 0])
    p2 = np.array([3, 4])
    dist = compute_distance(p1, p2)
    assert abs(dist - 5.0) < 1e-6, f"Expected 5.0, got {dist}"


# === Test 2: compute_residual ===
def test_compute_residual_zero_when_position_matches():
    anchors = np.array([
        [0, 0],
        [0, 5],
        [5, 0],
        [5, 5]
    ])
    reference_index = 0
    true_position = np.array([3, 3])

    # Constructing an ideal TDOA
    dists = np.linalg.norm(anchors - true_position, axis=1)
    ref_dist = dists[reference_index]
    tdoa = dists - ref_dist
    tdoa = np.delete(tdoa, reference_index)

    # Calculate the residuals
    residual = compute_residual(true_position, anchors, tdoa, reference_index)
    assert residual.shape == tdoa.shape
    assert np.allclose(residual, 0, atol=1e-6), f"Expected near-zero residual, got {residual}"
