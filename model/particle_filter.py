import numpy as np
from config import ANCHORS, SPACE_X, SPACE_Y, SPACE_Z

def estimate_positions_pf(tdoa_measurements, reference_index=0, num_particles=500, iterations=5):
    """
    Particle filtering is used for TDOA localisation and is suitable for dealing with noisy or non-linear scenes. 
    Monte Carlo sampling is used for position estimation.
    """
    anchors = ANCHORS
    ref_anchor = anchors[reference_index]
    estimated_positions = []
    other_anchors = np.delete(anchors, reference_index, axis=0)

    for tdoa in tdoa_measurements:
        if np.any(np.isnan(tdoa)):
            estimated_positions.append(np.array([np.nan, np.nan]))
            continue

        # Initialising the particle swarm
        # particles = np.random.uniform([0, 0], [SPACE_X, SPACE_Y], size=(num_particles, 2))
        dim = ANCHORS.shape[1]

        if dim == 2:
            particles = np.random.uniform([0, 0], [SPACE_X, SPACE_Y], size=(num_particles, 2))
        else:
            particles = np.random.uniform([0, 0, 0], [SPACE_X, SPACE_Y, SPACE_Z], size=(num_particles, 3))


        for _ in range(iterations):
            distances = np.linalg.norm(particles[:, np.newaxis, :] - anchors[np.newaxis, :, :], axis=2)
            d0 = distances[:, reference_index][:, np.newaxis]
            tdoa_particles = distances - d0
            tdoa_particles = np.delete(tdoa_particles, reference_index, axis=1)

            # Calculate particle weights
            diff = tdoa_particles - tdoa
            # weights = np.exp(-np.sum(diff ** 2, axis=1) / 0.05)
            # weights /= np.sum(weights) + 1e-9
            weights = np.exp(-np.sum(diff ** 2, axis=1) / 0.05)

            if np.all(weights == 0) or np.isnan(weights).any():
                weights = np.ones(num_particles) / num_particles  # equal weight
            else:
                weights /= np.sum(weights)


            # Resampling + Scrambling
            indices = np.random.choice(np.arange(num_particles), size=num_particles, p=weights)
            particles = particles[indices] + np.random.normal(0, 0.5, size=particles.shape)

        mean_est = np.mean(particles, axis=0)
        estimated_positions.append(mean_est)
    return np.array(estimated_positions)
