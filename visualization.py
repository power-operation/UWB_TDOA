import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from config import ANCHORS
from config import get_anchors

ANCHORS = get_anchors()

# def plot_results(true_positions, estimated_positions, title="TDOA Positioning Results"):
#     if true_positions.shape[1] == 2:
#         plt.figure(figsize=(8, 8))
#         plt.scatter(true_positions[:, 0], true_positions[:, 1], c='blue', label='True Positions', marker='o')
#         plt.scatter(estimated_positions[:, 0], estimated_positions[:, 1], c='red', label='Estimated Positions', marker='x')
#         plt.scatter(ANCHORS[:, 0], ANCHORS[:, 1], c='green', label='Anchors', marker='^')
#         for i in range(len(true_positions)):
#             plt.plot([true_positions[i, 0], estimated_positions[i, 0]],
#                     [true_positions[i, 1], estimated_positions[i, 1]], 'gray', linestyle='dotted', linewidth=0.5)
#         plt.legend()
#         plt.xlabel("X (m)")
#         plt.ylabel("Y (m)")
#         plt.title(title)
#         plt.grid(True)
#         plt.axis("equal")
#         plt.show()

#     else:
#         fig = plt.figure(figsize=(10, 8))
#         ax = fig.add_subplot(111, projection='3d')
#         ax.scatter(*true_positions.T, c='blue', label='True Positions')
#         ax.scatter(*estimated_positions.T, c='red', label='Estimated Positions')
#         ax.scatter(*ANCHORS.T, c='green', label='Anchors')
#         for i in range(len(true_positions)):
#             xs = [true_positions[i, 0], estimated_positions[i, 0]]
#             ys = [true_positions[i, 1], estimated_positions[i, 1]]
#             zs = [true_positions[i, 2], estimated_positions[i, 2]]
#             ax.plot(xs, ys, zs, linestyle='dotted', color='gray', linewidth=0.5)
            
#         ax.set_xlabel("X (m)")
#         ax.set_ylabel("Y (m)")
#         ax.set_zlabel("Z (m)")
#         ax.set_title(f"{title} (3D)")
#         ax.legend()
#         plt.tight_layout()
#         plt.show()


def plot_results(true_positions, estimated_positions, title="TDOA Positioning Results"):
    if true_positions.shape[1] == 2:
        plt.figure(figsize=(8, 8))
        plt.plot(true_positions[:, 0], true_positions[:, 1], 'b-', label='True Trajectory')  # The ground truth
        plt.plot(estimated_positions[:, 0], estimated_positions[:, 1], 'r--', label='Estimated Trajectory')  # The predict trajectory (dotted line)
        plt.scatter(true_positions[:, 0], true_positions[:, 1], c='blue', label='True Positions', marker='o')
        plt.scatter(estimated_positions[:, 0], estimated_positions[:, 1], c='red', label='Estimated Positions', marker='x')
        plt.scatter(ANCHORS[:, 0], ANCHORS[:, 1], c='green', label='Anchors', marker='^')
        for i in range(len(true_positions)):
            plt.plot([true_positions[i, 0], estimated_positions[i, 0]],
                     [true_positions[i, 1], estimated_positions[i, 1]], 'gray', linestyle='dotted', linewidth=0.5)
        plt.legend()
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title(title)
        plt.grid(True)
        plt.axis("equal")
        plt.show()

    else:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(*true_positions.T, 'b-', label='True Trajectory')  # ground truth
        ax.plot(*estimated_positions.T, 'r--', label='Estimated Trajectory')  # The predict trajectory (dotted line)
        ax.scatter(*true_positions.T, c='blue', label='True Positions')
        ax.scatter(*estimated_positions.T, c='red', label='Estimated Positions')
        ax.scatter(*ANCHORS.T, c='green', label='Anchors')
        for i in range(len(true_positions)):
            xs = [true_positions[i, 0], estimated_positions[i, 0]]
            ys = [true_positions[i, 1], estimated_positions[i, 1]]
            zs = [true_positions[i, 2], estimated_positions[i, 2]]
            ax.plot(xs, ys, zs, linestyle='dotted', color='gray', linewidth=0.5)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(f"{title} (3D)")
        ax.legend()
        plt.tight_layout()
        plt.show()