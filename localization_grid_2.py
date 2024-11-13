import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Wedge
import matplotlib.cm as cm

map_size = 50
num_workers = 10
robot_fov = 90
robot_fov_rad = np.deg2rad(robot_fov)
robot_view_distance = 300
robot_position = np.array([map_size / 2, map_size / 2])
robot_orientation = 0
safe_zone = Rectangle((0, 0), map_size, map_size)
worker_positions = np.random.uniform(0, map_size, (num_workers, 2))

def cost_function(candidate_position, robot_orientation, worker_positions, robot_position, weights):
    """
    Calculate the total cost for a candidate robot position.

    Parameters:
    - candidate_position: np.array([x, y])
    - robot_orientation: float (radians)
    - worker_positions: np.array([[x1, y1], [x2, y2], ...])
    - robot_position: np.array([x_current, y_current])
    - weights: dict with weights for each component

    Returns:
    - total_cost: float
    """

    w_visibility = weights['visibility']
    w_proximity = weights['proximity']
    w_movement = weights['movement']
    w_safe_zone = weights['safe_zone']

    total_cost = 0

    visibility_cost = 0
    for worker_pos in worker_positions:

        vec_robot_to_worker = worker_pos - candidate_position
        distance = np.linalg.norm(vec_robot_to_worker)
        angle = np.arctan2(vec_robot_to_worker[1], vec_robot_to_worker[0]) - robot_orientation
        angle = np.mod(angle + np.pi, 2 * np.pi) - np.pi  # Normalize angle to [-pi, pi]

        # Check if worker is within the field of view and view distance
        if np.abs(angle) > robot_fov_rad / 2 or distance > robot_view_distance:
            visibility_cost += 1  # Penalty for each worker not visible

    # Proximity Penalty
    proximity_cost = np.sum(np.linalg.norm(worker_positions - candidate_position, axis=1))

    # Movement Penalty
    movement_cost = np.linalg.norm(candidate_position - robot_position)

    # Safe Zone Penalty
    safe_zone_cost = 0
    if not (0 <= candidate_position[0] <= map_size and 0 <= candidate_position[1] <= map_size):
        safe_zone_cost = 1000  # Large penalty for being outside the safe zone

    # Total Cost
    total_cost = (
        w_visibility * visibility_cost +
        w_proximity * proximity_cost +
        w_movement * movement_cost +
        w_safe_zone * safe_zone_cost
    )

    return total_cost


grid_size = 20  # Number of points along each axis
grid_range = 25  # Range to sample around the robot
x_candidates = np.linspace(robot_position[0] - grid_range, robot_position[0] + grid_range, grid_size)
y_candidates = np.linspace(robot_position[1] - grid_range, robot_position[1] + grid_range, grid_size)
candidate_positions = np.array(np.meshgrid(x_candidates, y_candidates)).T.reshape(-1, 2)

weights = {
    'visibility': 1000,
    'proximity': 2,
    'movement': 1,
    'safe_zone': 1
}

costs = []
for candidate in candidate_positions:
    cost = cost_function(candidate, robot_orientation, worker_positions, robot_position, weights)
    costs.append(cost)
costs = np.array(costs)

# 5. Select the Optimal Position
min_cost_index = np.argmin(costs)
optimal_position = candidate_positions[min_cost_index]

# Visualize the Results
fig, ax = plt.subplots(figsize=(8, 8))
ax.add_patch(Rectangle((0, 0), map_size, map_size, fill=False, edgecolor='black'))
ax.scatter(worker_positions[:, 0], worker_positions[:, 1], c='blue', label='Workers')
scatter = ax.scatter(candidate_positions[:, 0], candidate_positions[:, 1], c=costs, cmap='viridis', alpha=0.6, label='Candidate Positions')
fig.colorbar(scatter, ax=ax, label='Cost')
ax.plot(robot_position[0], robot_position[1], 'ro', label='Robot Start')
ax.plot(optimal_position[0], optimal_position[1], 'go', label='Optimal Position')
fov = Wedge(
    (optimal_position[0], optimal_position[1]),
    robot_view_distance,
    np.rad2deg(robot_orientation - robot_fov_rad / 2),
    np.rad2deg(robot_orientation + robot_fov_rad / 2),
    color='green',
    alpha=0.2
)
ax.add_patch(fov)
ax.set_xlim(0, map_size)
ax.set_ylim(0, map_size)
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_title('Robot Position Optimization')
ax.legend()
ax.grid(True)
plt.show()