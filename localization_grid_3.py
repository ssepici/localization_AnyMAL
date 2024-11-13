import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

map_size = 50
num_workers = 6
robot_fov = 90 
robot_fov_rad = np.deg2rad(robot_fov)
robot_view_distance = 300
robot_position = np.array([map_size / 2, map_size / 2])
robot_orientation = 0
worker_positions = np.random.uniform(0, map_size, (num_workers, 2))

def cost_function(candidate_position, worker_positions, robot_position, weights):
    """
    Calculate the total cost for a candidate robot position.

    Parameters:
    - candidate_position: np.array([x, y])
    - worker_positions: np.array([[x1, y1], [x2, y2], ...])
    - robot_position: np.array([x_current, y_current])
    - weights: dict with weights for each component

    Returns:
    - total_cost: float
    """
    w_visibility = weights['visibility']
    w_angle_sep = weights['angle_separation']
    w_proximity = weights['proximity']
    w_movement = weights['movement']

    # Visibility Penalty
    visibility_cost = 0
    for worker_pos in worker_positions:
        vec_to_worker = worker_pos - candidate_position
        distance = np.linalg.norm(vec_to_worker)
        angle_to_worker = np.arctan2(vec_to_worker[1], vec_to_worker[0]) - robot_orientation
        angle_to_worker = np.mod(angle_to_worker + np.pi, 2 * np.pi) - np.pi
        if (np.abs(angle_to_worker) > robot_fov_rad / 2) or (distance > robot_view_distance):
            visibility_cost += 1

    # Angle of Separation Penalty
    vecs_to_workers = worker_positions - candidate_position
    angles_to_workers = np.arctan2(vecs_to_workers[:,1], vecs_to_workers[:,0])
    angles_to_workers = np.mod(angles_to_workers + np.pi, 2 * np.pi) - np.pi
    max_angle = np.max(angles_to_workers)
    min_angle = np.min(angles_to_workers)
    angle_separation = max_angle - min_angle
    if angle_separation > np.pi:
        angle_separation = 2 * np.pi - angle_separation
    angle_sep_cost = angle_separation  

    # Proximity Penalty
    distances_to_workers = np.linalg.norm(worker_positions - candidate_position, axis=1)
    proximity_cost = np.sum(distances_to_workers)

    # Movement Penalty
    movement_cost = np.linalg.norm(candidate_position - robot_position)

    # Total Cost
    total_cost = (
        w_visibility * visibility_cost +
        w_angle_sep * angle_sep_cost +
        w_proximity * proximity_cost +
        w_movement * movement_cost
    )

    return total_cost

grid_size = 50  # Number of points along each axis
grid_range = 25  # Range to sample around the robot
x_candidates = np.linspace(robot_position[0] - grid_range, robot_position[0] + grid_range, grid_size)
y_candidates = np.linspace(robot_position[1] - grid_range, robot_position[1] + grid_range, grid_size)
candidate_positions = np.array(np.meshgrid(x_candidates, y_candidates)).T.reshape(-1, 2)


weights = {
    'visibility': 10000.0,
    'angle_separation': 50.0,
    'proximity': 1.0,
    'movement': 0.1
}

costs = []
for candidate in candidate_positions:
    cost = cost_function(candidate, worker_positions, robot_position, weights)
    costs.append(cost)
costs = np.array(costs)

min_cost_index = np.argmin(costs)
optimal_position = candidate_positions[min_cost_index]

# Visualize the Results
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, map_size)
ax.set_ylim(0, map_size)
ax.scatter(worker_positions[:, 0], worker_positions[:, 1], c='blue', label='Workers')
scatter = ax.scatter(candidate_positions[:, 0], candidate_positions[:, 1], c=costs, cmap='viridis', alpha=0.6, label='Candidate Positions')
cbar = fig.colorbar(scatter, ax=ax, label='Cost')
cbar.ax.set_ylabel('Cost', rotation=270, labelpad=15)
ax.plot(optimal_position[0], optimal_position[1], 'go', label='Optimal Position')


# Draw the robot's field of view at the optimal position
# For visualization, we'll set the orientation to point towards the average worker position
avg_worker_direction = np.mean(np.arctan2(worker_positions[:,1] - optimal_position[1], worker_positions[:,0] - optimal_position[0]))
fov = Wedge(
    (optimal_position[0], optimal_position[1]),
    robot_view_distance,
    np.rad2deg(avg_worker_direction - robot_fov_rad / 2),
    np.rad2deg(avg_worker_direction + robot_fov_rad / 2),
    color='green',
    alpha=0.2
)
ax.add_patch(fov)
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_title('Robot Position Optimization with Visibility, Angle of Separation, and Proximity')
ax.grid(True)
plt.show()