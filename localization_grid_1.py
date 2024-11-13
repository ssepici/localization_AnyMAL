import numpy as np
import matplotlib.pyplot as plt

map_size = 30
num_workers = 5
workers = np.random.randint(1, map_size, (num_workers, 2))
view_radius = 8

alpha = 10
beta = 1
gamma = 5

def calculate_angle(a, b, c):
    ab = b - a
    ac = c - a
    dot_product = np.dot(ab, ac)
    norm_ab = np.linalg.norm(ab)
    norm_ac = np.linalg.norm(ac)
    angle_rad = np.arccos(dot_product / (norm_ab * norm_ac))
    return np.degrees(angle_rad)

# Function to check if a worker is within the robot's 180-degree field of view
def is_within_view(robot_position, worker_position, orientation, fov=180):
    direction_vector = worker_position - robot_position
    angle_to_worker = np.degrees(np.arctan2(direction_vector[1], direction_vector[0]))
    relative_angle = (angle_to_worker - orientation + 360) % 360  # Normalize to 0-360 degrees
    return relative_angle <= fov or relative_angle >= (360 - fov)

# Function to count visible workers for a given position and orientation
def count_visible_workers(robot_position, workers, orientation, fov=180):
    visible_count = 0
    for worker in workers:
        if is_within_view(robot_position, worker, orientation, fov):
            visible_count += 1
    return visible_count

# Function to compute the cost for a given robot position and orientation
def compute_cost(robot_position, workers, orientation, centroid, fov=180):
    # 1. Calculate visible workers count
    visible_count = count_visible_workers(robot_position, workers, orientation, fov)
    
    # 2. Calculate distance to centroid
    distance_to_centroid = np.linalg.norm(robot_position - centroid)
    
    # 3. Calculate angle of separation among visible workers (if more than one visible)
    visible_workers = [worker for worker in workers if is_within_view(robot_position, worker, orientation, fov)]
    if len(visible_workers) > 1:
        separation_angles = []
        for i in range(len(visible_workers)):
            for j in range(i + 1, len(visible_workers)):
                angle = calculate_angle(robot_position, visible_workers[i], visible_workers[j])
                separation_angles.append(angle)
        avg_separation_angle = np.mean(separation_angles) if separation_angles else 0
    else:
        avg_separation_angle = 0  # No separation angle if 0 or 1 visible workers

    # Total cost with weights
    cost = -alpha * visible_count + beta * distance_to_centroid + gamma * avg_separation_angle
    return cost

# Calculate the centroid of worker positions
centroid = np.mean(workers, axis=0)
outside_distance = view_radius + 2  # Slightly beyond visibility radius

# Search for the optimal position and orientation to minimize cost
best_position = None
best_orientation = None
min_cost = float('inf')

for angle in range(0, 360, 15):
    test_position = centroid + outside_distance * np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
    test_orientation = (angle + 180) % 360  # Orient toward the centroid
    cost = compute_cost(test_position, workers, test_orientation, centroid)

    if cost < min_cost:
        min_cost = cost
        best_position = test_position
        best_orientation = test_orientation

# Visualization
plt.figure(figsize=(10, 10))
plt.xlim(0, map_size)
plt.ylim(0, map_size)
plt.grid(True)

# Plot workers
for i, worker in enumerate(workers, start=1):
    plt.plot(worker[0], worker[1], 'ro', markersize=10, label=f'Worker {i}' if i == 1 else "")

# Plot robot position and 180-degree viewing area
plt.plot(best_position[0], best_position[1], 'bo', markersize=10, label='Robot')
for angle in [best_orientation - 90, best_orientation + 90]:
    x = best_position[0] + view_radius * np.cos(np.radians(angle))
    y = best_position[1] + view_radius * np.sin(np.radians(angle))
    plt.plot([best_position[0], x], [best_position[1], y], 'b--')

# Shade viewing area
angles = np.linspace(best_orientation - 90, best_orientation + 90, 100)
x = best_position[0] + view_radius * np.cos(np.radians(angles))
y = best_position[1] + view_radius * np.sin(np.radians(angles))
plt.fill_betweenx(y, best_position[0], x, color='blue', alpha=0.1)

# Labels and Legend
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Robot Positioning Optimized by Cost Function")
plt.legend(loc='upper left')

plt.show()