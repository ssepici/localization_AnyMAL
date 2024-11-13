import numpy as np
import matplotlib.pyplot as plt

map_size = 20
num_workers = 9
num_obstacles = 2
obstacle_radius = 3
view_radius = 5

alpha = 80  # Weight for visibility
beta = 8    # Weight for proximity
gamma = 2   # Weight for angle of separation

workers = np.random.randint(1, map_size, (num_workers, 2))
obstacles = np.random.randint(1, map_size, (num_obstacles, 2))

def calculate_angle(a, b, c):
    ab = b - a
    ac = c - a
    dot_product = np.dot(ab, ac)
    norm_ab = np.linalg.norm(ab)
    norm_ac = np.linalg.norm(ac)
    angle_rad = np.arccos(dot_product / (norm_ab * norm_ac))
    return np.degrees(angle_rad)

def is_within_view(robot_position, worker_position, obstacles, orientation, fov=180):
    direction_vector = worker_position - robot_position
    angle_to_worker = np.degrees(np.arctan2(direction_vector[1], direction_vector[0]))
    relative_angle = (angle_to_worker - orientation + 360) % 360
    
    if not (relative_angle <= fov or relative_angle >= (360 - fov)):
        return False

    for obstacle in obstacles:
        obstacle_vector = obstacle - robot_position
        distance_to_worker = np.linalg.norm(direction_vector)
        distance_to_obstacle = np.linalg.norm(obstacle_vector)

        if distance_to_obstacle < distance_to_worker and distance_to_obstacle < view_radius:
            angle_to_obstacle = np.degrees(np.arctan2(obstacle_vector[1], obstacle_vector[0]))
            if np.abs(angle_to_worker - angle_to_obstacle) < np.degrees(np.arcsin(obstacle_radius / distance_to_obstacle)):
                return False
    return True

def count_visible_workers(robot_position, workers, obstacles, orientation, fov=180):
    visible_count = 0
    for worker in workers:
        if is_within_view(robot_position, worker, obstacles, orientation, fov):
            visible_count += 1
    return visible_count

def compute_cost(robot_position, workers, obstacles, orientation, centroid, fov=180):
    visible_count = count_visible_workers(robot_position, workers, obstacles, orientation, fov)
    distance_to_centroid = np.linalg.norm(robot_position - centroid)
    
    visible_workers = [worker for worker in workers if is_within_view(robot_position, worker, obstacles, orientation, fov)]
    if len(visible_workers) > 1:
        separation_angles = []
        for i in range(len(visible_workers)):
            for j in range(i + 1, len(visible_workers)):
                angle = calculate_angle(robot_position, visible_workers[i], visible_workers[j])
                separation_angles.append(angle)
        avg_separation_angle = np.mean(separation_angles) if separation_angles else 0
    else:
        avg_separation_angle = 0

    cost = -alpha * visible_count + beta * distance_to_centroid + gamma * avg_separation_angle
    return cost

centroid = np.mean(workers, axis=0)
outside_distance = view_radius + 1  # Place robot slightly beyond view radius

best_position = None
best_orientation = None
min_cost = float('inf')

for angle in range(0, 360, 30):
    test_position = centroid + outside_distance * np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
    test_orientation = (angle + 180) % 360
    cost = compute_cost(test_position, workers, obstacles, test_orientation, centroid)

    if cost < min_cost:
        min_cost = cost
        best_position = test_position
        best_orientation = test_orientation

plt.figure(figsize=(10, 10))
plt.xlim(0, map_size)
plt.ylim(0, map_size)
plt.grid(True)
for i, worker in enumerate(workers, start=1):
    plt.plot(worker[0], worker[1], 'ro', markersize=10, label=f'Worker {i}' if i == 1 else "")
for i, obstacle in enumerate(obstacles, start=1):
    obstacle_circle = plt.Circle(obstacle, obstacle_radius, color='black', alpha=0.5)
    plt.gca().add_patch(obstacle_circle)
    plt.plot(obstacle[0], obstacle[1], 'ks', markersize=10, label='Obstacle' if i == 1 else "")
plt.plot(best_position[0], best_position[1], 'bo', markersize=10, label='Robot')
for angle in [best_orientation - 90, best_orientation + 90]:
    x = best_position[0] + view_radius * np.cos(np.radians(angle))
    y = best_position[1] + view_radius * np.sin(np.radians(angle))
    plt.plot([best_position[0], x], [best_position[1], y], 'b--')
angles = np.linspace(best_orientation - 90, best_orientation + 90, 100)
x = best_position[0] + view_radius * np.cos(np.radians(angles))
y = best_position[1] + view_radius * np.sin(np.radians(angles))
plt.fill_betweenx(y, best_position[0], x, color='blue', alpha=0.1)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Robot Positioning with Larger Obstacles Blocking Visibility")
plt.legend(loc='upper left')
plt.show()