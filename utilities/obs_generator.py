import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import ConvexHull

FRAME = [np.array([[-4, -2], [-4, 2] , [-5, 0]]),
         np.array([[-4, 2] , [4, 2]  , [0, 3]]),
         np.array([[4, 2]  , [4, -2] , [5, 0]]),
         np.array([[4, -2] , [-4, -2], [0, -3]])]

def challenge_map1():
    '''
    Use: 
        START = [-3, 0] 
        END   = [3, 0]
    '''
    return [np.array([[-2, -2], [-1,-2], [-1, 0.5], [-2, 0.5]]),
            np.array([[2, 2], [1,2], [1, -0.5], [2, -0.5]])]

def challenge_map2():
    '''
    Use: 
        START = [-3, 1] 
        END   = [-3, -1]
    '''
    return [np.array([[-10, 0.75], [3, 0], [-10, -0.75]])]

def challenge_map3():
    '''
    Use: 
        START = [-1, -1.5] 
        END   = [1, -1.5]
    '''
    return [np.array([[-0.5, -2], [-0.5, -1], [0.5, -1], [0.5, -2]]),
            np.array([[-1.5, -1], [-1.5 ,1], [1.5, 1], [1.5, -1]])]

def challenge_map4():
    '''
    Use: 
        START = [-3, 0] 
        END   = [3, 0]
    '''
    return [np.array([[-2, -2], [-1,-2], [-1, 0.5], [-2, 0.5]]),
            np.array([[2, 2], [1,2], [1, -0.5], [2, -0.5]])]

def challenge_map5():
    '''
    Use: 
        START = [-3, 0] 
        END   = [3, 0]
    '''
    return [np.array([[-2, -2], [-1,-2], [-1, 0.5], [-2, 0.5]]),
            np.array([[2, 2], [1,2], [1, -0.5], [2, -0.5]])]



def generate_obstacles(seed, number_of_obs):
    np.random.seed(seed)
    obstacles = []

    for _ in range(number_of_obs):
        shape_type = np.random.choice(['rectangle', 'square', 'triangle', 'irregular'])
        x, y = np.random.uniform(-4, 4), np.random.uniform(-2, 2)
        if shape_type == 'rectangle':
            width, height = np.random.uniform(-1, 1, 2)
            obstacle = np.array([
                [x, y],
                [x + width, y],
                [x + width, y + height],
                [x, y + height]
            ])
        elif shape_type == 'square':
            size = np.random.uniform(-1, 1)
            obstacle = np.array([
                [x, y],
                [x + size, y],
                [x + size, y + size],
                [x, y + size]
            ])
        elif shape_type == 'triangle':
            obstacle = np.array([
                [x, y],
                [x + np.random.uniform(0.5, 2.5), y],
                [x + np.random.uniform(0.5, 2.0), y + np.random.uniform(0.5, 2.5)]
            ])
        elif shape_type == 'irregular':
            obstacle = np.array([
                [x, y],
                [x + np.random.uniform(0.5, 2.5), y + np.random.uniform(0.5, 1.5)],
                [x + np.random.uniform(1.0, 2.5), y + np.random.uniform(1.5, 2.5)],
                [x + np.random.uniform(0.5, 1.5), y + np.random.uniform(1.0, 2.5)]
            ])

        obstacles.append(obstacle)

    return obstacles


def plot_obstacles(obstacles, start, end, spl = None):

    if spl is not None:
        t = spl.t
        k = spl.k
        control_points = spl.c
        # Evaluate the B-spline at 100 equally spaced points in the parameter range
        t_new = np.linspace(t[k], t[-k - 1], 100)
        bspline_points = spl(t_new)

        # Plotting
        plt.figure(figsize=(8, 6))

        # Plot B-spline
        plt.plot(bspline_points[:, 0], bspline_points[:, 1], label='B-spline', color='blue')

        # Plot control points
        plt.scatter(control_points[:, 0], control_points[:, 1], color='black', label='Control Points')

    obs_hulls = [ConvexHull(obs) for obs in obstacles]
    for hull in obs_hulls:
        points = hull.points

        # Draw the obstacles
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], color='b')

    plt.scatter(start[0], start[1], color='green')
    plt.scatter(end[0], end[1], color='red')
    # Labels and legend
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.xlim([-4, 4])
    plt.ylim([-2, 2])
    plt.title('Cubic B-spline and Polygons in 2D Space')
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.show()