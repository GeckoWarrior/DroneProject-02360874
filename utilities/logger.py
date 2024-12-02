import numpy as np
import json_numpy
import os
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.interpolate import BSpline


class Logger:

    def __init__(self, obstacles=None, spl=None):
        
        # state

        self.obstacles = obstacles
        self.spl = spl
        
        # record
        self.pos_s = []
        self.aim_s = []
        self.cmd_s = []

    def add(self, pos, aim, cmd):
        self.pos_s.append(pos)
        self.aim_s.append(aim)
        self.cmd_s.append(cmd)

    def set_spline(self, spline):
        self.spl = spline

    def set_obstacles(self, obs):
        self.obstacles = obs

    def save_as_file(self, time=0):
        data = {
            "obstacles": self.obstacles,
            "spl":  [self.spl.t, self.spl.c, self.spl.k],
            "positions": self.pos_s,
            "aims": self.aim_s,
            "commands": self.cmd_s
        }

        directory = 'logs'
        file_path = os.path.join(directory, f'flight_{int(time)}.log')
        os.makedirs(directory, exist_ok=True)

        with open(file_path, 'w') as file:
            json_numpy.dump(data, file, indent=4)

    def read_from_file(self, path):
        with open(path, 'r') as file:
            json_str = file.read()

        data = json_numpy.loads(json_str)
        self.obstacles = data["obstacles"] if ("obstacles" in data and data["obstacles"]) else None
        self.spl = BSpline(data["spl"][0], data["spl"][1], data["spl"][2]) if ("spl" in data and data["spl"]) else None
        self.pos_s = data["positions"] if ("positions" in data and data["positions"]) else []
        self.aim_s = data["aims"] if ("aims" in data and data["aims"]) else []
        self.cmd_s = data["commands"] if ("commands" in data and data["commands"]) else []

    def show_data(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # the obstacle part
        if self.obstacles is not None:
            for obstacle in self.obstacles:
                hull = ConvexHull(obstacle)
                obs_2d = obstacle[hull.vertices]
                z_base = 0
                z_top = 1.5
                build_len = obs_2d.shape[0]
                obs_3d_base = np.hstack([obs_2d, np.full((build_len, 1), z_base)])
                obs_3d_top = np.hstack([obs_2d, np.full((build_len, 1), z_top)])
                building_3d = np.vstack([obs_3d_base, obs_3d_top])

                # Scatter plot for the base and top points
                ax.scatter(building_3d[:build_len, 0], building_3d[:build_len, 1], building_3d[:build_len, 2],
                           color='b')
                ax.scatter(building_3d[build_len:, 0], building_3d[build_len:, 1], building_3d[build_len:, 2],
                           color='g')

                # Draw lines for the walls (connecting each point on the base with the corresponding point on the top)
                for i in range(build_len):
                    ax.plot([building_3d[i, 0], building_3d[i + build_len, 0]],
                            [building_3d[i, 1], building_3d[i + build_len, 1]],
                            [building_3d[i, 2], building_3d[i + build_len, 2]], 'r-')

                # Draw edges for the base and top
                for i in range(build_len):
                    next_i = (i + 1) % build_len
                    ax.plot([building_3d[i, 0], building_3d[next_i, 0]],
                            [building_3d[i, 1], building_3d[next_i, 1]],
                            [building_3d[i, 2], building_3d[next_i, 2]], 'b--')  # Base edges
                    ax.plot([building_3d[i + build_len, 0], building_3d[next_i + build_len, 0]],
                            [building_3d[i + build_len, 1], building_3d[next_i + build_len, 1]],
                            [building_3d[i + build_len, 2], building_3d[next_i + build_len, 2]], 'g--')  # Top edges

        # the spline part
        if self.spl is not None:
            control_points = self.spl.c
            k = self.spl.k
            t = self.spl.t

            bspline_points = self.spl(np.linspace(t[k], t[-k - 1], 100))
            ax.plot(bspline_points[:, 0], bspline_points[:, 1], bspline_points[:, 2], label='B-spline', color='blue')

            # the control points
            cp = np.array(control_points)
            ax.scatter(cp[:, 0], cp[:, 1], cp[:, 2], label='Control points', color='black')

        # the positions

        if 0 < len(self.pos_s) == len(self.pos_s):
            pos_s = np.array(self.pos_s)
            ax.plot(pos_s[:, 0], pos_s[:, 1], pos_s[:, 2], label='positions', color='green')

        # the planned
        if len(self.aim_s) > 0:
            planned = np.array(self.aim_s)
            ax.plot(planned[:, 0], planned[:, 1], planned[:, 2], label='planned', color='orange', markersize=2)

        #  epilog
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        ax.set_xlim([-4, 4])
        ax.set_ylim([-2, 2])
        ax.set_zlim([0, 2])

        ax.set_box_aspect((2, 1, 0.5))

        handles, labels = ax.get_legend_handles_labels()
        if labels:
            plt.legend()

        plt.show()
