import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from scipy.spatial import ConvexHull
import math

import path_planning.SPA_utils as utils


def plan_smooth_path(raw_obstacles, start_point, dest_point, minimal_distance_allowed=0.15):
    """
    Creates a smooth path according to the algorithm:

    The Algorithm:
        input: list of obstacles, starting point and destination point.
        output: B-spline describing the path from the start point to the destination point.
        the return spline is ensured to not encounter any obstacle.

        why of work (simplified, full description is in the report):
        Note: when mentioning the "rule" we mean that for every 4 sequential points in the control points,
        the convex hull of these points does not intersect any obstacle.
        Due to the convex hull property of B splines, this rule means that for sure the spline wont intersect
        any of the obstacles.

        1.  initialize a set of control points using A* star algorithm ob the visibility graph.
        2.  add control points on the path of the A* uniformly s.t. the maximum distance of 2 control points
            is not greater than CONTROL_POINTS_MAX_DIST.
        3.  goes from the 4th point to the last point, for each point do:
            3.1.    If the convex hull created by the last 4 points (indexed i-3 : i) intersect with any obstacles:
                3.1.1.  for the 3 points prior to the current point, only if it doesn't break the rule, move each point
                        away from the intersecting obstacle by POINT_DISTANCE_FROM_OBS_FACTOR. this allows us to satisfy
                        the rule by rotating the current point less in 3.1.2.
                3.1.2.  rotate the current point around its previous point left and right until the rule is satisfied.
                        if the rule is not satisfied after rotating 180 degrees to each side, move the point closer
                        to its previous point and try again do that over and over until the rule is satisfied (or until
                        the current point is too close to the previous point, in that case we put the current point on top
                        of the previous point).


            3.2.    Look for the next point: the next point the preferred to be the latest point (with the greatest
                    index, closest to the destination) that is visible to the current point w.r.t the obstacles.
                    If the is no such point, the next point is set to be the destination point. remove all the points
                    between the current point and the chosen next point.
            3.3     If the distance between the current point to the next point found in 3.2 is greater than
                    CONTROL_POINTS_MAX_DIST, then add points between the current point and the next point uniformly
                    so the maximal distance between 2 points is CONTROL_POINTS_MAX_DIST.
        4.  repeat until this has no effect:
            4.1.    For every point, from start to end (second, to second to last, to be specific), check if the
                    current point can be moved to the middle of it with its neighbor points, with increasing weight
                    to the current point (weights from 0 to FLATTEN_LEVEL_MAX). If it can be moved there without
                    breaking the rule, move it the middle with the least weight to the current point possible.
            4.2     For every point, from start to end, check if the point can be removed without breaking the rule.
                    if it can, remove it.
        5.  create and return the normalized B-spline of degree 3 according to the control points acquired.
    Args:
        raw_obstacles: list of the obstacles, each obstacle is a np.array of points, the vertices of the obstacle.
        start_point: the start point of the route (2D)
        dest_point: the destination points of the route (3D)
        minimal_distance_allowed: by how much to inflate the obstacles before runnning the algorithm.

    Returns:
        A B-spline of degree 3 describing the smooth path returned by the algorithm.

    """

    def smooth_obs_avoiding_path(c_points, obs_hulls):
        """
         the part of the algorithm responsible for parts 3,4,5 described.
        Args:
            c_points: the starting list of control points
            obs_hulls: a list of the convex hulls of the obstacles.

        Returns:
            A B-spline of degree 3 describing the smooth path returned by the algorithm.

        """
        # remove control points that are inside an obstacle.
        new_control_points = []
        for point in c_points:
            if not utils.is_point_in_any_hull(point, obs_hulls):
                new_control_points.append(point)
        c_points = np.array(new_control_points)

        inflated_obs_hulls = [utils.inflate_hull(hull, inflate_by=utils.VISIBILITY_INFLATE_FACTOR * utils.SPI_FACTOR) for hull in
                              obs_hulls]

        # region 3
        cp_len = len(c_points)
        i = 3
        while i < cp_len:

            curr_points = [c_points[i - 3], c_points[i - 2], c_points[i - 1], c_points[i]]

            next_rot = utils.ROTATE_STEP

            obs_intersecting = utils.intersects_any_obs(obs_hulls, curr_points)
            if obs_intersecting:
                c_points[i] = utils.align_collinear(c_points[i - 2], c_points[i - 1], c_points[i])

                # region 3.1.1
                for j in range(i - 1, i - 4, -1):
                    '''
                    this for loop is in charge of taking prevoius points away from the obstacle in case of clission
                    in order for the rotating point to rotate less, resulting in a smoother path.
                    '''
                    distancing_point = c_points[j][0:2]

                    visible_intersecting_obs_points = [obs_point for obs_point in obs_intersecting.points
                                                       if not utils.convex_hull_intersects(
                            utils.get_convex_hull_polygon(obs_intersecting.points),
                            np.array([distancing_point, (obs_point * 0.9 + distancing_point * 0.1)]))]

                    if not visible_intersecting_obs_points:
                        visible_intersecting_obs_points = [obs_point for obs_point in obs_intersecting.points]

                    move_point_by = utils.normalize_vector(
                        np.sum(np.array([utils.normalize_vector([distancing_point - obs_point],
                                                                 utils.POINT_DISTANCE_FROM_OBS_FACTOR) for obs_point in
                                         visible_intersecting_obs_points]), axis=0),
                        utils.POINT_DISTANCE_FROM_OBS_FACTOR)

                    move_point_by = np.append(move_point_by, 0)  # add z axis to move_point_by

                    if j != 0 and utils.is_point_safe_to_update(c_points, obs_hulls, c_points[j] + move_point_by, j):
                        c_points[j] = c_points[j] + move_point_by
                # endregion

                curr_points = [c_points[i - 3], c_points[i - 2], c_points[i - 1], c_points[i]]
                force_stop_rotation = False
                tot_rot = 0

                # region 3.1.2
                while ((not force_stop_rotation)
                       and ((utils.intersects_any_obs(obs_hulls, curr_points))
                            or (utils.is_point_in_any_hull(c_points[i], inflated_obs_hulls)))):
                    '''
                    This while loop is responsible for rotating the current point around the prev point in case of collision
                    '''
                    c_points[i] = utils.rotate_point(c_points[i - 1], c_points[i], next_rot)
                    tot_rot += next_rot
                    # Update next_rot so it goes in alternate direction growing by ROTATE_STEP each time
                    next_rot = -next_rot
                    if next_rot < 0:
                        next_rot -= utils.ROTATE_STEP
                    else:
                        next_rot += utils.ROTATE_STEP

                    # Update the polygone of the lat 4 control points.
                    curr_points = [c_points[i - 3], c_points[i - 2], c_points[i - 1], c_points[i]]

                    if math.fabs(tot_rot) >= 180:
                        c_points[i] = 0.5 * c_points[i - 1] + 0.5 * c_points[i]
                        c_points[i] = utils.align_collinear(c_points[i - 2], c_points[i - 1], c_points[i])
                        next_rot = utils.ROTATE_STEP
                        tot_rot = 0
                        if utils.distance(c_points[i - 1], c_points[i]) <= utils.CONTROL_POINTS_MIN_DIST:
                            force_stop_rotation = True
                            c_points[i] = c_points[i - 1]
                # endregion


            # create new control points if needed
            if i != cp_len - 1:
                # If we have a next point (the current point wasn't the last one):

                # region 3.2
                next_relevant_point_idx = utils.get_next_control_points_index(c_points, inflated_obs_hulls, i)
                c_points = np.delete(c_points, range(i + 1, next_relevant_point_idx), axis=0)
                # endregion

                # region 3.3
                if utils.distance(c_points[i], c_points[i + 1]) > utils.CONTROL_POINTS_MAX_DIST:
                    i_point = c_points[i]
                    i_plus_1_point = c_points[i + 1]
                    m = int(np.ceil(utils.distance(i_point, i_plus_1_point) / utils.CONTROL_POINTS_MAX_DIST))
                    # Define control points for the B-spline
                    new_points = np.array([i_point * (1 - 1 / m) + i_plus_1_point * (1 / m)])

                    c_points = np.concatenate((c_points[:i + 1], new_points, c_points[i + 1:]))
                # endregion
            cp_len = len(c_points)
            i += 1

        # endregion

        # region 4
        old_c_points = np.array([])
        while not np.array_equal(old_c_points, c_points):
            old_c_points = np.copy(c_points)
            c_points = utils.flatten_points(c_points, obs_hulls)  # 4.1
            c_points = utils.remove_redundant_points(c_points, obs_hulls)  # 4.2
        # endregion

        # region 5
        # Degree of the spline (cubic)
        k = utils.SPLINE_DEG

        # makes sure there ara enough control points for a spline:
        c_points = utils.add_minimal_points(c_points, k)


        # ===== Create Normalized SPL! =====

        n = len(c_points)  # number of control points

        # Step 1: Calculate chord lengths (Euclidean distances between consecutive points)
        distances = np.linalg.norm(np.diff(c_points, axis=0), axis=1)

        # Step 2: Calculate cumulative chord lengths and normalize to [0, 1]
        cumulative_lengths = np.cumsum(distances)
        cumulative_lengths = np.insert(cumulative_lengths, 0, 0)  # Start with 0
        normalized_lengths = cumulative_lengths / cumulative_lengths[-1]

        # Step 3: Create the knot vector
        # Degree p, there should be n + p + 1 knots
        p = utils.SPLINE_DEG
        knots = np.zeros(n + p + 1)

        # First p+1 knots are 0
        knots[:p + 1] = 0

        # Middle knots are spread according to normalized cumulative lengths
        num_internal_knots = n - p - 1  # The number of internal knots should be n - p - 1
        knots[p + 1:p + 1 + num_internal_knots] = normalized_lengths[1:num_internal_knots + 1]  # Assign internal knots

        # Last p+1 knots are 1
        knots[-p - 1:] = 1

        # Generate the B-spline
        spl = BSpline(knots, c_points, utils.SPLINE_DEG)
        return spl

        # endregion

    def generate_start_control_points(obs_hulls, start_point, dest_point):
        """
        Generate the initial control points according to parts 1,2 of the algorithm.

        Args:
            obs_hulls: the obstacles we wish to avoid
            start_point: the starting point of the drone
            dest_point: the destination point of the drone

        Returns:
            set of control points created using a_star algorithm, with maximal distance of CONTROL_POINTS_MAX_DIST from
            each other.
        """

        all_inflated_points, visibility = utils.get_visibility(start_point, dest_point, obs_hulls)
        g = utils.crate_g(all_inflated_points, visibility)

        # possibly need to find index of points
        start_idx = \
            np.where((all_inflated_points[:, 0] == start_point[0]) & (all_inflated_points[:, 1] == start_point[1]))[0][
                0]
        dest_point_idx = \
            np.where((all_inflated_points[:, 0] == dest_point[0]) & (all_inflated_points[:, 1] == dest_point[1]))[0][0]
        path_points_idx = utils.plan_a_star(all_inflated_points, g, start_idx, dest_point_idx)
        path_points_2d = [all_inflated_points[point_idx] for point_idx in path_points_idx]

        path_points = [np.array([x, y, 0.0]) for x, y in path_points_2d]

        control_points = []

        for i in range(len(path_points) - 1):
            curr_point = path_points[i]
            next_point = path_points[i + 1]

            m = int(np.ceil(utils.distance(curr_point, next_point) / utils.CONTROL_POINTS_MAX_DIST))

            # Define control points for the B-spline
            new_points = np.array([curr_point * (1 - i / m) + (i / m) * next_point for i in range(m)])
            control_points.extend(new_points)

        control_points.append(path_points[len(path_points) - 1])

        return control_points

    obs_hulls = [ConvexHull(obs) for obs in raw_obstacles]

    obs_hulls = utils.unite_intersecting_hulls(obs_hulls)

    obs_hulls = [utils.inflate_hull(raw_obs, minimal_distance_allowed) for raw_obs in obs_hulls]

    obstacles = utils.unite_intersecting_hulls(obs_hulls)

    start_point = np.array(start_point)
    dest_point = np.array(dest_point)

    control_points = generate_start_control_points(obs_hulls, start_point, dest_point)

    spl = smooth_obs_avoiding_path(control_points, obstacles)

    return spl
