import numpy as np
from shapely.geometry import Polygon, LineString, Point
from scipy.spatial import ConvexHull
import math
from math import atan2, pi

SPLINE_DEG = 3  # The spline's degree, should be 3.

SPI_FACTOR = 1  # Obstacle inflation factor for smooth path planing (try to keep it 1)

VISIBILITY_INFLATE_FACTOR = 0.3  # inflation factor for the algorithms extra gap, used for more smoothness.
CONTROL_POINTS_MAX_DIST_FACTOR = 2.25  # Multiplayer of the inflation factor for CONTROL_POINTS_MAX_DIST. UNPROVEN: If less than 2, no rotating will be needed.
CONTROL_POINTS_MAX_DIST = CONTROL_POINTS_MAX_DIST_FACTOR * VISIBILITY_INFLATE_FACTOR * SPI_FACTOR  # The maximal distance allowed between control points during construction

CONTROL_POINTS_MIN_DIST = 10 ** -3  # Minimal distance between control points. If 2 control points are getting too close during rotation, we merge them.

ROTATE_STEP = 1  # Rotation step size in degrees.

POINT_DISTANCE_FROM_OBS_FACTOR = 0.4  # step size for a previous points to go away from the obstacle in case of collision.

FLATTEN_LEVEL_MAX = 3  # how much will we take the centered points toward the original point when flattening a path at the end


def distance(point1, point2):
    """
    returns the Euclidean distance between two points.

    Args:
        point1: coordinates of the first point.
        point2: coordinates of the second point.

    Returns:
        the Euclidean distance between two points.
    """
    return round(np.sqrt(np.sum(np.square(point2 - point1))), 4)


def normalize_vector(vector, to_length=1):
    """
    normalizes the vector to be of length to_length
    Args:
        vector: the vector to normalize
        to_length: the required length of the normalized vector

    Returns:
        A vector with the same direction as the given vector, and length of to_length.
    """

    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector  # Avoid division by zero
    return (vector / norm) * to_length


def inflate_hull(hull, inflate_by):
    """
    inflates the given hull by inflate_by in each direction.
    NOTE: the inflate_by should be the required distance of the inflated corners from the original vertices.

    Args:
        hull: a convex hull to inflate
        inflate_by: by how much to inflate the hull away from the original vertices?

    Returns:
        The inflated convex hull
    """
    points = hull.points
    new_points = []
    for point in points:
        defined_1 = False
        adj_point1 = []
        adj_point2 = []
        for simplex in hull.simplices:
            for i in range(2):
                if points[simplex[i]][0] == point[0] and points[simplex[i]][1] == point[1]:

                    if not defined_1:
                        defined_1 = True
                        adj_point1 = points[simplex[1 - i]]
                    else:
                        adj_point2 = points[simplex[1 - i]]

            # Calculate vectors

        vector1 = np.array(adj_point1) - point
        vector2 = np.array(adj_point2) - point

        norm_vector1 = normalize_vector(vector1)
        norm_vector2 = normalize_vector(vector2)

        # Calculate the sum of vectors
        sum_vector = norm_vector1 + norm_vector2

        # Normalize sum_vector to have length n
        normalized_vector = normalize_vector(sum_vector, inflate_by)
        new_points.append(point - normalized_vector)

    return ConvexHull(new_points)


def unite_intersecting_hulls(hulls):
    """
    gets a list of convex hull that my intersect each other and
    return a list of convex hulls after uniting each intersecting hulls.
    keeps uniting the intersecting hulls over and over until none intersect.
    Args:
        hulls: A list of convex hulls

    Returns:
        A list of convex hulls after uniting intersecting hulls from the given list until there were no intersections.
    """
    def unite_intersecting_hulls_once(hulls):
        """
        gets a list of convex hulls and unite each intersecting hulls of the original list.
        Args:
            hulls: a list of convex hulls

        Returns:
            a list of the same convex hulls after uniting each intersecting hulls ones.
        """
        class UnionFind:
            """
            a classic union find data structure.
            """
            def __init__(self, size):
                self.parent = list(range(size))
                self.rank = [0] * size

            def find(self, x):
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]

            def union(self, x, y):
                rootX = self.find(x)
                rootY = self.find(y)
                if rootX != rootY:
                    if self.rank[rootX] > self.rank[rootY]:
                        self.parent[rootY] = rootX
                    elif self.rank[rootX] < self.rank[rootY]:
                        self.parent[rootX] = rootY
                    else:
                        self.parent[rootY] = rootX
                        self.rank[rootX] += 1

        def hulls_intersect(hull1, hull2):
            """
            returns whether the two given convex hulls intersect each other.
            Args:
                hull1: a convex hull
                hull2: a convex hull

            Returns:
                True if the two hulls intersect each other, False otherwise.

            """
            poly1 = Polygon(hull1.points[hull1.vertices])
            poly2 = Polygon(hull2.points[hull2.vertices])
            return poly1.intersects(poly2)

        def unite_hull(points_list):
            """
            gets list containing np.arrays of points and return the convex hull of all the points.
            Args:
                points_list: list of np.arrays of points

            Returns:
                the convex hull of all the points in the lists.
            """
            all_points = np.vstack(points_list)
            return ConvexHull(all_points)

        def filter_hulls(hulls_to_filter):
            """
            takes a list of convex hull objects and filters the points
            of each hull to include only those that lie on the convex hull boundary.
            It returns a list of new convex hull objects created from these filtered points.
            Args:
                hulls_to_filter: A list of convex hulls.

            Returns:
                a list of new convex hull objects created from these filtered points.
            """
            filtered_hulls = []

            for hull in hulls_to_filter:
                points = np.array(hull.points)
                hull_indices = hull.vertices
                hull_points = points[hull_indices]

                # Filter points that are on the convex hull
                filtered_points = []
                for point in points:
                    if np.any(np.all(np.isclose(hull_points, point), axis=1)):
                        filtered_points.append(point)

                filtered_hull = ConvexHull(np.array(filtered_points))
                filtered_hulls.append(filtered_hull)

            return filtered_hulls

        num_hulls = len(hulls)
        uf = UnionFind(num_hulls)

        for i in range(num_hulls):
            for j in range(i + 1, num_hulls):
                if hulls_intersect(hulls[i], hulls[j]):
                    uf.union(i, j)

        from collections import defaultdict
        groups = defaultdict(list)
        for i in range(num_hulls):
            root = uf.find(i)
            groups[root].append(hulls[i].points)

        hulls = [unite_hull(group) for group in groups.values()]

        return filter_hulls(hulls)

    prev_hulls = []
    hulls = unite_intersecting_hulls_once(hulls)
    while len(prev_hulls) != len(hulls):
        prev_hulls = hulls
        hulls = unite_intersecting_hulls_once(hulls)

    return hulls


def rotate_point(axis_point, rot_point, deg):
    """
    Rotate a point rot_point around axis_point by deg degrees clockwise.

    Parameters:
    axis_point (tuple): The (x, y) coordinates of the axis point.
    rot_point (tuple): The (x, y) coordinates of the point to rotate.
    deg (float): The angle in degrees to rotate clockwise.

    Returns:
    tuple: The (x, y) coordinates of the rotated point.
    """
    # Convert degrees to radians
    rad = math.radians(-deg)  # negative for clockwise rotation

    # Translate the point to the origin
    trans_x = rot_point[0] - axis_point[0]
    trans_y = rot_point[1] - axis_point[1]

    # Perform the rotation
    rot_x = trans_x * math.cos(rad) - trans_y * math.sin(rad)
    rot_y = trans_x * math.sin(rad) + trans_y * math.cos(rad)

    # Translate the point back
    final_x = rot_x + axis_point[0]
    final_y = rot_y + axis_point[1]

    return final_x, final_y, axis_point[2]


def vector_angle(v1, v2):
    """
    Calculate the angle between two vectors in degrees.

    Parameters:
    v1 (array-like): The first vector.
    v2 (array-like): The second vector.

    Returns:
    float: The angle between v1 and v2 in degrees.
    """
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    dot_product = np.dot(v1, v2)
    cos_theta = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
    angle = np.arccos(cos_theta) * (180 / np.pi)
    return angle


def get_convex_hull_polygon(points):
    """
    Given a set of points, this function computes the convex hull and
    returns a Shapely Polygon object representing the convex hull.

    Args:
        points: a set of points.

    Returns:
        A Shapely Polygon object representing the convex hull.
    """
    # Compute the convex hull
    hull = ConvexHull(points)

    # Extract the vertices of the convex hull
    hull_points = [points[vertex] for vertex in hull.vertices]

    # Create and return a Shapely polygon from the hull points
    convex_hull_polygon = Polygon(hull_points)
    return convex_hull_polygon


def align_collinear(p1, p2, p3):
    """
    Rotate p3 around p2 such that p1, p2, and p3 are collinear.

    Args:
    p0 (tuple): The first point.
    p1 (tuple): The second point (the center of rotation).
    p3 (tuple): The third point (the point to rotate).

    Returns:
    tuple: The (x, y, z) coordinates of the rotated point.
    """
    # Convert points to numpy arrays
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)

    # Vectors
    v0 = p2 - p1
    v3 = p3 - p2

    # Calculate angle to rotate
    angle = vector_angle(v0, v3)

    # Determine the direction of rotation
    cross_product = np.cross(v0, v3)
    if cross_product[2] < 0:  # Check the z-component of the cross product
        angle = -angle

    # Rotate p3 around p1 in the z-plane
    return rotate_point(p2, p3, angle)


def get_next_control_points_index(points, obs_hulls, idx):
    """
    Returns the next relevant control point to the control point of index idx.
    prefer to return the latest control visible control point from the point at idx,
    if there is no visible control point after idx, returns the last point.
    Args:
        points: a list of the control points
        obs_hulls: list of the convex hulls of the obstacles
        idx: the index of the current points.

    Returns:
        the index of the next relevant control points according to the rules described.

    """
    for i in range(len(points) - 1, idx, -1):
        if intersects_any_obs(obs_hulls, [points[idx], points[i]]) is None:
            return i

    return len(points) - 1


def is_point_in_any_hull(point, obs_hulls):
    """
    Check if a 3D point (considering only its x and y coordinates) is inside any of the given convex hulls.

    Args:
        point: the point.
        obs_hulls: convex hulls

    Returns:
        bool: True if the point is inside any of the convex hulls, False otherwise.
    """
    # Extract x and y coordinates from the 3D point
    point_2d = Point(point[0], point[1])

    # Polygonise hulls
    polygons = [get_convex_hull_polygon(hull.points) for hull in obs_hulls]

    # Check if the 2D point is inside any of the polygons
    for polygon in polygons:
        if polygon.contains(point_2d):
            return True

    return False


def convex_hull_intersects(obs, points):
    """
    Check if the convex hull of the given points intersects with the given polygon.
    Also consider it as an intersection if any point is inside the polygon.

    Args:
        obs (Polygon): The polygon to check for intersection, given as a 3D polygon with the same z value.
        points (np.array or list): Array of four 3D points to form the convex hull.

    Returns:
        bool: True if the convex hull of the points intersects with the polygon, or if any point is inside the polygon,
        False otherwise.
    """
    # Convert points to a NumPy array if it's a list
    if isinstance(points, list):
        points = np.array(points)

    # Extract 2D coordinates (x, y) from the 3D points
    points_2d = points[:, :2]

    # Convert the 3D polygon `obs` to a 2D Polygon
    if isinstance(obs, Polygon):
        obs_2d = get_convex_hull_polygon([(x, y) for x, y in obs.exterior.coords])
    else:
        raise ValueError("The `obs` parameter should be a shapely Polygon.")

    # Check if any point is inside the polygon
    def any_point_inside(points, polygon):
        return any(polygon.contains(Point(p)) for p in points)

    if any_point_inside(points_2d, obs_2d):
        return True

    # Compute convex hull
    try:
        hull = ConvexHull(points_2d)
        hull_points = points_2d[hull.vertices]
    except:
        # Handle case when points are collinear or cannot form a valid convex hull
        # Treat the points as a line segment
        line = LineString([points_2d[0], points_2d[-1]])
        return line.intersects(obs_2d)

    # Create a polygon for the convex hull
    hull_polygon = get_convex_hull_polygon(hull_points)

    # Check for intersection
    return hull_polygon.intersects(obs_2d)


def intersects_any_obs(obs_hulls, points, inflated_by=0):
    """
    If the convex hull of the given points intersect with any of the obstacles, return the obstacle
    that intersect with the points.
    Inflate_by for inflating the obstacles before checking for intersection.
    inflate_by = 0 means don't infalte the obstacles.
    Args:
        obs_hulls: list of the convex hulls of teh obstacles.
        points: list of points to check if convex hull of those points intersect with any of the obstacles.
        inflated_by: by how much to infalte the obstacles. default = 0.

    Returns:
        If the convex hull of the given points intersect with any of the obstacles, return the obstacle
        that intersect with the points. else returns None.

    """
    obs_hulls_copy = obs_hulls.copy()
    if inflated_by != 0:
        for i in range(len(obs_hulls_copy)):
            obs_hulls_copy[i] = inflate_hull(obs_hulls_copy[i], inflated_by)

    for hull in obs_hulls_copy:
        obs_polygon = get_convex_hull_polygon(hull.points)
        if convex_hull_intersects(obs_polygon, points):
            return hull
    return None


def remove_redundant_points(points, obs_hulls):
    """
    check for any point in points if removing the point will result in 4 sequential point's convex hull
    intersect with any obstacle. If it doesn't, remove the point.
    goes from first to last.
    Args:
        points: list of the points.
        obs_hulls: list of the convex hulls of the obstacles.

    Returns:
        the list of points after removing the removable points.
    """
    def is_points_removable(points, idx):
        """
        returns True if for the point in points at index idx, removing the point won't result in 4 sequential point's
        convex hull. False otherwise.
        uses remove_redundant_points's obs_hulls.
        Args:
            points: the list of the points.
            idx: the index of the points we want to remove.

        Returns:
            True if for the point in points at index idx, removing the point won't result in 4 sequential point's
            convex hull. False otherwise.

        """
        removable = True
        for i in range(max(0, idx - 3), idx):
            low_idx = max(0, i)
            high_idx = min(len(points) - 1, i + 4)

            checking_points = points[low_idx:high_idx + 1]
            checking_points = np.delete(checking_points, idx - i, axis=0)
            if intersects_any_obs(obs_hulls, checking_points):
                removable = False
        return removable

    points_len = len(points)
    i = points_len - 2

    while i > 0:
        if is_points_removable(points, i):
            points = np.delete(points, i, axis=0)
            i -= 1
        else:
            i -= 1
        points_len = len(points)
    return points


def flatten_points(points, obs_hulls):
    """
    Given a list of points, checks of every point if we can move it into the center of mass of it
    with its neighbor points. if it cant, give more weight to the current point and check's again. continue checking
    until the point's weight reach weight FLATTEN_LEVEL_MAX, or until moving the point into thi center of mass will
    result in a legal configuration of points where for each group of 4 sequential point, the convex hull of
    these points doesn't intersect with any of the obstacles. If we don't reach such legal configuration
    of points before giving the current points weight of FLATTEN_LEVEL_MAX, we don't move this points at all.
    goes from start to finish (second points to second-to-last point).

    Args:
        points: list of the points
        obs_hulls:  list of the convex hulls of the obstacles.

    Returns:
        list of the points after flattening the way described.

    """
    i = 1
    while i < len(points) - 1:
        for flattening_level in range(FLATTEN_LEVEL_MAX):
            centered_point = (points[i - 1] + points[i + 1] + flattening_level * points[i]) / (2 + flattening_level)
            points_candidate = np.copy(points)
            points_candidate[i] = centered_point
            legal_change = True
            for j in range(max(0, i - 3), i + 1):
                low_idx = max(0, j)
                high_idx = min(len(points_candidate) - 1, j + 3)
                checking_points = points_candidate[low_idx:high_idx + 1]
                if intersects_any_obs(obs_hulls, checking_points):
                    legal_change = False
                    break

            if legal_change:
                points[i] = centered_point
                break
        i += 1

    return points


def add_minimal_points(control_points, k=SPLINE_DEG):
    """
    adds the minimal amount of points required to crate a B-spline of degree k
    Args:
        control_points: the current list of control points
        k: the required degree of the b spline

    Returns:
        list of control points after adding the minimal amount of points required to crate a B-spline of degree k,
        points will be added (if needed) uniformly.
    """

    if len(control_points) < k + 1:
        num_of_sections = len(control_points) - 1
        num_of_added_points = (k + 1) - len(control_points)
        base = num_of_added_points // num_of_sections
        remainder = num_of_added_points % num_of_sections
        i = 0
        c_points_len = len(control_points[i])
        while i < c_points_len - 1:
            to_add = base + 1
            if remainder > 0:
                to_add += 1
                remainder -= 1

            # Define control points for the B-spline
            new_points = np.array([control_points[i] * (1 - j / (to_add)) + (j / (to_add)) * control_points[i + 1]
                                   for j in range(1, to_add)])
            control_points = np.concatenate((control_points[:i + 1], new_points, control_points[i + 1:]))
            c_points_len = len(control_points[i])
            i += 1 + len(new_points)

    return control_points


def is_point_safe_to_update(c_points, obs_hulls, new_point, idx):
    """
    checks if changing the point in index of idx in c_points to new_points will keep the rule:
    'for each group of 4 sequential point, the convex hull of these points doesn't intersect with any of the obstacles'.
    if so, returns true, else returns false
    Args:
        c_points: list of the points.
        obs_hulls: list of the convex hulls of the obstacles.
        new_point: the points we want to check.
        idx: the index of the point we want to replace in c_points.

    Returns:
        True if for each group of 4 sequential point, the convex hull of these points doesn't intersect with any of the obstacles.
        False otherwise.
    """
    updated_c_points = c_points.copy()
    updated_c_points[idx] = new_point
    point_squads = []

    end_idx = len(updated_c_points)
    for i in range(max(idx - 3, 0), idx):
        if i + 4 < end_idx:
            point_squads.append(
                [updated_c_points[i], updated_c_points[i + 1], updated_c_points[i + 2], updated_c_points[i + 3]])

    for point_squad in point_squads:
        if intersects_any_obs(obs_hulls, point_squad):
            return False
    return True


def get_visibility(start_point, dest_point, obs_hulls):
    """
    Get the visibility of a the set of points start_point U dest_point U obs_hulls's points,
    in relation of obs_hulls.
    Args:
        start_point: the start point of the route
        dest_point: the destination point of the route.
        obs_hulls: list of the convex hulls of the obstacles.

    Returns:
        list of tuples, each of the nested lists contains 2 points that are visible to each other in relation to
        obs_hulls.

    """
    def intersect(a, b, c, d):
        """
        Check if two line segments intersect.

        The function determines whether the line segment defined by points `a` and `b`
        intersects with the line segment defined by points `c` and `d`. It uses a combination
        of the counterclockwise (ccw) test and a collinearity check to verify intersection.

        Args:
            a: A tuple representing the first endpoint of the first line segment (x1, y1).
            b: A tuple representing the second endpoint of the first line segment (x2, y2).
            c: A tuple representing the first endpoint of the second line segment (x3, y3).
            d: A tuple representing the second endpoint of the second line segment (x4, y4).

        Returns:
            bool: `True` if the line segments intersect, `False` otherwise.

        Notes:
            - The function returns `False` for collinear segments even if they overlap.
            - Intersection is determined strictly for non-collinear segments.
        """

        def ccw(a, b, c):
            """Check if three points A, B, and C are in counterclockwise order."""
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

        def collinear(a, b, c):
            """Check if three points A, B, and C are in collinear order."""
            return (c[1] - a[1]) * (b[0] - a[0]) == (b[1] - a[1]) * (c[0] - a[0])

        if collinear(a, c, d) or collinear(b, c, d) or collinear(a, b, c) or collinear(a, b, d):
            return False
        return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)

    def is_in_a_hull(point_to_check, hulls):
        """
           Check if a point is inside any of a given set of convex hulls.

           This function determines whether a given point lies inside any of the convex hulls
           in a provided list. Each hull is defined by a set of points, and the function uses
           a helper method to check if the point lies inside the polygon formed by the hull.

           Args:
               point_to_check: A tuple representing the point to check (x, y).
               hulls: A list of convex hulls, where each hull is represented by an object
                      containing a `points` attribute (an array of 2D points).

           Returns:
               bool: `True` if the point is inside any of the convex hulls, `False` otherwise.
        """
        def point_inside_polygon(point, polygon):
            """
            Determines whether a given point lies inside a polygon.

            Args:
                point: A tuple representing the point to check (x, y).
                polygon: A list of tuples representing the vertices of the polygon
                         (each tuple is a 2D coordinate).

            Returns:
                bool: `True` if the point is inside the polygon, `False` otherwise.
            """
            x, y = point[0], point[1]
            n = len(polygon)
            inside = False

            # Sort the polygon vertices in counterclockwise order
            sorted_polygon = sorted(polygon, key=lambda p: (atan2(p[1] - y, p[0] - x) + 2 * pi) % (2 * pi))

            p1x, p1y = sorted_polygon[0]
            for i in range(n + 1):
                p2x, p2y = sorted_polygon[i % n]
                if y > min(p1y, p2y):
                    if y <= max(p1y, p2y):
                        if x <= max(p1x, p2x):
                            if p1y != p2y:
                                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or x <= xinters:
                                inside = not inside
                p1x, p1y = p2x, p2y

            return inside

        for hull in hulls:
            if point_to_check in hull.points:
                continue
            # Check if the point is inside the convex hull
            if point_inside_polygon(point_to_check, hull.points):
                return True

        return False

    # Compute and plot visibility graph
    inflated_hulls = []
    for obs_hull in obs_hulls:
        inflated_hulls.append(inflate_hull(obs_hull, VISIBILITY_INFLATE_FACTOR))

    inflated_hulls = unite_intersecting_hulls(inflated_hulls)

    # Populate the dictionary with obstacle types for each point
    point_to_obs_hull = {}
    inf_all_points = []
    for obs_hull in inflated_hulls:
        for point in obs_hull.points:
            inf_all_points.append(point)
            point_to_obs_hull[tuple(point)] = obs_hull
    inf_all_points.extend([start_point, dest_point])

    inf_all_points = np.array(inf_all_points)

    if is_in_a_hull(start_point, inflated_hulls) or is_in_a_hull(dest_point, inflated_hulls):
        raise ValueError("there will be no solution")

    visibility = []
    obs_lines = []
    for hull in inflated_hulls:

        points = hull.points

        for simplex in hull.simplices:
            obs_lines.append([points[simplex[0]], points[simplex[1]]])

            vis_line = ([points[simplex[0]], points[simplex[1]]])
            vis_inv_line = ([points[simplex[1]], points[simplex[0]]])
            visibility.append(vis_line)
            visibility.append(vis_inv_line)

    for i in range(len(inf_all_points)):
        if is_in_a_hull(inf_all_points[i], inflated_hulls):
            continue
        for j in range(i + 1, len(inf_all_points)):
            if is_in_a_hull(inf_all_points[j], inflated_hulls):
                continue
            line = [inf_all_points[i], inf_all_points[j]]
            inv_line = [inf_all_points[j], inf_all_points[i]]
            intersects = False

            points_tuples = [tuple(point) for point in line]
            start_point_tuple = [tuple(start_point)]
            no_start_or_dest = True
            for point in points_tuples:
                if ((point[0] == start_point[0] and point[1] == start_point[1])
                        or (point[0] == dest_point[0] and point[1] == dest_point[1])):
                    no_start_or_dest = False

            are_same_obs = (no_start_or_dest and
                            point_to_obs_hull.get(tuple(inf_all_points[i])) ==
                            point_to_obs_hull.get(tuple(inf_all_points[j])))
            if not are_same_obs:
                for obs_line in obs_lines:
                    if intersect(line[0], line[1], obs_line[0], obs_line[1]):
                        intersects = True
                if not intersects:
                    visibility.append(line)
                    visibility.append(inv_line)

    return inf_all_points, visibility


def crate_g(all_points, visibility):
    """
    Create a graph representation from points and visibility lines.

    This function constructs a graph where nodes represent points, and edges represent visible connections between
    the points. Each edge is assigned a weight equal to the Euclidean distance between the connected points.

    Args:
        all_points: A 2D numpy array of shape (n, 2), where each row represents a point (x, y).
        visibility: A list of tuples, where each tuple contains two points (point_a, point_b) representing a visible
        line segment.

    Returns:
        dict: A dictionary representing the graph, where:
            - Keys are indices of points in `all_points`.
            - Values are lists of tuples `(neighbor_idx, distance)` indicating the index of the connected point
              and the Euclidean distance to it.
    """
    g = {}
    for line in visibility:
        point_a = line[0]
        point_b = line[1]

        point_a_idx = np.where((all_points[:, 0] == point_a[0]) & (all_points[:, 1] == point_a[1]))[0][0]
        point_b_idx = np.where((all_points[:, 0] == point_b[0]) & (all_points[:, 1] == point_b[1]))[0][0]

        if point_a_idx not in g:
            g[point_a_idx] = []
        if point_b_idx not in g:
            g[point_b_idx] = []

        dist = distance(point_a, point_b)
        g[point_a_idx].append((point_b_idx, dist))
    return g

def plan_a_star(all_points, given_g, start, dest):
    """
    Plan a path using the A* algorithm.

    This function implements the A* pathfinding algorithm to find the shortest path between a start point
    and a destination point in a graph. The graph is represented by points (`all_points`) and
    their connections (`given_g`), with the heuristic calculated as the Euclidean distance to the destination.

    Args:
        all_points: A 2D numpy array of shape (n, 2), where each row represents a point (x, y).
        given_g: A dictionary representing the graph, where:
            - Keys are point indices (as in `all_points`).
            - Values are lists of tuples `(neighbor_idx, distance)` indicating neighbors and distances.
        start: An integer representing the index of the starting point in `all_points`.
        dest: An integer representing the index of the destination point in `all_points`.

    Returns:
        list: A list of indices representing the shortest path from `start` to `dest` in `all_points`.
    """
    def calc_h(all_points, dest_point):
        """
        Calculate heuristic values for A* algorithm.

        This function computes the heuristic values (`h`) for all points relative to a destination point using the Euclidean distance. The result is a dictionary mapping each point's index to its heuristic value.

        Args:
            all_points: A 2D numpy array of shape (n, 2), where each row represents a point (x, y).
            dest_point: An integer representing the index of the destination point in `all_points`.

        Returns:
            dict: A dictionary where:
                - Keys are indices of points in `all_points`.
                - Values are the heuristic values (Euclidean distances) from the point to the destination point.
        """
        a_star_h = {}
        for point in all_points:
            point_idx = np.where((all_points[:, 0] == point[0]) & (all_points[:, 1] == point[1]))[0][0]
            a_star_h[point_idx] = distance(point, all_points[dest_point])
        return a_star_h

    def get_next_from_open(open_a_star, g, h):
        """
        Determine the next point to process in the A* algorithm by finding the point with the lowest `f = g + h`.

        Args:
            open_a_star: A list of currently open points (indices).
            g: A dictionary of actual distances from the start point.
            h: A dictionary of heuristic distances to the destination.

        Returns:
            int: The index of the point with the lowest `f`.
        """
        min_f = open_a_star[0]
        for vertex in open_a_star:
            if h[min_f] + g[min_f] > h[vertex] + g[vertex]:
                min_f = vertex
        open_a_star.remove(min_f)
        return min_f

    h = calc_h(all_points, dest)
    g = {}
    prev = {}

    open_a_star = []
    close_a_star = []

    open_a_star.append(start)
    g[start] = 0

    current_ver = get_next_from_open(open_a_star, h, g)
    prev_current_ver = start
    while current_ver != dest:
        if current_ver not in close_a_star:
            for adj in given_g[current_ver]:
                if adj[0] not in close_a_star:
                    if adj[0] not in open_a_star:
                        # new vertex
                        open_a_star.append(adj[0])
                        g[adj[0]] = g[current_ver] + adj[1]
                        prev[adj[0]] = current_ver

                    else:
                        # already been to this vertex
                        if g[current_ver] + adj[1] < g[adj[0]]:
                            g[adj[0]] = g[current_ver] + adj[1]
                            prev[adj[0]] = current_ver

        close_a_star.append(current_ver)
        prev_current_ver = current_ver
        current_ver = get_next_from_open(open_a_star, g, h)

    prev[dest] = prev_current_ver
    inverted_path = [dest]
    next_to_add = prev[dest]
    while next_to_add != start:
        inverted_path.append(next_to_add)
        next_to_add = prev[next_to_add]

    inverted_path.append(start)

    path = inverted_path[::-1]
    return path
