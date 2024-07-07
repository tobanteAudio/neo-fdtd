import json
import numpy as np


def find_third_vertex(A, B):
    A = np.array(A)
    B = np.array(B)

    # Calculate the vector from A to B
    AB = B - A

    # Calculate the distance AB
    d = np.linalg.norm(AB)

    # Normalize the vector AB
    AB_normalized = AB / d

    # Find two vectors perpendicular to AB and to each other
    if AB_normalized[0] != 0 or AB_normalized[1] != 0:
        perp_vector_1 = np.array([-AB_normalized[1], AB_normalized[0], 0])
    else:
        perp_vector_1 = np.array([0, AB_normalized[2], -AB_normalized[1]])

    perp_vector_1 = perp_vector_1 / np.linalg.norm(perp_vector_1)

    perp_vector_2 = np.cross(AB_normalized, perp_vector_1)
    perp_vector_2 = perp_vector_2 / np.linalg.norm(perp_vector_2)

    # Calculate the height of the equilateral triangle
    h = (np.sqrt(3) / 2) * d

    # Calculate the midpoint M
    M = (A + B) / 2

    # Calculate the two possible locations for the third vertex
    C1 = M + h * perp_vector_1
    C2 = M - h * perp_vector_1

    return C1, C2


def transform_point(point, scale, rotation, translation):
    """
    Transform a 3D point by scaling, rotating, and translating.

    :param point: The original 3D point as a numpy array [x, y, z].
    :param scale: Scaling factors as a numpy array [sx, sy, sz].
    :param rotation: Rotation angles in degrees as a numpy array [rx, ry, rz].
    :param translation: Translation vector as a numpy array [tx, ty, tz].
    :return: Transformed 3D point as a numpy array [x', y', z'].
    """
    # Scaling matrix
    S = np.diag(scale)

    # Rotation matrices for X, Y, Z axes
    rx, ry, rz = np.deg2rad(rotation)
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]])
    R_y = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])
    R_z = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])

    # Combined rotation matrix
    R = R_z @ R_y @ R_x

    # Scaling
    scaled_point = S @ point

    # Rotation
    rotated_point = R @ scaled_point

    return list(rotated_point + translation)


def make_receiver_grid(L, W, H, step):
    receivers = []
    for x in range(int(W/step)):
        for y in range(int(L/step)):
            for z in range(int(H/step)):
                receivers.append({
                    "name": "",
                    "xyz": [step/2+x*step, step/2+y*step, step/2+z*step],
                })
    return receivers


def make_box(W, L, H, translate, rotate=None):
    if not rotate:
        rotate = [0, 0, 0]
    points = [
        # Back
        transform_point([0, 0, 0], [W, L, H], [0, 0, 0], translate),
        transform_point([1, 0, 0], [W, L, H], [0, 0, 0], translate),
        transform_point([1, 0, 1], [W, L, H], [0, 0, 0], translate),
        transform_point([0, 0, 1], [W, L, H], [0, 0, 0], translate),

        # Front
        transform_point([0, 1, 0], [W, L, H], [0, 0, 0], translate),
        transform_point([1, 1, 0], [W, L, H], [0, 0, 0], translate),
        transform_point([1, 1, 1], [W, L, H], [0, 0, 0], translate),
        transform_point([0, 1, 1], [W, L, H], [0, 0, 0], translate),
    ]

    triangles = [
        # Back
        [0, 1, 2],
        [0, 2, 3],

        # Front
        [6, 5, 4],
        [7, 6, 4],

        # Left
        [0, 3, 4],
        [7, 4, 3],

        # Right
        [5, 2, 1],
        [2, 5, 6],

        # Bottom
        [0, 5, 1],
        [0, 4, 5],

        # Top
        [2, 6, 3],
        [6, 7, 3],
    ]

    return points, triangles


def main():
    L = 6.0
    W = 3.65
    H = 3.12

    # L = 7.0
    # W = 5.19
    # H = 3.70

    src_height = 1.2
    src_backwall = 1
    src_distance = 1.75
    src_left = [W/2-src_distance/2, L-src_backwall, src_height]
    src_right = [W/2+src_distance/2, L-src_backwall, src_height]

    l1, l2 = find_third_vertex(src_left, src_right)
    listener = l1 if l1[1] < l2[1] else l2

    producer_low = listener.copy()
    producer_low[1] = 1
    producer_low[2] = 0.9

    producer_high = listener.copy()
    producer_high[1] = 1

    offset = np.array([-0.05, -0.05, -1.2])
    stl_ps, stl_ts = make_box(0.1, 0.1, 1.1, src_left+offset)
    str_ps, str_ts = make_box(0.1, 0.1, 1.1, src_right+offset)
    lis_ps, lis_ts = make_box(0.1, 0.1, 1.1, listener+offset)

    p1_ps, p1_ts = make_box(2.5, 0.1, 1.5, [W/2-2.5/2, L-0.3, 0.75])
    p2_ps, p2_ts = make_box(0.1, 2.5, 1.5, [0.2, L-2.5-0.5, 0.75])
    p3_ps, p3_ts = make_box(0.1, 2.5, 1.5, [W-0.1-0.2, L-2.5-0.5, 0.75])
    p4_ps, p4_ts = make_box(2.5, 2, 0.1, [W/2-2.5/2, L-2-0.5, H-0.1-0.3])
    p5_ps, p5_ts = make_box(2.5, 2, 0.1, [W/2-2.5/2, L-2-2.1-0.5, H-0.1-0.3])
    p6_ps, p6_ts = make_box(2.5, 0.1, 1.5, [W/2-2.5/2, 0.3, 0.75])

    sofa_points, sofa_ts = make_box(1.75, 1, 0.5, [W/2-1.75/2, 0.1, 0.1])
    table_ps, table_ts = make_box(2, 0.8, 0.02, [W/2-1, listener[1]+0.4, 0.8])

    model = {
        "mats_hash": {
            "Walls": {
                "tris": [
                    # Back Wall
                    [0, 1, 2],
                    [0, 2, 3],

                    # Front Wall
                    [6, 5, 4],
                    [7, 6, 4],

                    # Left Wall
                    [0, 3, 4],
                    [7, 4, 3],

                    # Right Wall
                    [5, 2, 1],
                    [2, 5, 6],
                ],
                "pts": [
                    # Back Wall
                    [0.0, 0.0, 0.0],
                    [W, 0.0, 0.0],
                    [W, 0.0, H],
                    [0.0, 0.0, H],

                    # Front Wall
                    [0.0, L, 0.0],
                    [W, L, 0.0],
                    [W, L, H],
                    [0.0, L, H],
                ],
                "color": [204, 255, 255],
                "sides": [1, 1, 1, 1, 1, 1, 1, 1]
            },
            "Floor": {
                "tris": [[0, 1, 2], [1, 3, 2]],
                "pts": [
                    [0, 0, 0],
                    [0, L, 0],
                    [W, 0, 0],
                    [W, L, 0]
                ],
                "color": [151, 134, 122],
                "sides": [1, 1]
            },
            "Ceiling": {
                "tris": [[2, 1, 0], [2, 3, 1]],
                "pts": [
                    [0, 0, H],
                    [0, L, H],
                    [W, 0, H],
                    [W, L, H]
                ],
                "color": [204, 204, 204],
                "sides": [1, 1]
            },
            "Panel_1": {
                "tris": p1_ts,
                "pts": p1_ps,
                "color": [111, 55, 10],
                "sides": [2]*len(p1_ts)
            },
            "Panel_2": {
                "tris": p2_ts,
                "pts": p2_ps,
                "color": [111, 55, 10],
                "sides": [2]*len(p2_ts)
            },
            "Panel_3": {
                "tris": p3_ts,
                "pts": p3_ps,
                "color": [111, 55, 10],
                "sides": [2]*len(p3_ts)
            },
            "Panel_4": {
                "tris": p4_ts,
                "pts": p4_ps,
                "color": [111, 55, 10],
                "sides": [2]*len(p4_ts)
            },
            "Panel_5": {
                "tris": p5_ts,
                "pts": p5_ps,
                "color": [111, 55, 10],
                "sides": [2]*len(p5_ts)
            },
            "Panel_6": {
                "tris": p6_ts,
                "pts": p6_ps,
                "color": [111, 55, 10],
                "sides": [2]*len(p6_ts)
            },
            "Table": {
                "tris": table_ts,
                "pts": table_ps,
                "color": [130, 75, 25],
                "sides": [2]*len(table_ts)
            },
            "Sofa": {
                "tris": sofa_ts,
                "pts": sofa_points,
                "color": [25, 25, 25],
                "sides": [2]*len(sofa_ts)
            },
            # "Stand_Left": {
            #     "tris": stl_ts,
            #     "pts": stl_ps,
            #     "color": [25, 25, 25],
            #     "sides": [2]*len(stl_ps)
            # },
            # "Stand_Right": {
            #     "tris": str_ts,
            #     "pts": str_ps,
            #     "color": [25, 25, 25],
            #     "sides": [2]*len(str_ps)
            # },
            # "Listener": {
            #     "tris": lis_ts,
            #     "pts": lis_ps,
            #     "color": [25, 25, 25],
            #     "sides": [2]*len(lis_ps)
            # },
        },
        "sources": [{
            "xyz": src_left,
            "name": "Speaker Left"
        }, {
            "xyz": src_right,
            "name": "Speaker Right"
        }],
        "receivers": [
            {
                "xyz": listener.tolist(),
                "name": "Engineer"
            },
            {
                "xyz": producer_low.tolist(),
                "name": "Producer Low"
            },
            {
                "xyz": producer_high.tolist(),
                "name": "Producer High"
            }],
        # "receivers": make_receiver_grid(L, W, H, step=1.5),
    }

    with open("model_export.json", "w") as file:
        json.dump(obj=model, fp=file)


main()
