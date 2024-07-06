import json
import numpy as np


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

    p1_pts, p1_tris = make_box(2.5, 0.1, 1.5, [W/2-2.5/2, L-0.3, 0.75])
    p2_pts, p2_tris = make_box(0.1, 2.5, 1.5, [0.2, L-2.5-0.5, 0.75])
    p3_pts, p3_tris = make_box(0.1, 2.5, 1.5, [W-0.1-0.2, L-2.5-0.5, 0.75])
    p4_pts, p4_tris = make_box(2.5, 2.0, 0.1, [W/2-2.5/2, L-2.0-0.5, H-0.1-0.3])
    p5_pts, p5_tris = make_box(2.5, 2.0, 0.1, [W/2-2.5/2, L-2.0-2.1-0.5, H-0.1-0.3])
    p6_pts, p6_tris = make_box(2.5, 0.1, 1.5, [W/2-2.5/2, 0.3, 0.75])
    sofa_points, sofa_tris = make_box(1.75, 1, 0.5, [W/2-1.75/2, 0.1, 0.1])

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
                "tris": p1_tris,
                "pts": p1_pts,
                "color": [111, 55, 10],
                "sides": [2]*len(p1_tris)
            },
            "Panel_2": {
                "tris": p2_tris,
                "pts": p2_pts,
                "color": [111, 55, 10],
                "sides": [2]*len(p2_tris)
            },
            "Panel_3": {
                "tris": p3_tris,
                "pts": p3_pts,
                "color": [111, 55, 10],
                "sides": [2]*len(p3_tris)
            },
            "Panel_4": {
                "tris": p4_tris,
                "pts": p4_pts,
                "color": [111, 55, 10],
                "sides": [2]*len(p4_tris)
            },
            "Panel_5": {
                "tris": p5_tris,
                "pts": p5_pts,
                "color": [111, 55, 10],
                "sides": [2]*len(p5_tris)
            },
            "Panel_6": {
                "tris": p6_tris,
                "pts": p6_pts,
                "color": [111, 55, 10],
                "sides": [2]*len(p6_tris)
            },
            "Sofa": {
                "tris": sofa_tris,
                "pts": sofa_points,
                "color": [25, 25, 25],
                "sides": [2]*len(sofa_tris)
            },
        },
        "sources": [{
            "xyz": [(W/2)-1, L-0.75, 1.2],
            "name": "Speaker Left"
        }, {
            "xyz": [(W/2)+1, L-0.75, 1.2],
            "name": "Speaker Right"
        }],
        "receivers": [{
            "xyz": [W/2, L-2.75, 1.2],
            "name": "Engineer"
        }, {
            "xyz": [(W/3)*1, 0.8, 0.9],
            "name": "Producer Low"
        }, {
            "xyz": [(W/3)*2, 0.8, 0.9],
            "name": "Producer High"
        }],
        # "receivers": make_receiver_grid(L, W, H, step=1.5),
    }

    with open("model_export.json", "w") as file:
        json.dump(obj=model, fp=file)


main()
