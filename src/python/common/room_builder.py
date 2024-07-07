from collections import defaultdict
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


def make_box(W, L, H, translate, rotate=None, first_idx=0):
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
        [first_idx+0, first_idx+1, first_idx+2],  # Back
        [first_idx+0, first_idx+2, first_idx+3],

        [first_idx+6, first_idx+5, first_idx+4],  # Front
        [first_idx+7, first_idx+6, first_idx+4],

        [first_idx+0, first_idx+3, first_idx+4],  # Left
        [first_idx+7, first_idx+4, first_idx+3],

        [first_idx+5, first_idx+2, first_idx+1],  # Right
        [first_idx+2, first_idx+5, first_idx+6],

        [first_idx+0, first_idx+5, first_idx+1],  # Bottom
        [first_idx+0, first_idx+4, first_idx+5],

        [first_idx+2, first_idx+6, first_idx+3],  # Top
        [first_idx+6, first_idx+7, first_idx+3],
    ]

    return points, triangles


class RoomBuilder:
    def __init__(self, width, length, height, wall_color=None):
        self.length = length
        self.width = width
        self.height = height
        self.sources = []
        self.receivers = []
        self.boxes = defaultdict(list)
        self.wall_color = wall_color
        self.colors = {}

    def get_color(self, material):
        if material in self.colors:
            return list(self.colors[material])
        return list(np.floor(np.random.rand(3)*255))

    def with_colors(self, colors):
        self.colors = colors
        return self

    def add_source(self, name, position):
        self.sources.append({
            "name": name,
            "xyz": position,
        })
        return self

    def add_receiver(self, name, position):
        self.receivers.append({
            "name": name,
            "xyz": position,
        })
        return self

    def add_box(self, material, size, position, rotation=None):
        self.boxes[material].append({
            "size": size,
            "position": position,
            "rotation": rotation if rotation else [0, 0, 0],
        })
        return self

    def build(self, file_path):
        L = self.length
        W = self.width
        H = self.height

        model = {
            "mats_hash": {
                "Walls": {
                    "tris": [
                        [0, 1, 2],  # Back Wall
                        [0, 2, 3],

                        [6, 5, 4],  # Front Wall
                        [7, 6, 4],

                        [0, 3, 4],  # Left Wall
                        [7, 4, 3],

                        [5, 2, 1],  # Right Wall
                        [2, 5, 6],
                    ],
                    "pts": [
                        [0.0, 0.0, 0.0],  # Back Wall
                        [W, 0.0, 0.0],
                        [W, 0.0, H],
                        [0.0, 0.0, H],

                        [0.0, L, 0.0],  # Front Wall
                        [W, L, 0.0],
                        [W, L, H],
                        [0.0, L, H],
                    ],
                    "color": self.get_color("Walls"),
                    "sides": [1, 1, 1, 1, 1, 1, 1, 1]
                },
                "Ceiling": {
                    "tris": [[2, 1, 0], [2, 3, 1]],
                    "pts": [
                        [0, 0, H],
                        [0, L, H],
                        [W, 0, H],
                        [W, L, H]
                    ],
                    "color": self.get_color("Ceiling"),
                    "sides": [1, 1]
                },
                "Floor": {
                    "tris": [[0, 1, 2], [1, 3, 2]],
                    "pts": [
                        [0, 0, 0],
                        [0, L, 0],
                        [W, 0, 0],
                        [W, L, 0]
                    ],
                    "color": self.get_color("Floor"),
                    "sides": [1, 1]
                },
            },
            "sources": self.sources,
            "receivers": self.receivers,
        }

        for key, boxes in self.boxes.items():
            spec = {
                "tris": [],
                "pts": [],
                "color": self.get_color(key),
                "sides": []
            }

            counter = 0
            for box in boxes:
                size = box["size"]
                pos = box["position"]
                rot = box["rotation"]
                ps, ts = make_box(size[0], size[1], size[2], pos, rot, counter)

                spec["tris"] += ts
                spec["pts"] += ps
                spec["sides"] += [2]*len(ts)

                counter += len(ps)

            model["mats_hash"][key] = spec

        with open(file_path, "w") as file:
            json.dump(model, file, indent=1)
            print("", file=file)
