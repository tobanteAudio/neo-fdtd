import json

import numpy as np


def load_obj_mesh(obj_file, reverse=False):
    with open(obj_file) as f:
        lines = [line.rstrip() for line in f]

    tris = []
    pts = []

    for line in lines:
        if not "vn " in line:
            if "v " in line:
                parts = line.split(" ")
                parts = parts[1:4]
                parts = [float(part)/1000.0 for part in parts]
                pts.append(parts)
            if "f " in line:
                parts = line.split(" ")
                parts = parts[1:4]
                parts = [int(part.split("/")[0])-1 for part in parts]
                tris.append(parts if not reverse else parts[::-1])

    return pts, tris


def point_along_line(p1, p2, t):
    return [
        p1[0] + (p2[0]-p1[0]) * t,
        p1[1] + (p2[1]-p1[1]) * t,
        p1[2] + (p2[2]-p1[2]) * t,
    ]


def with_x_offset(p: list, offset):
    tmp = p.copy()
    tmp[0] += offset
    return tmp


def with_z(p: list, z):
    tmp = p.copy()
    tmp[2] = z
    return tmp


class ModelBuilder:
    def __init__(self) -> None:
        self.root = {
            "mats_hash": {},
            "sources": [],
            "receivers": []
        }

    def add(self, name, obj_file, color, reverse=False):
        assert name not in self.root
        pts, tris = load_obj_mesh(obj_file, reverse=reverse)
        self.root["mats_hash"][name] = {
            "tris": tris,
            "pts": pts,
            "color": color,
            "sides": [1]*len(tris)
        }

    def add_receiver(self, name, pos):
        self.root["receivers"].append({"name": name, "xyz": pos})

    def add_source(self, name, pos):
        self.root["sources"].append({"name": name, "xyz": pos})

    def write(self, model_file):
        src = np.array(self.root["sources"][0]["xyz"])
        distance_ref = np.linalg.norm(
            src - np.array(self.root["receivers"][0]["xyz"]))
        for r in self.root["receivers"]:
            distance = np.linalg.norm(src - np.array(r["xyz"]))
            print(r["name"], 20*np.log10(distance_ref/distance))

        with open(model_file, "w") as f:
            json.dump(self.root, f)
            print("", file=f)
