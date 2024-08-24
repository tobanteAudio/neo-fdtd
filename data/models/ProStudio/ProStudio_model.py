import json
import pathlib

import numpy as np

from pffdtd.sim3d.room_builder import find_third_vertex


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


def main():
    dir = pathlib.Path(".")
    obj = dir/"obj"

    offset = 0.07

    s1 = [3.38789-3.288/2, 6.90-offset, 1.062+0.332]
    sub1 = s1.copy()
    sub1[2] = 0.0 + offset

    s2 = s1.copy()
    s2[0] += 3.288
    sub2 = s2.copy()
    sub2[2] = 0.0 + offset

    s3 = [2.25, 3.55, 0.4]

    r1 = list(find_third_vertex(s1, s2)[1])
    r1[1] += (1.0-0.0)
    r1[2] = 1.2

    r2 = r1.copy()
    r2[1] -= 1.0

    r3 = point_along_line(r2, r1, 0.20)
    r4 = point_along_line(r2, r1, 0.40)
    r5 = point_along_line(r2, r1, 0.60)
    r6 = point_along_line(r2, r1, 0.80)

    # # Couch
    # r2 = r1.copy()
    # r2[1] = 1.2
    # r2[2] = 1.3

    # r3 = with_x_offset(r2, -0.73*1.5)
    # r4 = with_x_offset(r2, -0.73*0.5)
    # r5 = with_x_offset(r2, +0.73*0.5)
    # r6 = with_x_offset(r2, +0.73*1.5)

    m = ModelBuilder()
    m.add("ATC Left", obj / 'atc_left.obj', [5, 5, 5], reverse=True)
    m.add("ATC Right", obj / 'atc_right.obj', [5, 5, 5], reverse=True)
    m.add("Ceiling", obj / 'ceiling.obj', [60, 60, 60])
    m.add("Console", obj / 'console.obj', [60, 60, 60], reverse=True)
    # m.add("Couch", obj / 'couch.obj', [5, 5, 48], reverse=True)
    # m.add("Diffusor", obj / 'diffusor.obj', [53, 33, 0], reverse=True)
    m.add("Floor", obj / 'floor.obj', [53, 33, 0])
    m.add("Rack", obj / 'rack.obj', [25, 25, 25], reverse=True)
    m.add("Raised Floor", obj / 'raised_floor.obj', [25, 25, 25], reverse=True)
    m.add("Walls Back", obj / 'walls_back.obj', [100, 100, 100])
    m.add("Walls Front", obj / 'walls_front.obj', [100, 100, 100])
    m.add("Walls Side", obj / 'walls_side.obj', [180, 180, 180])
    m.add_source("S1", s1)
    m.add_source("S2", s2)
    m.add_source("S3", s3)
    m.add_source("SUB1", sub1)
    m.add_source("SUB2", sub2)
    m.add_receiver("R1", r1)
    m.add_receiver("R2", r2)
    m.add_receiver("R3", r3)
    m.add_receiver("R4", r4)
    m.add_receiver("R5", r5)
    m.add_receiver("R6", r6)
    m.write(dir / "model.json")


main()
