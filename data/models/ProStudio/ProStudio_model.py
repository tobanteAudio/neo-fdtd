import json
import pathlib

import numpy as np

from pffdtd.sim3d.room_builder import find_third_vertex


def load_mesh(obj_file, reverse=False):
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


def main():
    dir = pathlib.Path(".")

    ceiling_pts, ceiling_tris = load_mesh(dir / 'model_ceiling.obj')
    floor_pts, floor_tris = load_mesh(dir / 'model_floor.obj')
    walls_back_pts, walls_back_tris = load_mesh(dir / 'model_walls_back.obj')
    walls_front_pts, walls_front_tris = load_mesh(
        dir / 'model_walls_front.obj')
    walls_side_pts, walls_side_tris = load_mesh(dir / 'model_walls_side.obj')

    diffusor_pts, diffusor_tris = load_mesh(
        dir / 'model_diffusor.obj', reverse=True)
    atc_left_pts, atc_left_tris = load_mesh(
        dir / 'model_atc_left.obj', reverse=True)
    atc_right_pts, atc_right_tris = load_mesh(
        dir / 'model_atc_right.obj', reverse=True)
    couch_pts, couch_tris = load_mesh(dir / 'model_couch.obj', reverse=True)
    rack_pts, rack_tris = load_mesh(dir / 'model_rack.obj', reverse=True)
    raised_floor_pts, raised_floor_tris = load_mesh(
        dir / 'model_raised_floor.obj', reverse=True)
    console_pts, console_tris = load_mesh(
        dir / 'model_console.obj', reverse=True)

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

    # r1 = [3.58, 5.5, 1.2]
    # r2 = [3.58, 0.7+offset, 1.2]

    r2 = r1.copy()
    r2[1] = 1.2
    r2[2] = 1.3

    # r3 = point_along_line(r2, r1, 0.1)
    # r4 = point_along_line(r2, r1, 0.2)
    # r5 = point_along_line(r2, r1, 0.3)

    r3 = with_x_offset(r2, -0.73*1.5)
    r4 = with_x_offset(r2, -0.73*0.5)
    r5 = with_x_offset(r2, +0.73*0.5)
    r6 = with_x_offset(r2, +0.73*1.5)

    # r3 = with_z(r2, 1.0)
    # r4 = with_z(r2, 1.5)
    # r5 = with_z(r2, 2.1)

    root = {
        "mats_hash": {
            "ATC Left": {
                "tris": atc_left_tris,
                "pts": atc_left_pts,
                "color": [5, 5, 5],
                "sides": [1]*len(atc_left_tris)
            },
            "ATC Right": {
                "tris": atc_right_tris,
                "pts": atc_right_pts,
                "color": [5, 5, 5],
                "sides": [1]*len(atc_right_tris)
            },
            "Ceiling": {
                "tris": ceiling_tris,
                "pts": ceiling_pts,
                "color": [60, 60, 60],
                "sides": [1]*len(ceiling_tris)
            },
            "Console": {
                "tris": console_tris,
                "pts": console_pts,
                "color": [60, 60, 60],
                "sides": [1]*len(console_tris)
            },
            # "Couch": {
            #     "tris": couch_tris,
            #     "pts": couch_pts,
            #     "color": [5, 5, 48],
            #     "sides": [1]*len(couch_tris)
            # },
            # "Diffusor": {
            #     "tris": diffusor_tris,
            #     "pts": diffusor_pts,
            #     "color": [53, 33, 0],
            #     "sides": [1]*len(diffusor_tris)
            # },
            "Floor": {
                "tris": floor_tris,
                "pts": floor_pts,
                "color": [53, 33, 0],
                "sides": [1]*len(floor_tris)
            },
            "Rack": {
                "tris": rack_tris,
                "pts": rack_pts,
                "color": [25, 25, 25],
                "sides": [1]*len(rack_tris)
            },
            "Raised Floor": {
                "tris": raised_floor_tris,
                "pts": raised_floor_pts,
                "color": [25, 25, 25],
                "sides": [1]*len(raised_floor_tris)
            },
            "Walls Back": {
                "tris": walls_back_tris,
                "pts": walls_back_pts,
                "color": [100, 100, 100],
                "sides": [1]*len(walls_back_tris)
            },
            "Walls Front": {
                "tris": walls_front_tris,
                "pts": walls_front_pts,
                "color": [100, 100, 100],
                "sides": [1]*len(walls_front_tris)
            },
            "Walls Side": {
                "tris": walls_side_tris,
                "pts": walls_side_pts,
                "color": [180, 180, 180],
                "sides": [1]*len(walls_side_tris)
            },
        },
        "sources": [
            {"name": "S1", "xyz": s1},
            {"name": "S2", "xyz": s2},
            {"name": "S3", "xyz": s3},
            {"name": "SUB1", "xyz": sub1},
            {"name": "SUB2", "xyz": sub2},
        ],
        "receivers": [
            {"name": "R1", "xyz": r1},
            {"name": "R2", "xyz": r2},
            {"name": "R3", "xyz": r3},
            {"name": "R4", "xyz": r4},
            {"name": "R5", "xyz": r5},
            {"name": "R6", "xyz": r6},
        ]
    }

    src = np.array(root["sources"][0]["xyz"])
    distance_ref = np.linalg.norm(src - np.array(root["receivers"][0]["xyz"]))
    for r in root["receivers"]:
        distance = np.linalg.norm(src - np.array(r["xyz"]))
        print(r["name"], 20*np.log10(distance_ref/distance))

    model_file = dir / 'model.json'
    with open(model_file, "w") as f:
        json.dump(root, f)


main()
