import json
import pathlib


def load_surface_mesh(obj_file):
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
                tris.append(parts)

    return pts, tris


def main():
    model_dir = pathlib.Path("../../data/models/ProStudio")
    ceiling_pts, ceiling_tris = load_surface_mesh(model_dir/'model_ceiling.obj')
    floor_pts, floor_tris = load_surface_mesh(model_dir/'model_floor.obj')
    walls_pts, walls_tris = load_surface_mesh(model_dir/'model_walls.obj')

    root = {
        "mats_hash": {
            "Ceiling": {
                "tris": ceiling_tris,
                "pts": ceiling_pts,
                "color": [2, 2, 2],
                "sides": [1]*len(ceiling_tris)
            },
            "Floor": {
                "tris": floor_tris,
                "pts": floor_pts,
                "color": [2, 2, 2],
                "sides": [1]*len(floor_tris)
            },
            "Walls": {
                "tris": walls_tris,
                "pts": walls_pts,
                "color": [255, 255, 255],
                "sides": [1]*len(walls_tris)
            },
        },
        "sources": [{"name": "S1", "xyz": [0.85, 6.7, 1.4]}],
        "receivers": [
            {"name": "R1", "xyz": [2.9, 5.5, 1.2]},
            {"name": "R2", "xyz": [3.8, 1.0, 1.2]},
        ]
    }

    model_file = model_dir/'model.json'
    with open(model_file, "w") as f:
        json.dump(root, f)


main()
