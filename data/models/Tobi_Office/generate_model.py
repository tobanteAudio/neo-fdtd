import json


def main():
    L = 7.0
    W = 5.19
    H = 3.70
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
                "color": [0, 204, 204],
                "sides": [1, 1]
            },
        },
        "sources": [{
            "xyz": [W/2, L-1, 1.2],
            "name": "Speaker"
        }],
        "receivers": [{
            "xyz": [W/2, L/2, 1.2],
            "name": "Listener"
        }],
    }

    with open("model_export.json", "w") as file:
        json.dump(model, file)


main()
