# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

import json

from pffdtd.sim3d.constants import SimConstants


def main():
    fcc = False
    fmax = 16000.0
    ppw = 7.7
    constants = SimConstants(20, 50, fmax=fmax, PPW=ppw, fcc=fcc)
    mul = 3.0 if fcc else 2.0
    offset = constants.h * mul

    width = 3.0
    length = 2.0
    height = 3.0

    model = {
        "mats_hash": {
            "_RIGID": {
                "tris": [
                    [0, 2, 1],
                    [0, 3, 2],
                    [1, 5, 4],
                    [1, 3, 5]
                ],
                "pts": [
                    [0.0, length, 0.0],
                    [width/2, length, 0.0],
                    [width/2, length, height],
                    [0.0, length, height],
                    [width, length, 0.0],
                    [width, length, height]
                ],
                "color": [255, 255, 255],
                "sides": [0, 0, 0, 0]
            }
        },
        "sources": [
            {"name": "S1", "xyz": [width/2, length-offset, height/2]},
        ],
        "receivers": [
            {"name": "R1", "xyz": [width/2, offset, height/2 - 0.75]},
            {"name": "R2", "xyz": [width/2, offset, height/2 - 0.25]},
            {"name": "R3", "xyz": [width/2, offset, height/2 + 0.00]},
            {"name": "R4", "xyz": [width/2, offset, height/2 + 0.25]},
            {"name": "R5", "xyz": [width/2, offset, height/2 + 0.75]},
        ]
    }

    with open("model.json", "w") as file:
        json.dump(model, file)


main()
