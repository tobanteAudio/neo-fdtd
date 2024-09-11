# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

import pathlib

from pffdtd.geometry.math import find_third_vertex, point_along_line
from pffdtd.sim3d.constants import SimConstants
from pffdtd.sim3d.model_builder import MeshModelBuilder


def main():
    dir = pathlib.Path(".")
    obj = dir/"obj"

    fcc = True
    constants = SimConstants(Tc=20, rh=50, fmax=800, PPW=10.5, fcc=fcc)
    mul = 4.0 if fcc else 2.0
    offset = constants.h * mul

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

    m = MeshModelBuilder()
    m.add("ATC Left", obj / 'atc_left.obj', [5, 5, 5], reverse=True)
    m.add("ATC Right", obj / 'atc_right.obj', [5, 5, 5], reverse=True)
    m.add("Ceiling", obj / 'ceiling.obj', [60, 60, 60])
    m.add("Console", obj / 'console.obj', [60, 60, 60], reverse=True)
    m.add("Couch", obj / 'couch.obj', [5, 5, 48], reverse=True)
    # m.add("Diffusor", obj / 'diffusor.obj', [53, 33, 0], reverse=True)
    m.add("Floor", obj / 'floor.obj', [53, 33, 0])
    m.add("Outboard", obj / 'outboard.obj', [0, 0, 0], reverse=True)
    m.add("Rack", obj / 'rack.obj', [25, 25, 25], reverse=True)
    m.add("Raised Floor", obj / 'raised_floor.obj', [25, 25, 25], reverse=True)
    m.add("Walls Back", obj / 'walls_back.obj', [100, 100, 100])
    m.add("Walls Front", obj / 'walls_front.obj', [100, 100, 100])
    m.add("Walls Side", obj / 'walls_side.obj', [180, 180, 180])
    m.add("Windows", obj / 'windows.obj', [137, 207, 240], reverse=True)
    m.add_source("S1", s1)
    m.add_source("S2", s2)
    # m.add_source("S3", s3)
    # m.add_source("SUB1", sub1)
    # m.add_source("SUB2", sub2)
    m.add_receiver("R1", r1)
    m.add_receiver("R2", r2)
    m.add_receiver("R3", r3)
    m.add_receiver("R4", r4)
    m.add_receiver("R5", r5)
    m.add_receiver("R6", r6)
    m.write(dir / "model.json")


main()
