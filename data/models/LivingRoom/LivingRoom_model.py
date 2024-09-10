import pathlib

from pffdtd.sim3d.constants import SimConstants
from pffdtd.sim3d.model_builder import MeshModelBuilder


def main():
    dir = pathlib.Path(".")
    obj = dir/"obj"

    fcc = False
    constants = SimConstants(Tc=20, rh=50, fmax=2500, PPW=10.5, fcc=fcc)
    mul = 4.0 if fcc else 2.0
    offset = constants.h * mul

    m = MeshModelBuilder()
    m.add("Book Shelf", obj / 'book_shelf.obj', [90, 90, 90], reverse=True)
    m.add("Ceiling", obj / 'ceiling.obj', [150, 150, 150], reverse=True)
    m.add("Coffee Table", obj / 'coffee_table.obj', [10, 10, 10], reverse=True)
    m.add("Couch", obj / 'couch.obj', [29, 50, 112], reverse=True)
    m.add("Desk", obj / 'desk.obj', [103, 70, 55], reverse=True)
    m.add("Floor", obj / 'floor.obj', [133, 94, 66], reverse=True)
    m.add("Kallax", obj / 'kallax.obj', [200, 200, 200], reverse=True)
    m.add("Table", obj / 'table.obj', [200, 200, 200], reverse=True)
    m.add("TV 42", obj / 'tv_42.obj', [10, 10, 10], reverse=True)
    m.add("TV 55", obj / 'tv_55.obj', [10, 10, 10], reverse=True)
    m.add("TV Table", obj / 'tv_table.obj', [120, 120, 120], reverse=True)
    m.add("Walls", obj / 'walls.obj', [175, 175, 175], reverse=True)
    m.add_source("S1", [offset, offset, 3.12-offset])
    m.add_receiver("R1", [3.65-offset, 6-offset, offset])
    m.add_receiver("R2", [3.65-offset, 6-offset, offset+1.0])
    m.write(dir / "model.json")


main()
