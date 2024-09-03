import os
import pathlib
import subprocess

import h5py
import numpy as np
import pytest

from pffdtd.common.myfuncs import to_ixy
from pffdtd.sim2d.setup import sim_setup_2d
from pffdtd.sim2d.engine import Engine2D


def model(*, Lx=None, Ly=None, Nx=None, Ny=None, dx=None, X=None, Y=None, in_mask=None):
    in_mask[(X >= 0) & (Y >= 0) & (X < Lx) & (Y < Ly)] = True
    inx = 2
    iny = 2
    out_ixy = [to_ixy(Nx-4, Ny-4, Nx, Ny)]
    assert in_mask[inx, iny]
    return in_mask, inx, iny, out_ixy


def test_sim3d_detect_room_modes(tmp_path):
    if not os.environ.get("PFFDTD_ENGINE_2D"):
        pytest.skip("Native 2D engine not available")


    sim_setup_2d(
        sim_dir=tmp_path,
        room=(2, 2),
        Tc=20,
        rh=50,
        fmax=1000.0,
        ppw=10.5,
        duration=6.0,
        refl_coeff=0.991,
        model_factory=model,
        apply_loss=True,
        diff=True,
        image=True,
        verbose=True,
    )

    engine = Engine2D(sim_dir=tmp_path, video=False)
    engine.run()
    engine.save_output("out-python.h5")

    exe = pathlib.Path(os.environ.get("PFFDTD_ENGINE_2D")).absolute()
    assert exe.exists()
    assert exe.is_file()

    result = subprocess.run(
        args=[str(exe), "-s", str(tmp_path), "-o", "out-native.h5"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0

    out_file = h5py.File(tmp_path / 'out-python.h5', 'r')
    out_py = out_file['out'][...]
    out_file.close()

    out_file = h5py.File(tmp_path / 'out-native.h5', 'r')
    out_native = out_file['out'][...]
    out_file.close()

    assert np.allclose(out_py, out_native)
