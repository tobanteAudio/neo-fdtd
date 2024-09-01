from contextlib import chdir
import json
import os
import pathlib
import subprocess

import numpy as np
import psutil
import pytest
import scipy.io.wavfile as wavfile

from pffdtd.analysis.room_modes import find_nearest
from pffdtd.analysis.room_modes import detect_room_modes
from pffdtd.materials.adm_funcs import write_freq_ind_mat_from_Yn, convert_Sabs_to_Yn
from pffdtd.sim2d.fdtd import point_on_circle
from pffdtd.sim3d.room_builder import RoomBuilder
from pffdtd.sim3d.setup import sim_setup
from pffdtd.sim3d.engine import EnginePython3D
from pffdtd.sim3d.process_outputs import process_outputs
from pffdtd.localization.tdoa import locate_sound_source


def _run_engine(sim_dir, engine):
    if engine == "python":
        eng = EnginePython3D(sim_dir)
        eng.run_all(1)
        eng.save_outputs()
    else:
        exe = pathlib.Path(os.environ.get("PFFDTD_ENGINE")).absolute()
        assert engine == "native"
        assert exe.exists()

        with chdir(sim_dir):
            result = subprocess.run(
                args=[str(exe)],
                capture_output=True,
                text=True,
                check=True,
            )
            assert result.returncode == 0


def _skip_if_native_engine_unavailable(engine):
    if engine == "native":
        if not os.environ.get("PFFDTD_ENGINE"):
            pytest.skip("Native engine not available")


def _skip_if_not_enough_memory(engine):
    if engine == "python":
        if psutil.virtual_memory().available <= 6e9:
            pytest.skip("Not enough memory")


@pytest.mark.parametrize("engine", ["python", "native"])
def test_sim3d_locate_sound_source(tmp_path, engine):
    _skip_if_native_engine_unavailable(engine)
    _skip_if_not_enough_memory(engine)

    fmin = 20
    fmax = 1000
    ppw = 10.5
    root_dir = tmp_path
    sim_dir = root_dir/'cpu'
    model_file = root_dir/'model.json'
    material = 'sabine_9512.h5'

    length = 3.0
    width = 3.0
    height = 3.0

    source = [width/2, length-0.1, height/2]
    mics = [
        np.array([0, 0, 0]),
        np.array([1, 0, 0]),
        np.array([0.5, np.sqrt(3)/2, 0]),
        np.array([0.5, np.sqrt(3)/6, np.sqrt(6)/3]),
    ]

    builder = RoomBuilder(length, width, height)
    builder.with_colors({
        "Ceiling": [200, 200, 200],
        "Floor": [151, 134, 122],
        "Walls": [255, 255, 255],
    })

    builder.add_source("S1", source)
    builder.add_receiver("R1", list(mics[1-1]/2+[0.5, 0.5, 0.5]))
    builder.add_receiver("R2", list(mics[2-1]/2+[0.5, 0.5, 0.5]))
    builder.add_receiver("R3", list(mics[3-1]/2+[0.5, 0.5, 0.5]))
    builder.add_receiver("R4", list(mics[4-1]/2+[0.5, 0.5, 0.5]))
    builder.build(model_file)

    write_freq_ind_mat_from_Yn(convert_Sabs_to_Yn(0.9512), root_dir / material)

    sim_setup(
        model_json_file=model_file,
        mat_folder=root_dir,
        mat_files_dict={
            'Ceiling': material,
            'Floor': material,
            'Walls': material,
        },
        diff_source=True,
        duration=0.5,
        fcc_flag=False,
        fmax=fmax,
        PPW=ppw,
        insig_type='impulse',
        save_folder=sim_dir,
    )

    _run_engine(sim_dir=sim_dir, engine=engine)

    process_outputs(
        data_dir=sim_dir,
        resample_Fs=48_000,
        fcut_lowcut=fmin,
        N_order_lowcut=4,
        fcut_lowpass=fmax,
        N_order_lowpass=8,
        symmetric_lowpass=True,
        air_abs_filter="none",
        save_wav=True,
        plot_raw=False,
        plot=False,
    )

    fs1, mic1 = wavfile.read(sim_dir/"R001_out_normalised.wav")
    fs2, mic2 = wavfile.read(sim_dir/"R002_out_normalised.wav")
    fs3, mic3 = wavfile.read(sim_dir/"R003_out_normalised.wav")
    fs4, mic4 = wavfile.read(sim_dir/"R004_out_normalised.wav")
    assert fs1 == fs2
    assert fs1 == fs3
    assert fs1 == fs4

    fs = fs1
    mic_sigs = [mic1, mic2, mic3, mic4]

    with open(model_file, "r") as f:
        model = json.load(f)
    mic_pos = np.array([
        model["receivers"][0]["xyz"],
        model["receivers"][1]["xyz"],
        model["receivers"][2]["xyz"],
        model["receivers"][3]["xyz"],
    ])

    actual = model["sources"][0]["xyz"]
    estimated = locate_sound_source(mic_pos, mic_sigs, fs, verbose=True)
    assert np.linalg.norm(actual-estimated) <= 0.1


@pytest.mark.parametrize("engine", ["python", "native"])
def test_sim3d_infinite_baffle(tmp_path, engine):
    _skip_if_native_engine_unavailable(engine)
    _skip_if_not_enough_memory(engine)

    fmin = 20
    fmax = 1000
    ppw = 10.5
    dx = 343/(fmax*ppw)
    offset = dx*2.0
    root_dir = tmp_path
    sim_dir = root_dir/'cpu'
    model_file = root_dir/'model.json'
    material = 'sabine_01.h5'
    baffle_size = 1.5
    depth = baffle_size/2
    radius = depth-offset*2

    s1 = [baffle_size/2, depth-offset, baffle_size/2]
    r1 = point_on_circle((s1[0], s1[1]), radius, np.deg2rad(-90))
    r2 = point_on_circle((s1[0], s1[1]), radius, np.deg2rad(-45))
    r3 = point_on_circle((s1[0], s1[1]), radius, np.deg2rad(-135))

    r1 = [r1[0], r1[1], s1[2]]
    r2 = [r2[0], r2[1], s1[2]]
    r3 = [r3[0], r3[1], s1[2]]

    model = {
        "mats_hash": {
            "Baffle": {
                "tris": [
                    [0, 2, 1],
                    [0, 3, 2],
                    [1, 5, 4],
                    [1, 3, 5]
                ],
                "pts": [
                    [0.0, depth, 0.0],
                    [baffle_size/2, depth, 0.0],
                    [baffle_size/2, depth, baffle_size],
                    [0.0, depth, baffle_size],
                    [baffle_size, depth, 0.0],
                    [baffle_size, depth, baffle_size]
                ],
                "color": [255, 255, 255],
                "sides": [1, 1, 1, 1]
            }
        },
        "sources": [{"name": "S1", "xyz": s1}],
        "receivers": [
            {"name": "R1", "xyz": r1},
            {"name": "R2", "xyz": r2},
            {"name": "R3", "xyz": r3},
        ]
    }

    with open(model_file, "w") as file:
        json.dump(model, file)

    write_freq_ind_mat_from_Yn(convert_Sabs_to_Yn(0.01), root_dir / material)

    sim_setup(
        model_json_file=model_file,
        mat_folder=root_dir,
        mat_files_dict={'Baffle': material},
        diff_source=True,
        duration=3.75,
        fcc_flag=False,
        fmax=fmax,
        PPW=ppw,
        insig_type='impulse',
        save_folder=sim_dir,
        bmax=[baffle_size, depth, baffle_size],
        bmin=[0, 0, 0],
    )

    _run_engine(sim_dir=sim_dir, engine=engine)

    process_outputs(
        data_dir=sim_dir,
        resample_Fs=48_000,
        fcut_lowcut=fmin,
        N_order_lowcut=4,
        fcut_lowpass=fmax,
        N_order_lowpass=8,
        symmetric_lowpass=True,
        air_abs_filter="none",
        save_wav=True,
        plot_raw=False,
        plot=False,
    )

    fs_1, buf_1 = wavfile.read(sim_dir / "R001_out_normalised.wav")
    fs_2, buf_2 = wavfile.read(sim_dir / "R002_out_normalised.wav")
    fs_3, buf_3 = wavfile.read(sim_dir / "R003_out_normalised.wav")
    assert fs_1 == fs_2
    assert fs_1 == fs_3

    error_2 = 20*np.log10(np.max(np.abs(buf_1-buf_2)))
    error_3 = 20*np.log10(np.max(np.abs(buf_1-buf_3)))
    assert error_2 < -12.0
    assert error_3 < -12.0


@pytest.mark.parametrize("engine", ["python", "native"])
@pytest.mark.parametrize(
    "room,fmax,ppw,fcc,dx_scale,tolerance",
    [
        ((2.8, 2.076, 1.48), 400, 10.5, False, 2, 3.0),
        ((3.0, 1.0, 2.0), 600, 7.7, True, 3, 6),
    ]
)
def test_sim3d_detect_room_modes(tmp_path, engine, room, fmax, ppw, fcc, dx_scale, tolerance):
    _skip_if_native_engine_unavailable(engine)
    _skip_if_not_enough_memory(engine)

    L = room[0]
    W = room[1]
    H = room[2]
    fmin = 20
    dx = 343/(fmax*ppw)
    root_dir = tmp_path
    sim_dir = root_dir/'cpu'
    model_file = root_dir/'model.json'
    material = 'sabine_02.h5'
    num_modes = 25

    offset = dx*dx_scale
    room = RoomBuilder(L, W, H)
    room.add_source("S1", [offset, offset, offset])
    room.add_receiver("R1", [W-offset, L-offset, H-offset])
    room.build(model_file)

    write_freq_ind_mat_from_Yn(convert_Sabs_to_Yn(0.02), root_dir / material)

    sim_setup(
        model_json_file=model_file,
        mat_folder=root_dir,
        mat_files_dict={
            'Ceiling': material,
            'Floor': material,
            'Walls': material,
        },
        diff_source=True,
        duration=3.75,
        fcc_flag=fcc,
        fmax=fmax,
        PPW=ppw,
        insig_type='impulse',
        save_folder=sim_dir,
    )

    _run_engine(sim_dir=sim_dir, engine=engine)

    process_outputs(
        data_dir=sim_dir,
        resample_Fs=48_000,
        fcut_lowcut=fmin,
        N_order_lowcut=4,
        fcut_lowpass=fmax,
        N_order_lowpass=8,
        symmetric_lowpass=True,
        air_abs_filter="none",
        save_wav=True,
        plot_raw=False,
        plot=False,
    )

    actual, measured = detect_room_modes(
        filename=None,
        data_dir=sim_dir,
        fmin=fmin,
        fmax=fmax,
        num_modes=num_modes,
        plot=False,
        width=W,
        length=L,
        height=H,
    )

    for mode in actual[:num_modes]:
        nearest = find_nearest(measured, mode)
        assert abs(mode-nearest) < tolerance
