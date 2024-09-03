import pytest

from pffdtd.analysis.room_modes import find_nearest
from pffdtd.analysis.room_modes import detect_room_modes
from pffdtd.materials.adm_funcs import write_freq_ind_mat_from_Yn, convert_Sabs_to_Yn
from pffdtd.sim3d.room_builder import RoomBuilder
from pffdtd.sim3d.setup import sim_setup
from pffdtd.sim3d.testing import run_engine, skip_if_native_engine_unavailable
from pffdtd.sim3d.process_outputs import process_outputs


@pytest.mark.parametrize("engine", ["python", "native"])
@pytest.mark.parametrize(
    "room,fmax,ppw,fcc,dx_scale,tolerance",
    [
        ((2.8, 2.076, 1.48), 400, 10.5, False, 2, 3.0),
        ((3.0, 1.0, 2.0), 600, 7.7, True, 3, 6),
    ]
)
def test_sim3d_detect_room_modes(tmp_path, engine, room, fmax, ppw, fcc, dx_scale, tolerance):
    skip_if_native_engine_unavailable(engine)

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

    run_engine(sim_dir=sim_dir, engine=engine)

    process_outputs(
        data_dir=sim_dir,
        resample_fs=48_000,
        fcut_lowcut=fmin,
        order_lowcut=4,
        fcut_lowpass=fmax,
        order_lowpass=8,
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
