import pytest

from pffdtd.analysis.room_modes import find_nearest
from pffdtd.analysis.room_modes import main as room_modes
from pffdtd.materials.adm_funcs import write_freq_ind_mat_from_Yn, convert_Sabs_to_Yn
from pffdtd.sim3d.room_builder import RoomBuilder
from pffdtd.sim3d.setup import sim_setup
from pffdtd.sim3d.engine import EnginePython3D
from pffdtd.sim3d.process_outputs import process_outputs


@pytest.mark.parametrize(
    "size,fmax,ppw,fcc,dx_scale,tolerance",
    [
        ((2.8, 2.076, 1.48), 400, 10.5, False, 2, 3.0),
        ((3.0, 1.0, 2.0), 800, 7.7, True, 3, 6),
    ]
)
def test_sim3d(tmp_path, size, fmax, ppw, fcc, dx_scale, tolerance):
    L = size[0]
    W = size[1]
    H = size[2]
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

    eng = EnginePython3D(sim_dir)
    eng.run_all(1)
    eng.save_outputs()

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

    actual, measured = room_modes(
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
