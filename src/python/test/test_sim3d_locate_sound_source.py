import json

import numpy as np
import pytest

from pffdtd.common.wavfile import wavread
from pffdtd.localization.tdoa import locate_sound_source
from pffdtd.materials.adm_funcs import write_freq_ind_mat_from_Yn, convert_Sabs_to_Yn
from pffdtd.sim3d.room_builder import RoomBuilder
from pffdtd.sim3d.sim_setup import sim_setup
from pffdtd.sim3d.testing import run_engine, skip_if_native_engine_unavailable
from pffdtd.sim3d.process_outputs import process_outputs


@pytest.mark.parametrize("engine", ["python", "native"])
def test_sim3d_locate_sound_source(tmp_path, engine):
    skip_if_native_engine_unavailable(engine)

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

    fs1, mic1 = wavread(sim_dir/"R001_out_normalised.wav")
    fs2, mic2 = wavread(sim_dir/"R002_out_normalised.wav")
    fs3, mic3 = wavread(sim_dir/"R003_out_normalised.wav")
    fs4, mic4 = wavread(sim_dir/"R004_out_normalised.wav")
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
