# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch
from pathlib import Path

import click
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from pffdtd.common.wavfile import collect_wav_files, load_wav_files


def bandpass_filter(y, lowcut, highcut, fs, order=8):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = signal.butter(order, [low, high], btype='band', output='sos')
    return signal.sosfilt(sos, y, axis=-1)


def polar_response(y: np.array, fs: float, min_angle=0, max_angle=180, trim_angle=5):
    octave_bands = [
        # (63, 125),
        (125, 250),
        (250, 500),
        (500, 1000),
        (1000, 2000),
        (2000, 4000),
        (4000, 8000),
        (8000, 16000),
    ]

    mic_angles = np.linspace(min_angle, max_angle, y.shape[0], endpoint=True)
    mic_angles = mic_angles[trim_angle:-trim_angle]

    impulse = y[trim_angle:-trim_angle, :]

    bands = []
    for lowcut, highcut in octave_bands:
        band = bandpass_filter(impulse, lowcut, highcut, fs)
        label = f'{lowcut}-{highcut} Hz'
        bands.append((np.sqrt(np.mean(band**2, axis=1)), label))

    scale = np.max([np.max(y) for y, _ in bands])
    norm = []
    for band in bands:
        norm.append((band[0] / scale * 100, band[1]))

    return norm, mic_angles


@click.command(name='measurement', help='Measure polar response.')
@click.argument('sim_dir', nargs=1,  type=click.Path(exists=True))
def main(sim_dir):
    sim_dir = Path(sim_dir)
    files = collect_wav_files(sim_dir, '*_out_normalised.wav')
    fs, out = load_wav_files(files)

    constants = h5py.File(sim_dir / 'constants.h5', 'r')
    fmax = float(constants['fmax'][...])
    trim_ms = 20
    trim_samples = int(fs/1000*trim_ms)

    print(len(files))
    print(f"{fs=:.3f} Hz")
    print(f"{fmax=}")
    print(f"{trim_ms=}")
    print(f"{trim_samples=}")
    print(f"len={out.shape[-1]/fs:.2f} s")
    print(f"{out.shape=}")

    out = out[:, trim_samples:]
    times: np.ndarray = np.linspace(0.0, out.shape[-1]/fs, out.shape[-1])

    plt.plot(times, out[15, :], label=f'{15}deg')
    plt.plot(times, out[45, :], label=f'{45}deg')
    plt.plot(times, out[90, :], label=f'{90}deg')
    plt.grid(which='both')
    plt.legend()
    plt.show()

    rms_values, mic_angles = polar_response(out, fs, 1, 180, trim_angle=15)

    fig, ax = plt.subplots(3, 2, constrained_layout=True, subplot_kw={'projection': 'polar'})
    fig.suptitle('Diffusion')

    ax[0][0].plot(np.deg2rad(mic_angles), rms_values[0][0])
    ax[0][0].set_title(rms_values[0][1])
    ax[0][0].set_ylim((0.0, 100.0))
    ax[0][0].set_thetamin(0)
    ax[0][0].set_thetamax(180)

    ax[0][1].plot(np.deg2rad(mic_angles), rms_values[1][0])
    ax[0][1].set_title(rms_values[1][1])
    ax[0][1].set_ylim((0.0, 100.0))
    ax[0][1].set_thetamin(0)
    ax[0][1].set_thetamax(180)

    ax[1][0].plot(np.deg2rad(mic_angles), rms_values[2][0])
    ax[1][0].set_title(rms_values[2][1])
    ax[1][0].set_ylim((0.0, 100.0))
    ax[1][0].set_thetamin(0)
    ax[1][0].set_thetamax(180)

    ax[1][1].plot(np.deg2rad(mic_angles), rms_values[3][0])
    ax[1][1].set_title(rms_values[3][1])
    ax[1][1].set_ylim((0.0, 100.0))
    ax[1][1].set_thetamin(0)
    ax[1][1].set_thetamax(180)

    ax[2][0].plot(np.deg2rad(mic_angles), rms_values[4][0])
    ax[2][0].set_title(rms_values[4][1])
    ax[2][0].set_ylim((0.0, 100.0))
    ax[2][0].set_thetamin(0)
    ax[2][0].set_thetamax(180)

    ax[2][1].plot(np.deg2rad(mic_angles), rms_values[5][0])
    ax[2][1].set_title(rms_values[5][1])
    ax[2][1].set_ylim((0.0, 100.0))
    ax[2][1].set_thetamin(0)
    ax[2][1].set_thetamax(180)

    plt.show()
