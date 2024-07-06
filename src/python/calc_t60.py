import argparse
import glob
import os

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def collect_wav_files(directory, pattern="*.wav"):
    search_pattern = os.path.join(directory, pattern)
    wav_files = glob.glob(search_pattern)
    return wav_files


def third_octave_filter(sig, fs, center):
    factor = 2 ** (1/6)  # One-third octave factor
    low = center / factor
    high = center * factor
    sos = signal.butter(4, [low, high], btype='band', fs=fs, output='sos')
    return signal.sosfilt(sos, sig)


def energy_decay_curve(ir):
    edc = np.cumsum(ir[::-1]**2)[::-1]
    edc_db = 10 * np.log10(edc / np.max(edc))
    return edc_db


def calculate_t60(edc_db, fs):
    t = np.arange(len(edc_db)) / fs
    edc_db -= np.max(edc_db)  # Normalize to 0 dB at the start
    start_idx = np.where(edc_db <= -5)[0][0]
    end_idx = np.where(edc_db <= -35)[0][0]
    t60 = 2 * (t[end_idx] - t[start_idx])
    return t60


def ebu_3000_t60_threshold_upper(freqs):
    times = np.zeros_like(freqs)
    for i, freq in enumerate(freqs):
        if freq < 63:
            times[i] = 0.3
        elif freq <= 200:
            times[i] = 0.3 - (0.25 * (freq - 63) / (200 - 63))
        else:
            times[i] = 0.05
    return times


def ebu_3000_t60_threshold_lower(freqs):
    times = np.zeros_like(freqs)
    for i, freq in enumerate(freqs):
        if freq < 4000:
            times[i] = -0.05
        else:
            times[i] = -0.10
    return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='run directory')
    parser.add_argument('--fmin', type=float, help='min third-octave band')
    parser.add_argument('--fmax', type=float, help='max third-octave band')
    parser.set_defaults(fmin=20.0)
    parser.set_defaults(fmax=1000.0)

    args = parser.parse_args()
    directory = args.data_dir
    files = collect_wav_files(directory, "*_out_normalised.wav")

    # ISO 1/3 octaves
    center_freqs = np.array([
        20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
        200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
        2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000,
        20000
    ])

    center_freqs = center_freqs[np.where(center_freqs >= args.fmin)]
    center_freqs = center_freqs[np.where(center_freqs <= args.fmax)]

    file_times = []
    for file in files:
        fs, ir = wavfile.read(file)
        t60_times = []
        print(f"{file}")
        for center_freq in center_freqs:
            filtered_ir = third_octave_filter(ir, fs, center_freq)
            edc_db = energy_decay_curve(filtered_ir)
            t60 = calculate_t60(edc_db, fs)
            t60_times.append(round(t60, 3))
            print(f"T60 at {center_freq} Hz: {t60:.2f} seconds")

        file_times.append(np.array(t60_times))

    plot_styles = {
        'axes.edgecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'axes.grid.which': 'both',
        'axes.spines.left': False,
        'axes.spines.right': False,
        'axes.spines.top': False,
        'axes.spines.bottom': False,
        'figure.constrained_layout.use': True,
        'grid.color': '#CCCCCC',
        'grid.linewidth': '0.8',
        'xtick.color': '#666666',
        'xtick.major.bottom': True,
        'xtick.minor.bottom': False,
        'ytick.color': '#666666',
        'ytick.major.left': True,
        'ytick.minor.left': False,
    }

    plt.rcParams.update(plot_styles)

    fig, axs = plt.subplots(2, 1)
    formatter = ScalarFormatter()
    formatter.set_scientific(False)

    # T60
    axs[0].margins(0, 0.1)
    axs[0].semilogx(center_freqs, file_times[0])

    axs[0].set_title("Reverberation Time (T60)")
    axs[0].set_ylabel("Decay [s]")
    axs[0].set_xlabel("Frequency [Hz]")
    axs[0].xaxis.set_major_formatter(formatter)

    axs[0].set_xlim((center_freqs[0], center_freqs[-1]))
    axs[0].set_ylim((0.0, np.max(file_times[0])+0.1))

    axs[0].grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
    axs[0].minorticks_on()

    # Tolerance
    diff = np.insert(np.diff(file_times[0]), 0, 0.0)
    diff = np.insert(file_times[0][:-1]-file_times[0][1:], 0, 0.0)

    axs[1].margins(0, 0.1)
    axs[1].semilogx(center_freqs, diff)
    axs[1].semilogx(center_freqs, ebu_3000_t60_threshold_upper(center_freqs))
    axs[1].semilogx(center_freqs, ebu_3000_t60_threshold_lower(center_freqs))

    axs[1].set_title(f"T60 Tolerance (EBU Tech 3000)")
    axs[1].set_ylabel("Difference [s]")
    axs[1].set_xlabel("Frequency [Hz]")
    axs[1].xaxis.set_major_formatter(formatter)

    axs[1].set_xlim((center_freqs[0], center_freqs[-1]))
    axs[1].set_ylim((np.min(diff)-0.1, 0.4))

    axs[1].grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
    axs[1].minorticks_on()

    plt.show()


main()
