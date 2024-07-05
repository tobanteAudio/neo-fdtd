import csv
import glob
import os

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import matplotlib.pyplot as plt


def collect_wav_files(directory, pattern="*.wav"):
    search_pattern = os.path.join(directory, pattern)
    wav_files = glob.glob(search_pattern)
    return wav_files


def octave_filter(sig, fs, center_freq):
    # Design octave bandpass filter
    low_freq = center_freq / np.sqrt(2)
    high_freq = center_freq * np.sqrt(2)
    sos = signal.butter(4, [low_freq, high_freq],
                        btype='band', fs=fs, output='sos')
    filtered_signal = signal.sosfilt(sos, sig)
    return filtered_signal


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


def main():
    center_freqs = [15.625, 31.5, 63, 125, 250, 500, 1000, 2000, 4000]
    directory = "/Users/tobante/Developer/lib/pffdtd/data/sim_data/Tobi/cpu_2"
    files = collect_wav_files(directory, "*_out_normalised.wav")

    file_times = []
    for file in files:
        fs, ir = wavfile.read(file)
        t60_times = []
        print("")
        for center_freq in center_freqs:
            filtered_ir = octave_filter(ir, fs, center_freq)
            edc_db = energy_decay_curve(filtered_ir)
            t60 = calculate_t60(edc_db, fs)
            t60_times.append(round(t60, 3))
            print(f"T60 at {center_freq} Hz: {t60:.2f} seconds")

        file_times.append(t60_times)

    with open('out.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(center_freqs)
        writer.writerows(file_times)
    # # Optionally plot the EDC for one of the bands
    # plt.plot(edc_db)
    # plt.title(f"Energy Decay Curve at {center_freqs[5]} Hz")
    # plt.xlabel("Time (samples)")
    # plt.ylabel("Decay (dB)")
    # plt.grid(True)
    # plt.show()


main()
