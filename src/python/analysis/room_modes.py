import glob
import os
import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import scipy.io.wavfile as wavfile

from common.myfuncs import iceil
from common.plot import plot_styles


def collect_wav_paths(dir, pattern="*.wav"):
    return list(sorted(glob.glob(os.path.join(dir, pattern))))


def hz_to_note(frequency):
    # Reference frequency for A4
    A4_frequency = 440.0
    # Reference position for A4 in the note list
    A4_position = 9
    # List of note names
    note_names = ["C", "C#", "D", "D#", "E",
                  "F", "F#", "G", "G#", "A", "A#", "B"]

    # Calculate the number of semitones between the given frequency and A4
    semitones_from_A4 = 12 * np.log2(frequency / A4_frequency)
    # Round to the nearest semitone
    semitone_offset = round(semitones_from_A4)

    # Calculate the octave
    octave = 4 + (A4_position + semitone_offset) // 12
    # Calculate the note position
    note_position = (A4_position + semitone_offset) % 12

    # Get the note name
    note_name = note_names[note_position]

    return f"{note_name}{octave}"


def room_mode(L, W, H, m, n, p):
    c = 343
    return c*0.5*np.sqrt((m/L)**2 + (n/W)**2 + (p/H)**2)


def room_mode_kind(m, n, p):
    non_zero = (m != 0) + (n != 0) + (p != 0)
    if non_zero == 1:
        return "axial"
    if non_zero == 2:
        return "tangential"
    return "oblique"


def frequency_spacing_index(modes):
    psi = 0.0
    num = len(modes)
    f0 = modes[0]['frequency']
    fn = modes[-1]['frequency']
    delta_hat = (fn-f0)/(num-1)
    for n in range(1, num):
        prev = modes[n-1]['frequency']
        mode = modes[n]
        freq = mode['frequency']
        delta = freq-prev
        psi += (delta/delta_hat)**2
    return psi / (num-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', nargs='*')
    parser.add_argument('--data_dir', type=str, help='run directory')
    parser.add_argument('--fmax', type=float, default=200.0)
    parser.add_argument('--fmin', type=float, default=1.0)
    parser.add_argument('--modes', type=int, default=10)
    args = parser.parse_args()

    directory = args.data_dir
    paths = args.filename
    if not paths:
        paths = collect_wav_paths(directory, "*_out_normalised.wav")

    scale = 0.9

    # Tobi
    L = 6.0 * scale
    W = 3.65 * scale
    H = 3.12 * scale

    # Optimal ratio A
    # L = 6 * scale
    # W = 4.96 * scale
    # H = 4.14 * scale

    # Optimal ratio B
    L = 7 * scale
    W = 5.19 * scale
    H = 3.70 * scale

    # Worst ratio (Cube)
    # W = L
    # H = L

    A = L*W
    V = A*H
    S = 2*(L*W+L*H+W*H)

    modes = []
    max_order = 6
    for m in range(max_order+1):
        for n in range(max_order+1):
            for p in range(max_order+1):
                if m+n+p > 0:
                    modes.append({
                        "m": m,
                        "n": n,
                        "p": p,
                        "frequency": room_mode(L, W, H, m, n, p)
                    })

    modes = sorted(modes, key=lambda x: x['frequency'])[:25]

    print(f"{L=:.3f}m {W=:.3f}m {H=:.3f}m")
    print(f"{A=:.2f}m^2 {S=:.2f}m^2 {V=:.2f}m^3")
    print(f"w/h={W/H:.2f} l/h={L/H:.2f} l/w={L/W:.2f}")
    print(f"FSI({len(modes)}): {frequency_spacing_index(modes):.2f}")
    print("")

    for mode in modes[:10]:
        m = mode['m']
        n = mode['n']
        p = mode['p']
        freq = mode['frequency']
        note = hz_to_note(freq)
        kind = room_mode_kind(m, n, p)
        print(f"[{m},{n},{p}] = {freq:.2f}Hz ({note}) {kind}")

    plt.rcParams.update(plot_styles)

    for file in paths:
        file = pathlib.Path(file)
        fs, buf = wavfile.read(file)
        fmin = args.fmin if args.fmin > 0 else 1
        fmax = args.fmax if args.fmax > 0 else fs/2

        nfft = (2**iceil(np.log2(buf.shape[0])))*2
        spectrum = np.fft.rfft(buf, nfft)
        freqs = np.fft.rfftfreq(nfft, 1/fs)

        dB = 20*np.log10(np.abs(spectrum)+np.spacing(1))
        dB -= np.max(dB)
        dB += 75.0

        dB_max = np.max(dB)
        peaks, _ = find_peaks(dB, width=2)

        print(freqs[peaks][5:10])

        plt.plot(freqs, dB, linestyle='-', label=f'{file.stem[:4]}')

    if args.modes > 0:
        mode_freqs = [mode['frequency'] for mode in modes][:args.modes]
        plt.vlines(mode_freqs, dB_max-80, dB_max+10,
                   colors='#AAAAAA', linestyles='--', label="Modes")
        if len(paths) == 1:
            plt.plot(freqs[peaks], dB[peaks], 'r.',
                     markersize=10, label='Peaks')

    plt.title("")
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.xscale('log')
    plt.ylim((dB_max-80, dB_max+5))
    plt.xlim((fmin, fmax))
    plt.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
