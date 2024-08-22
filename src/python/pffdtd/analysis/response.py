import argparse

import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

from pffdtd.common.plot import plot_styles
from pffdtd.common.myfuncs import iceil


def fractional_octave_smoothing(magnitudes, fs, nfft, fraction=3):
    """
    Apply fractional octave smoothing to FFT magnitudes.

    Parameters:
    - magnitudes: Array of FFT magnitudes.
    - fs: Sampling rate of the signal.
    - nfft: Size of the FFT.
    - fraction: Fraction of the octave for smoothing (e.g., 3 for 1/3 octave, 6 for 1/6 octave).

    Returns:
    - smoothed: Array of smoothed FFT magnitudes.
    """

    # Define the reference frequency and calculate center frequencies
    f0 = 1000  # Reference frequency (1 kHz)
    start_freq = 15.625  # Start frequency (20 Hz)
    end_freq = 20480  # End frequency (20.48 kHz)

    # Calculate the number of bands needed to cover the frequency range
    n_bands = int(np.ceil(np.log2(end_freq / start_freq) * fraction))

    center_freqs = start_freq * 2**(np.arange(n_bands) / fraction)
    fft_freqs = np.fft.rfftfreq(nfft, 1/fs)
    smoothed = np.zeros_like(magnitudes)

    # for fc in center_freqs:
    #     fl = fc / 2**(1/(2*fraction))
    #     fu = fc * 2**(1/(2*fraction))
    #     indices = np.where((fft_freqs >= fl) & (fft_freqs <= fu))[0]
    #     if len(indices) > 0:
    #         smoothed[indices] = np.mean(magnitudes[indices])

    for i in range(magnitudes.shape[-1]):
        fc = fft_freqs[i]
        fl = fc / 2**(1/(2*fraction))
        fu = fc * 2**(1/(2*fraction))
        indices = np.where((fft_freqs >= fl) & (fft_freqs <= fu))[0]
        if len(indices) > 0:
            smoothed[i] = np.mean(magnitudes[indices])

    return smoothed


def parse_file_label(filename: str, fallback: str):
    s = filename.split(';')
    if len(s) == 1:
        return filename, fallback
    return s[0], s[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', nargs='*')
    parser.add_argument('--fmin', type=float, default=1.0)
    parser.add_argument('--fmax', type=float, default=1000.0)
    parser.add_argument('--label_a', type=str, default='A')
    parser.add_argument('--label_b', type=str, default='B')
    parser.add_argument('--smoothing', type=float, default=0.0)
    parser.add_argument('--target', type=float, default=0.0)

    args = parser.parse_args()

    file_a = args.filename[0]
    file_b = args.filename[1]

    label_a = args.label_a
    label_b = args.label_b

    fs_a, buf_a = wavfile.read(file_a)
    fs_b, buf_b = wavfile.read(file_b)

    assert fs_a == fs_b
    assert buf_a.shape == buf_b.shape

    fmax = args.fmax if args.fmax != 0.0 else fs_a/2
    nfft = 2**iceil(np.log2(buf_a.shape[0]))
    freqs = np.fft.rfftfreq(nfft, 1/fs_a)

    spectrum_a = np.fft.rfft(buf_a, nfft)
    spectrum_b = np.fft.rfft(buf_b, nfft)

    dB_a = 20*np.log10(np.abs(spectrum_a)/nfft+np.spacing(1))
    dB_b = 20*np.log10(np.abs(spectrum_b)/nfft+np.spacing(1))

    norm = max(np.max(dB_a), np.max(dB_b))
    dB_a -= norm
    dB_b -= norm

    dB_a += 75.0
    dB_b += 75.0

    if args.smoothing > 0.0:
        smoothing = args.smoothing
        dB_a = fractional_octave_smoothing(dB_a, fs_a, nfft, smoothing)
        dB_b = fractional_octave_smoothing(dB_b, fs_b, nfft, smoothing)

    difference = dB_b-dB_a

    plt.rcParams.update(plot_styles)

    fig, ax = plt.subplots(2, 1, constrained_layout=True)
    fig.suptitle(f"{label_a} vs. {label_b}")
    formatter = ScalarFormatter()
    formatter.set_scientific(False)

    ax[0].semilogx(freqs, dB_a, linestyle='-', label=f'{label_a}')
    ax[0].semilogx(freqs, dB_b, linestyle='-', label=f'{label_b}')
    ax[0].set_title('Spectrum')
    ax[0].set_xlabel('Frequency [Hz]')
    ax[0].set_ylabel('Amplitude [dB]')
    ax[0].set_ylim((20, 80))
    ax[0].set_xlim((args.fmin, fmax))
    ax[0].xaxis.set_major_formatter(formatter)
    ax[0].grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
    ax[0].minorticks_on()
    ax[0].legend()

    label = f'{label_b}-{label_a}'
    ax[1].semilogx(freqs, difference, linestyle='-', label=label)
    if args.target != 0.0:
        ax[1].hlines(args.target, args.fmin, fmax, linestyle='--', label=f"Target {args.target} dB")
    ax[1].set_title('Difference')
    ax[1].set_xlabel('Frequency [Hz]')
    ax[1].set_ylabel('Amplitude [dB]')
    ax[1].set_xlim((args.fmin, fmax))
    # ax[1].set_ylim((-np.max(np.abs(difference)), np.max(np.abs(difference))))
    ax[1].set_ylim((-30, 30))
    ax[1].xaxis.set_major_formatter(formatter)
    ax[1].grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
    ax[1].minorticks_on()
    ax[1].legend()

    plt.show()


if __name__ == '__main__':
    main()
