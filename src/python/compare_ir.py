import argparse

import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

from common.plot import plot_styles
from common.myfuncs import iceil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', nargs='*')
    parser.add_argument('--fmin', type=float, default=20.0)
    parser.add_argument('--fmax', type=float, default=0.0)

    args = parser.parse_args()

    file_a = args.filename[0]
    file_b = args.filename[1]

    fs_a, buf_a = wavfile.read(file_a)
    fs_b, buf_b = wavfile.read(file_b)

    assert fs_a == fs_b
    assert buf_a.shape == buf_b.shape

    fmax = args.fmax if args.fmax != 0.0 else fs_a/2
    nfft = 2**iceil(np.log2(buf_a.shape[0]))
    freqs = np.fft.rfftfreq(nfft, 1/fs_a)

    spectrum_a = np.fft.rfft(buf_a, nfft)
    spectrum_b = np.fft.rfft(buf_b, nfft)

    dB_a = 20*np.log10(np.abs(spectrum_a)+np.spacing(1))
    dB_a -= np.max(dB_a)
    dB_a += 75.0

    dB_b = 20*np.log10(np.abs(spectrum_b)+np.spacing(1))
    dB_b -= np.max(dB_b)
    dB_b += 75.0

    plt.rcParams.update(plot_styles)

    fig, ax = plt.subplots(2, 1, constrained_layout=True)
    fig.suptitle("A vs. B")
    formatter = ScalarFormatter()
    formatter.set_scientific(False)

    ax[0].semilogx(freqs, dB_a, linestyle='-', label=f'A')
    ax[0].semilogx(freqs, dB_b, linestyle='-', label=f'B')
    ax[0].set_title('Spectrum')
    ax[0].set_xlabel('Frequency [Hz]')
    ax[0].set_ylabel('Amplitude [dB]')
    ax[0].set_ylim((30, 80))
    ax[0].set_xlim((args.fmin, fmax))
    ax[0].xaxis.set_major_formatter(formatter)
    ax[0].grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
    ax[0].minorticks_on()
    ax[0].legend()

    ax[1].semilogx(freqs, dB_a-dB_b, linestyle='-', label=f'A - B')
    ax[1].set_title('Difference')
    ax[1].set_xlabel('Frequency [Hz]')
    ax[1].set_ylabel('Amplitude [dB]')
    ax[1].set_xlim((args.fmin, fmax))
    ax[1].xaxis.set_major_formatter(formatter)
    ax[1].grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
    ax[1].minorticks_on()
    ax[1].legend()

    plt.show()


if __name__ == '__main__':
    main()
