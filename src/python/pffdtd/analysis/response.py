import click
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

    frequencies = np.fft.rfftfreq(nfft, 1/fs)
    smoothed = np.zeros_like(magnitudes)

    for i in range(magnitudes.shape[-1]):
        fc = frequencies[i]
        fl = fc / 2**(1/(2*fraction))
        fu = fc * 2**(1/(2*fraction))
        indices = np.where((frequencies >= fl) & (frequencies <= fu))[0]
        if len(indices) > 0:
            smoothed[i] = np.mean(magnitudes[indices])

    return smoothed


@click.command(name="response", help="Plot frequency response.")
@click.argument('filename', nargs=2, type=click.Path(exists=True))
@click.option('--fmin', default=1.0)
@click.option('--fmax', default=1000.0)
@click.option('--label_a', default="A")
@click.option('--label_b', default="B")
@click.option('--smoothing', default=0.0)
@click.option('--target', default=0.0)
def main(filename, fmin, fmax, label_a, label_b, smoothing, target):
    file_a = filename[0]
    file_b = filename[1]

    label_a = label_a
    label_b = label_b

    fs_a, buf_a = wavfile.read(file_a)
    fs_b, buf_b = wavfile.read(file_b)

    assert fs_a == fs_b
    assert buf_a.shape == buf_b.shape

    fmax = fmax if fmax != 0.0 else fs_a/2
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

    if smoothing > 0.0:
        smoothing = smoothing
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
    ax[0].set_xlim((fmin, fmax))
    ax[0].xaxis.set_major_formatter(formatter)
    ax[0].grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
    ax[0].minorticks_on()
    ax[0].legend()

    label = f'{label_b}-{label_a}'
    ax[1].semilogx(freqs, difference, linestyle='-', label=label)
    if target != 0.0:
        ax[1].hlines(target, fmin, fmax, linestyle='--',
                     label=f"Target {target} dB")
    ax[1].set_title('Difference')
    ax[1].set_xlabel('Frequency [Hz]')
    ax[1].set_ylabel('Amplitude [dB]')
    ax[1].set_xlim((fmin, fmax))
    # ax[1].set_ylim((-np.max(np.abs(difference)), np.max(np.abs(difference))))
    ax[1].set_ylim((-30, 30))
    ax[1].xaxis.set_major_formatter(formatter)
    ax[1].grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
    ax[1].minorticks_on()
    ax[1].legend()

    plt.show()
