import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, firwin, lfilter


def main():
    fc = 400
    fs = 48000
    dt = 1/fs
    n = fs

    sig = np.zeros(n, dtype=np.float64)
    sig[0] = 1.0

    num_taps = 101
    lowpass_taps = firwin(num_taps, fc, window='hamming', fs=fs)

    # Design a highpass FIR filter by spectral inversion
    highpass_taps = -lowpass_taps
    highpass_taps[num_taps // 2] += 1

    lowpass_sig = lfilter(lowpass_taps, 1.0, sig)
    highpass_sig = lfilter(highpass_taps, 1.0, sig)

    freqs = np.fft.rfftfreq(n, d=dt)
    lowpass_spectrum = np.fft.rfft(lowpass_sig)
    highpass_spectrum = np.fft.rfft(highpass_sig)

    lowpass_amplitude = np.abs(lowpass_spectrum)
    highpass_amplitude = np.abs(highpass_spectrum)
    mix_amplitude = lowpass_amplitude+highpass_amplitude

    # plt.semilogx(freqs, 20*np.log10(lowpass_amplitude), label="LP")
    # plt.semilogx(freqs, 20*np.log10(highpass_amplitude), label="HP")
    plt.semilogx(freqs, 20*np.log10(mix_amplitude), label="LP+HP")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
