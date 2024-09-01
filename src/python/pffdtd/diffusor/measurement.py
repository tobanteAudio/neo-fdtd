import numpy as np
from scipy import signal


def bandpass_filter(y, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, y, axis=1)


def polar_response(y: np.array, fs: float, min_angle=0, max_angle=180, trim_angle=5):
    octave_bands = [
        # (63, 125),
        # (125, 250),
        (250, 500),
        (500, 1000),
        (1000, 2000),
        (2000, 4000),
        (4000, 8000),
        (8000, 16000),
    ]

    impulse = y[trim_angle:-trim_angle, :]

    mic_angles = np.linspace(min_angle, max_angle, 180, endpoint=True)
    mic_angles = mic_angles[trim_angle:-trim_angle]

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
