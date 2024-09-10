import numpy as np
from scipy.optimize import minimize
from scipy.signal import correlate


def _normalize(signal):
    return signal / np.max(np.abs(signal))


def _cross_correlation(signal1, signal2):
    correlation = correlate(signal1, signal2, mode='full')
    lags = np.arange(-len(signal1) + 1, len(signal1))
    return correlation, lags


def time_difference_of_arrival(signal1, signal2, fs):
    correlation, lags = _cross_correlation(signal1, signal2)
    # Find the index of the maximum correlation
    lag_idx = np.argmax(correlation)
    tdoa = lags[lag_idx] / fs  # Convert lag index to time difference
    return tdoa


def tdoa_residuals(source_pos, mic_positions, tdoas, c):
    # Function to compute the residual between observed and estimated TDOAs
    estimated_tdoas = []
    for i, mic_i in enumerate(mic_positions):
        for j in range(i+1, len(mic_positions)):
            di = np.linalg.norm(source_pos - mic_i)
            dj = np.linalg.norm(source_pos - mic_positions[j])
            estimated_tdoas.append((di - dj) / c)
    return np.sum((np.array(estimated_tdoas) - tdoas)**2)


def locate_sound_source(mic_positions, mic_sigs, fs, c=343.0, verbose=False):
    mic1 = _normalize(mic_sigs[0])
    mic2 = _normalize(mic_sigs[1])
    mic3 = _normalize(mic_sigs[2])
    mic4 = _normalize(mic_sigs[3])

    tdoa_12 = time_difference_of_arrival(mic1, mic2, fs)
    tdoa_13 = time_difference_of_arrival(mic1, mic3, fs)
    tdoa_14 = time_difference_of_arrival(mic1, mic4, fs)
    tdoa_23 = time_difference_of_arrival(mic2, mic3, fs)
    tdoa_24 = time_difference_of_arrival(mic2, mic4, fs)
    tdoa_34 = time_difference_of_arrival(mic3, mic4, fs)

    if verbose:
        print(f"TDOA between Mic1 and Mic2: {tdoa_12*1000:.4f} ms")
        print(f"TDOA between Mic1 and Mic3: {tdoa_13*1000:.4f} ms")
        print(f"TDOA between Mic1 and Mic4: {tdoa_14*1000:.4f} ms")
        print(f"TDOA between Mic2 and Mic3: {tdoa_23*1000:.4f} ms")
        print(f"TDOA between Mic2 and Mic4: {tdoa_24*1000:.4f} ms")
        print(f"TDOA between Mic3 and Mic4: {tdoa_34*1000:.4f} ms")

    tdoas = np.array([tdoa_12, tdoa_13, tdoa_14, tdoa_23, tdoa_24, tdoa_34])
    initial_guess = np.array([2.0, 2.0, 2.0])

    args = (mic_positions, tdoas, c)
    result = minimize(tdoa_residuals, initial_guess, args=args, tol=1e-10)

    return result.x
