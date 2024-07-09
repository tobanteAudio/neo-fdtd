import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim_data', type=str)
    parser.add_argument('results', nargs='*')

    args = parser.parse_args()
    results = args.results[0]

    sim_data = h5py.File(args.sim_data, 'r')
    fs = sim_data['fs'][...]
    fmax = sim_data['fmax'][...]
    fmin = 20.0

    r_file = h5py.File(results, 'r')
    out = r_file['out'][...]

    print(results)
    print(out.shape)
    print(np.max(out))
    print(np.min(out))

    trim = out[:, 1:-1]
    spectrum = np.fft.rfft(trim, axis=-1)
    frequencies = np.fft.rfftfreq(trim.shape[1], 1/fs)

    dB = 20*np.log10(np.abs(spectrum)+np.spacing(1))
    dB = dB-np.max(dB)

    plt.plot(trim[179,:])
    # plt.plot(frequencies, dB[30, :], label=f'{30}deg')
    # plt.plot(frequencies, dB[60, :], label=f'{60}deg')
    # plt.plot(frequencies, dB[90, :], label=f'{90}deg')
    # plt.xlim((fmin, fmax))
    # plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
