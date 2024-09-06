from pathlib import Path

import numpy as np
import scipy.io.wavfile


def wavread(fname):
    fs, data = scipy.io.wavfile.read(fname)  # reads in (Nsamples,Nchannels)
    if data.dtype == np.int16:
        data = data/32768.0
        fs = np.float64(fs)
    return fs, np.float64(data.T)


def wavwrite(fname, fs, data):
    # expects (Nchannels,Nsamples), this will also assert that
    data = np.atleast_2d(data)
    # reads in (Nsamples,Nchannels)
    scipy.io.wavfile.write(fname, int(fs), np.float32(data.T))
    print(f'wrote {fname} at fs={fs/1000:.2f} kHz')


def save_as_wav_files(y, fs, sim_dir, verbose=True):
    """save in WAV files, with native scaling and normalised across group of receivers
    """
    # saves processed outputs
    y = np.atleast_2d(y)
    n_fac = np.max(np.abs(y.flat[:]))
    if verbose:
        print(f'headroom = {-20*np.log10(n_fac):.1}dB')

    for i in range(y.shape[0]):
        # normalised across receivers
        fname = Path(sim_dir / Path(f'R{i+1:03d}_out_normalised.wav'))
        wavwrite(fname, int(fs), y[i]/n_fac)
        if n_fac < 1.0:
            # not scaled, direct sound amplitude ~1/4πR
            fname = Path(sim_dir / Path(f'R{i+1:03d}_out_native.wav'))
            wavwrite(fname, int(fs), y[i])
