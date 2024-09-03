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
