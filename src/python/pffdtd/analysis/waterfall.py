import sys

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.signal import stft
from scipy.io import wavfile

fs, rir = wavfile.read(sys.argv[1])
rir = rir / np.max(np.abs(rir))  # Normalize
nfft = 1024
frequencies, times, Zxx = stft(rir, fs=fs, nperseg=nfft)

decay_time = np.zeros(frequencies.shape)
for i, freq in enumerate(frequencies):
    magnitude = np.abs(Zxx[i, :])
    magnitude_db = 20 * np.log10(magnitude / np.max(magnitude))
    try:
        decay_index = np.where(magnitude_db <= -20)[0][0]
        decay_time[i] = times[decay_index]
    except IndexError:
        decay_time[i] = times[-1]  # If it never decays by 60 dB


# Waterfall Plot
X, Y = np.meshgrid(times, frequencies)
Z = 20 * np.log10(np.abs(Zxx)/nfft)
Z -= np.max(Z)
# Z = np.clip(Z, -60, 0)
# Z[Z<-60] = -100

fig = go.Figure(data=[go.Surface(
    colorscale='viridis',
    x=X,
    y=Y,
    z=Z
)])
fig.update_layout(
    title='Decay Times',
    scene=dict(
        xaxis_title='X: Time [s]',
        yaxis_title='Y: Frequency [Hz]',
        zaxis_title='Z: Amplitude [dB]',

        xaxis=dict(range=[0, 1],),
        yaxis=dict(type='log',),
        # zaxis=dict(range=[-60, 0],),
    ),
)

Z = 20 * np.log10(np.abs(Zxx/nfft))
Z -= np.max(Z)

plt.figure(figsize=(10, 6))
plt.pcolormesh(times, frequencies, Z, shading='gouraud', vmin=-60, vmax=0)
plt.colorbar(label='Amplitude [dB]')
plt.title('Decay Times')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.yscale('log')
plt.ylim([frequencies[1], fs / 2])
plt.show()
