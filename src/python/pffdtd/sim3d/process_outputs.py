# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2021 Brian Hamilton

"""Read in sim_outs.h5 and process (integrate, low-cut, low-pass, etc.)
This gets called from command line with cmdline arguments (run after simulation)
"""

from pathlib import Path

import click
import h5py
import matplotlib.pyplot as plt
import numpy as np
from resampy import resample

from pffdtd.absorption.air import apply_visco_filter
from pffdtd.absorption.air import apply_modal_filter
from pffdtd.absorption.air import apply_ola_filter
from pffdtd.common.filter import apply_lowcut, apply_lowpass
from pffdtd.common.plot import plot_styles
from pffdtd.common.wavfile import save_as_wav_files
from pffdtd.geometry.math import iceil


class ProcessOutputs:
    # class to process sim_outs.h5 file
    def __init__(self, sim_dir):
        self.print('loading...')

        # get some integers from signals
        self.sim_dir = sim_dir
        h5f = h5py.File(sim_dir / Path('signals.h5'), 'r')
        out_alpha = h5f['out_alpha'][...]
        Nr = h5f['Nr'][()]
        Nt = h5f['Nt'][()]
        diff = h5f['diff'][()]
        h5f.close()

        # get some sim constants (floats) from constants
        h5f = h5py.File(sim_dir / Path('constants.h5'), 'r')
        Ts = h5f['Ts'][()]
        Tc = h5f['Tc'][()]
        rh = h5f['rh'][()]
        h5f.close()

        # read the raw outputs from sim_outs
        h5f = h5py.File(sim_dir / Path('sim_outs.h5'), 'r')
        u_out = h5f['u_out'][...]
        h5f.close()
        self.print('loading done...')

        assert out_alpha.size == Nr
        assert u_out.size == Nr*Nt
        assert out_alpha.ndim == 2

        self.r_out = None  # for recombined raw outputs (corresponding to Rxyz)
        self.r_out_f = None  # r_out filtered (and diffed)

        self.Ts = Ts
        self.Fs = 1/Ts
        self.Nt = Nt
        self.Ts_f = Ts
        self.Fs_f = 1/Ts
        self.Nt_f = Nt
        self.Nr = Nr
        # self.Nmic = out_alpha.shape[0]
        self.u_out = u_out
        self.diff = diff
        self.out_alpha = out_alpha
        self.sim_dir = sim_dir

        self.Tc = Tc
        self.rh = rh

    def print(self, fstring):
        print(f'--PROCESS_OUTPUTS: {fstring}')

    def initial_process(self, fcut=10.0, N_order=4):
        # initial process: consolidate receivers with linterp weights, and integrate/low-cut
        self.print('initial process...')
        u_out = self.u_out
        out_alpha = self.out_alpha
        sim_dir = self.sim_dir
        apply_int = self.diff
        Ts = self.Ts

        # just recombine outputs (from trilinear interpolation)
        r_out = np.sum(
            (u_out*out_alpha.flat[:][:, None]).reshape((*out_alpha.shape, -1)), axis=1)

        h5f = h5py.File(sim_dir / Path('sim_outs.h5'), 'r+')
        try:
            del h5f['r_out']
            self.print('overwrite r_out dataset (native sample rate)')
        except:
            pass
        h5f.create_dataset('r_out', data=r_out)
        h5f.close()

        r_out_f = apply_lowcut(r_out, 1/Ts, fcut, N_order, apply_int)
        self.print('initial process done')

        self.r_out = r_out
        self.r_out_f = r_out_f

    def apply_lowpass(self, fcut, N_order=8, symmetric=True):
        # lowpass filter for fmax (to remove freqs with too much numerical dispersion)
        self.r_out_f = apply_lowpass(
            self.r_out_f, self.Fs_f, fcut, N_order, symmetric)

    def resample(self, Fs_f=48e3):
        # resample with resampy, 48kHz default
        Fs = self.Fs  # raw Fs
        if Fs == Fs_f:
            return
        r_out_f = self.r_out_f

        self.print('resampling')
        r_out_f = resample(r_out_f, Fs, Fs_f, filter='kaiser_best')

        self.Fs_f = Fs_f
        self.Ts_f = 1/Fs_f
        self.Nt_f = r_out_f.shape[-1]
        self.r_out_f = r_out_f

    # to apply Stokes' filter (see DAFx2021 paper)
    def apply_stokes_filter(self, NdB=120):
        Fs_f = self.Fs_f
        Tc = self.Tc
        rh = self.rh
        r_out_f = self.r_out_f

        self.print('applying Stokes air absorption filter')
        r_out_f = apply_visco_filter(r_out_f, Fs_f, Tc=Tc, rh=rh, NdB=NdB)

        self.Nt_f = r_out_f.shape[-1]  # gets lengthened by filter
        self.r_out_f = r_out_f

    # to apply Stokes' filter (see I3DA 2021 paper)
    def apply_modal_filter(self):
        Fs_f = self.Fs_f
        Tc = self.Tc
        rh = self.rh
        r_out_f = self.r_out_f

        self.print('applying modal air absorption filter')
        r_out_f = apply_modal_filter(r_out_f, Fs_f, Tc=Tc, rh=rh)

        self.Nt_f = r_out_f.shape[-1]  # gets lengthened by filter
        self.r_out_f = r_out_f

    # to apply air absorption through STFT (overlap-add) framework
    def apply_ola_filter(self):  # default settings for 48kHz
        Fs_f = self.Fs_f
        Tc = self.Tc
        rh = self.rh
        r_out_f = self.r_out_f

        self.print('applying OLA air absorption filter')
        r_out_f = apply_ola_filter(r_out_f, Fs_f, Tc=Tc, rh=rh)

        self.Nt_f = r_out_f.shape[-1]  # maybe lengthened by filter
        self.r_out_f = r_out_f

    # plot the raw outputs (just to debug)
    def plot_raw_outputs(self):
        Nt = self.Nt
        Ts = self.Ts
        tv = np.arange(Nt)*Ts
        r_out = self.r_out

        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        # for out in u_out:
        # ax.plot(tv,out,linestyle='-')
        # ax.set_title('raw grid outputs')
        # ax.margins(0, 0.1)
        # ax.set_xlabel('time (s)')
        # ax.grid(which='both', axis='both')

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for i in range(r_out.shape[0]):
            ax.plot(tv, r_out[i], linestyle='-', label=f'{i}')
        ax.set_title('r_out')
        ax.margins(0, 0.1)
        ax.set_xlabel('time (s)')
        ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
        ax.minorticks_on()
        ax.legend()

    # plot the final processed outputs
    def plot_filtered_outputs(self):
        # possibly resampled
        r_out_f = self.r_out_f
        Nt_f = self.Nt_f
        Ts_f = self.Ts_f
        Fs_f = self.Fs_f
        tv = np.arange(Nt_f)*Ts_f
        Nfft = 2**iceil(np.log2(Nt_f))
        fv = np.arange(np.int_(Nfft//2)+1)/Nfft*Fs_f

        fig = plt.figure()
        ax = fig.add_subplot(2, 1, 1)
        for i in range(r_out_f.shape[0]):
            ax.plot(tv, r_out_f[i], linestyle='-', label=f'R{i+1}')
        ax.set_title('r_out filtered')
        ax.margins(0, 0.1)
        # ax.set_xlim((0,0.1))
        ax.set_xlabel('time (s)')
        ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
        ax.minorticks_on()
        ax.legend()

        ax = fig.add_subplot(2, 1, 2)
        r_out_f_fft_dB = 20 * \
            np.log10(np.abs(np.fft.rfft(r_out_f, Nfft, axis=-1))+np.spacing(1))
        dB_max = np.max(r_out_f_fft_dB)

        for i in range(r_out_f.shape[0]):
            ax.plot(fv, r_out_f_fft_dB[i], linestyle='-', label=f'R{i+1}')

        ax.set_title('r_out filtered')
        ax.margins(0, 0.1)
        ax.set_xlabel('freq (Hz)')
        ax.set_ylabel('dB')
        ax.set_xscale('log')
        ax.set_ylim((dB_max-80, dB_max+10))
        ax.set_xlim((1, Fs_f/2))
        ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
        ax.minorticks_on()
        ax.legend()

    def show_plots(self):
        plt.show()

    def save_wav(self):
        # save in WAV files, with native scaling and normalised across group of receivers
        # saves processed outputs
        save_as_wav_files(self.r_out_f, self.Fs_f, self.sim_dir, True)

    def save_h5(self):
        # saw processed outputs in .h5 (with native scaling)
        # saves processed outputs
        self.print('saving H5 data..')
        h5f = h5py.File(self.sim_dir / Path('sim_outs_processed.h5'), 'w')
        h5f.create_dataset('r_out_f', data=self.r_out_f)
        h5f.create_dataset('Fs_f', data=self.Fs_f)
        h5f.close()


def process_outputs(
    *,
    sim_dir=None,
    resample_fs=None,
    fcut_lowcut=None,
    order_lowcut=None,
    fcut_lowpass=None,
    order_lowpass=None,
    symmetric_lowpass=None,
    air_abs_filter=None,
    save_wav=None,
    plot_raw=None,
    plot=None,
):
    po = ProcessOutputs(sim_dir)

    po.initial_process(fcut=fcut_lowcut, N_order=order_lowcut)

    if resample_fs:
        po.resample(resample_fs)

    if fcut_lowpass > 0:
        po.apply_lowpass(fcut=fcut_lowpass, N_order=order_lowpass,
                         symmetric=symmetric_lowpass)

    # these are only needed if you're simulating with fmax >1kHz, but generally fine to use
    if air_abs_filter.lower() == 'modal':  # best, but slowest
        po.apply_modal_filter()
    elif air_abs_filter.lower() == 'stokes':  # generally fine for indoor air
        po.apply_stokes_filter()
    elif air_abs_filter.lower() == 'ola':  # fastest, but not as recommended
        po.apply_ola_filter()

    po.save_h5()

    if save_wav:
        po.save_wav()

    plt.rcParams.update(plot_styles)

    if plot_raw:
        po.plot_raw_outputs()

    if plot or plot_raw:
        po.plot_filtered_outputs()
        po.show_plots()


@click.command(name='process-outputs', help='Process raw simulation output.')
@click.option('--sim_dir', type=click.Path(exists=True))
@click.option('--plot', is_flag=True)
@click.option('--plot_raw', is_flag=True)
@click.option('--save_wav', is_flag=True)
@click.option('--resample_fs', default=48_000.0)
@click.option('--fcut_lowcut', default=10.0)
@click.option('--fcut_lowpass', default=0.0)
@click.option('--order_lowcut', default=8)
@click.option('--order_lowpass', default=8)
@click.option('--symmetric_lowpass', is_flag=True)
@click.option('--air_abs_filter', default='none')
def main(sim_dir, plot, plot_raw, save_wav, resample_fs, fcut_lowcut, fcut_lowpass, order_lowcut, order_lowpass, symmetric_lowpass, air_abs_filter):
    process_outputs(
        sim_dir=sim_dir,
        resample_fs=resample_fs,
        fcut_lowcut=fcut_lowcut,
        order_lowcut=order_lowcut,
        fcut_lowpass=fcut_lowpass,
        order_lowpass=order_lowpass,
        symmetric_lowpass=symmetric_lowpass,
        air_abs_filter=air_abs_filter,
        save_wav=save_wav,
        plot_raw=plot_raw,
        plot=plot,
    )
