# SPDX-License-Identifier: MIT

import numba as nb
import numpy as np
from numpy import exp, pi, cos
from scipy.fft import dct, idct  # default type2
from tqdm import tqdm

from pffdtd.air_abs.get_air_absorption import get_air_absorption
from pffdtd.common.myfuncs import iceil


def apply_modal_filter(x, Fs, Tc, rh, pad_t=0.0):
    """
    This is an implementation of an air absorption filter based on a modal approach.
    It solves a system of 1-d dissipative wave equations tune to air attenuation
    curves using a soft-source boundary condition.

    See paper for details:
    Hamilton, B. "Adding air attenuation to simulated room impulse responses: A
    modal approach", to be presented at I3DA 2021 conference in Bologna Italy.
    """

    # apply filter, x is (Nchannel,Nsamples) array
    # see I3DA paper for details
    # Tc is temperature in deg Celsius
    # rh is relative humidity

    Ts = 1/Fs

    x = np.atleast_2d(x)
    Nt0 = x.shape[-1]
    Nt = iceil(pad_t/Ts)+Nt0
    xp = np.zeros((x.shape[0], Nt))
    xp[:, :Nt0] = x
    del x

    y = np.zeros(xp.shape)

    Nx = Nt
    wqTs = pi*(np.arange(Nx)/Nx)
    wq = wqTs/Ts

    rd = get_air_absorption(wq/2/pi, Tc, rh)
    alphaq = rd['absfull_Np']
    c = rd['c']

    P0 = np.zeros(xp.shape)
    P1 = np.zeros(xp.shape)

    fx = np.zeros(xp.shape)
    fx[:, 0] = 1
    Fm = dct(fx, type=2, norm='ortho', axis=-1)

    sigqTs = c*alphaq*Ts
    a1 = 2*exp(-sigqTs)*cos(wqTs)
    a2 = -exp(-2*sigqTs)
    Fmsig1 = Fm*(1+sigqTs/2)/(1+sigqTs)
    Fmsig2 = Fm*(1-sigqTs/2)/(1+sigqTs)

    u = np.zeros((xp.shape[0], Nt+1))
    u[:, 1:] = xp[:, ::-1]  # flip

    pbar = tqdm(total=Nt, desc='modal filter', ascii=True)

    @nb.jit(nopython=True, parallel=True)
    def _run_step(P0, P1, a1, a2, Fmsig1, Fmsig2, un1, un0):
        P0[:] = a1*P1 + a2*P0 + Fmsig1*un1 - Fmsig2*un0

    for n in range(Nt):
        # P0 = a1*P1 + a2*P0 + Fmsig1*u[:,n+1] - Fmsig2*u[:,n]
        for i in range(P0.shape[0]):
            _run_step(P0[i], P1[i], a1, a2, Fmsig1[i],
                      Fmsig2[i], u[i, n+1], u[i, n])
        if n < Nt-1:  # dont swap on last sample
            P1, P0 = P0, P1
        pbar.update(1)
    pbar.close()

    y = idct(P0, type=2, norm='ortho', axis=-1)
    return np.squeeze(y)  # squeeze to 1d in case
