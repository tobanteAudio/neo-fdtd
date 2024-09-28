# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

import pathlib

import cv2
import h5py
import numpy as np
from scipy.signal import lfilter

from pffdtd.geometry.math import to_ixy
from pffdtd.sim3d.constants import SimConstants


def write_model_image(in_mask, in_ixy, out_ixy, img_path):
    img = np.zeros(in_mask.shape, dtype=np.uint8)
    img[~in_mask] = 255
    img.flat[in_ixy] = 255
    img.flat[out_ixy] = 255

    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(img_path, img.astype(np.uint8))


def sim_setup_2d(
    *,
    sim_dir=None,
    room=None,
    Tc=None,
    rh=None,
    fmax=None,
    ppw=None,
    duration=None,
    refl_coeff=None,
    model_factory=None,

    apply_loss=True,
    diff=True,
    image=True,
    verbose=True,
):
    sim_dir = pathlib.Path(sim_dir)
    if not sim_dir.exists():
        sim_dir.mkdir(parents=True)

    constants = SimConstants(
        Tc=Tc,
        rh=rh,
        fmax=fmax,
        PPW=ppw,
        fcc=False,
    )

    Lx, Ly = room[0], room[1]  # box dims (with lower corner at origin)
    dx = constants.h
    dt = constants.Ts

    print('--SIM-SETUP: Generate mesh & mask')
    Nx = int(np.ceil(Lx/dx)+2)  # number of points in x-dir
    Ny = int(np.ceil(Ly/dx)+2)  # number of points in y-dir
    Nt = int(np.ceil(duration/dt))  # number of time-steps to compute

    # x and y sampling points
    xv = np.arange(0, Nx) * dx - 0.5 * dx
    yv = np.arange(0, Ny) * dx - 0.5 * dx
    X, Y = np.meshgrid(xv, yv, indexing='ij')

    # Mask for 'interior' points
    assert model_factory
    in_mask = np.zeros((Nx, Ny), dtype=bool)
    in_mask, inx, iny, out_ixy = model_factory(
        Lx=Lx,
        Ly=Ly,
        Nx=Nx,
        Ny=Ny,
        dx=dx,
        X=X,
        Y=Y,
        in_mask=in_mask
    )

    in_ixy = [to_ixy(inx, iny, Nx, Ny)]

    print('--SIM-SETUP: Create node ABCs')
    # Calculate number of interior neighbours (for interior points only)
    K_map = np.zeros((Nx, Ny), dtype=int)
    K_map[1:-1, 1:-1] += in_mask[2:, 1:-1]
    K_map[1:-1, 1:-1] += in_mask[:-2, 1:-1]
    K_map[1:-1, 1:-1] += in_mask[1:-1, 2:]
    K_map[1:-1, 1:-1] += in_mask[1:-1, :-2]
    K_map[~in_mask] = 0
    bn_ixy = np.where((K_map.flat > 0) & (K_map.flat < 4))[0]
    adj_bn = K_map.flat[bn_ixy]

    print('--SIM-SETUP: Calculate loss factor')
    loss_factor = 0
    if apply_loss:
        # calculate specific admittance γ (g)
        assert abs(refl_coeff) <= 1.0
        g = (1-refl_coeff)/(1+refl_coeff)
        loss_factor = 0.5*np.sqrt(0.5)*g  # a loss factor

    # Set up an excitation signal
    in_sigs = np.zeros(Nt, dtype=np.float64)
    in_sigs[0] = 1.0
    in_sigs_f = in_sigs

    if diff:
        b = 2/constants.Ts*np.array([1.0, -1.0])
        a = np.array([1.0, 1.0])
        in_sigs_f = lfilter(b, a, in_sigs, axis=-1)

    sps30 = dt*30
    target_sps = 0.115
    fps = int(min(120, target_sps/sps30))

    print('--SIM-SETUP: Writing simulation to h5 files')
    constants.save(sim_dir)

    h5f = h5py.File(sim_dir / pathlib.Path('sim.h5'), 'w')
    h5f.create_dataset('video_fps', data=np.float64(fps))
    h5f.create_dataset('Nt', data=np.int64(Nt))
    h5f.create_dataset('Nx', data=np.int64(Nx))
    h5f.create_dataset('Ny', data=np.int64(Ny))
    h5f.create_dataset('inx', data=np.int64(inx))
    h5f.create_dataset('iny', data=np.int64(iny))
    h5f.create_dataset('loss_factor', data=np.float64(loss_factor))
    h5f.create_dataset('adj_bn', data=adj_bn)
    h5f.create_dataset('bn_ixy', data=bn_ixy)
    h5f.create_dataset('in_mask', data=in_mask.flatten().astype(np.uint8))
    h5f.create_dataset('out_ixy', data=out_ixy)
    h5f.create_dataset('in_sigs', data=in_sigs_f)
    h5f.close()

    if image:
        write_model_image(in_mask, in_ixy, out_ixy, sim_dir / 'model.png')

    if verbose:
        print(f'  fmax = {constants.fmax:.3f} Hz')
        print(f'  fs   = {constants.fs:.3f} Hz')
        print(f'  Δx   = {dx*100:.5f} cm / {dx*1000:.2f} mm')
        print(f'  Nb   = {bn_ixy.shape[0]}')
        print(f'  Nt   = {int(Nt)}')
        print(f'  Nx   = {int(Nx)}')
        print(f'  Ny   = {int(Ny)}')
        print(f'  N    = {int(Nx)*int(Ny)}')
