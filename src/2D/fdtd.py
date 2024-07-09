import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import numba as nb
from tqdm import tqdm


def add_diffusor(width, max_depth, in_mask, X, Y):
    depths = np.array([0.0, 0.25, 1.0, 0.5, 0.5, 1.0, 0.25])
    depths = np.array([0.0, 0.06, 0.25, 0.56, 1.0, 0.5, 0.12,
                      0.94, 0.81, 0.81, 0.94, 0.12, 0.5, 1, 0.56, 0.25, 0.06])
    assert depths.shape[0] == 17
    prime = depths.shape[0]
    n = int(10/width)
    for w in range(n):
        xs = (50/2-5)+w*width
        xe = xs+width
        ys = 50/2-5-max_depth
        ye = ys+depths[w % prime] * max_depth+0.05
        in_mask[(X >= xs) & (Y >= ys) & (X < xe) & (Y < ye)] = False

    return in_mask


@nb.jit(nopython=True, parallel=True)
def stencil_air_cart(u0, u1, u2, mask):
    Nx, Ny = u1.shape
    for ix in nb.prange(1, Nx-1):
        for iy in range(1, Ny-1):
            if mask[ix, iy]:
                left = u1[ix-1, iy]
                right = u1[ix+1, iy]
                bottom = u1[ix, iy-1]
                top = u1[ix, iy+1]
                last = u2[ix, iy]
                u0[ix, iy] = 0.5 * (left+right+bottom+top) - last


@nb.jit(nopython=True, parallel=True)
def stencil_boundary_rigid_cart(u0, u1, u2, bn_ixy, adj_bn):
    Nx, Ny = u1.shape
    Nb = bn_ixy.size
    for i in nb.prange(Nb):
        ib = bn_ixy[i]
        K = adj_bn[i]

        last1 = u1.flat[ib]
        last2 = u2.flat[ib]

        left = u1.flat[ib-1]
        right = u1.flat[ib + 1]
        top = u1.flat[ib + Ny]
        bottom = u1.flat[ib - Ny]

        neighbors = left + right + top + bottom
        u0.flat[ib] = (2 - 0.5 * K) * last1 + 0.5 * neighbors - last2


@nb.jit(nopython=True, parallel=True)
def stencil_boundary_loss_cart(u0,  u2, bn_ixy, adj_bn, lf):
    Nb = bn_ixy.size
    for i in nb.prange(Nb):
        ib = bn_ixy[i]
        K = adj_bn[i]

        prev = u2.flat[ib]
        current = u0.flat[ib]

        u0.flat[ib] = (current + lf * (4 - K) * prev) / (1 + lf * (4 - K))


def main():
    c = 343  # speed of sound m/s (20degC)
    fmax = 1000  # Hz
    PPW = 10.5  # points per wavelength at fmax
    duration = 0.1  # seconds
    refl_coeff = 0.99  # reflection coefficient

    Bx, By = 50.0, 50.0  # box dims (with lower corner at origin)
    x_in, y_in = Bx*0.5, By*0.5  # source input position
    R_dome = By*0.5  # heigh of dome (to be centered on roof of box)

    draw = True
    add_dome = False
    apply_rigid = True
    apply_loss = True

    if apply_loss:
        assert apply_rigid

    if add_dome:
        Lx = Bx
        Ly = By+R_dome
    else:
        Lx = Bx
        Ly = By

    # calculate grid spacing, time step, sample rate
    dx = c/fmax/PPW  # grid spacing
    dt = np.sqrt(0.5)*dx/c
    fs = 1/dt

    Nx = int(np.ceil(Lx/dx)+2)  # number of points in x-dir
    Ny = int(np.ceil(Ly/dx)+2)  # number of points in y-dir
    Nt = int(np.ceil(duration/dt))  # number of time-steps to compute

    # x and y sampling points
    xv = np.arange(0, Nx) * dx - 0.5 * dx
    yv = np.arange(0, Ny) * dx - 0.5 * dx
    X, Y = np.meshgrid(xv, yv, indexing='ij')

    # Mask for 'interior' points
    in_mask = np.zeros((Nx, Ny), dtype=bool)
    in_mask[(X >= 0) & (Y >= 0) & (X < Bx) & (Y < By)] = True

    if add_dome:
        in_mask[(X - 0.5 * Bx)**2 + (Y - By)**2 < R_dome**2] = True

    in_mask = add_diffusor(dx*3, 2.0, in_mask, X, Y)

    if apply_rigid:
        # Calculate number of interior neighbours (for interior points only)
        K_map = np.zeros((Nx, Ny), dtype=int)
        K_map[1:-1, 1:-1] += in_mask[2:, 1:-1]
        K_map[1:-1, 1:-1] += in_mask[:-2, 1:-1]
        K_map[1:-1, 1:-1] += in_mask[1:-1, 2:]
        K_map[1:-1, 1:-1] += in_mask[1:-1, :-2]
        K_map[~in_mask] = 0
        ib = np.where((K_map.flat > 0) & (K_map.flat < 4))[0]
        Kib = K_map.flat[ib]

    # Grid forcing points
    inx = int(np.round(x_in / dx + 0.5) + 1)
    iny = int(np.round(y_in / dx + 0.5) + 1)
    assert in_mask[inx, iny]

    if draw:
        draw_mask = np.nan*in_mask
        draw_mask[in_mask] = 1

    if apply_loss:
        # calculate specific admittance γ (g)
        assert abs(refl_coeff) <= 1.0
        g = (1-refl_coeff)/(1+refl_coeff)
        lf = 0.5*np.sqrt(0.5)*g  # a loss factor

    # Set up an excitation signal
    u_in = np.zeros(Nt, dtype=np.float64)
    u_in[0] = 1.0

    # Nh = int(np.ceil(5 * fs / fmax))
    # n = np.arange(Nh)
    # u_in[:Nh] = 0.5 - 0.5 * np.cos(2 * np.pi * n / Nh)
    # u_in[:Nh] *= np.sin(2 * np.pi * n / Nh)

    u0 = np.zeros((Nx, Ny), dtype=np.float64)
    u1 = np.zeros((Nx, Ny), dtype=np.float64)
    u2 = np.zeros((Nx, Ny), dtype=np.float64)

    sps30 = dt*30
    target_sps = 0.115
    fps = int(min(90, target_sps/sps30))

    video_name = 'output_video.avi'
    height, width = u0.shape
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(video_name, fourcc, fps,
                            (width, height), isColor=False)

    print(f'fmax = {fmax:.3f} Hz')
    print(f'fs   = {fs:.3f} Hz')
    print(f'Δx   = {dx*100:.5f} cm / {dx*1000:.2f} mm')
    print(f'fps  = {fps}')
    print(f'Nx   = {int(Nx)}')
    print(f'Ny   = {int(Ny)}')
    print(f'Nt   = {int(Nt)}')
    print(f'Nb   = {ib.shape[0]}')

    for nt in tqdm(range(Nt)):
        stencil_air_cart(u0, u1, u2, in_mask)
        if apply_rigid:
            stencil_boundary_rigid_cart(u0, u1, u2, ib, Kib)

            if apply_loss:
                stencil_boundary_loss_cart(u0, u2, ib, Kib, lf)
                # u0.flat[ib] = (u0.flat[ib] + lf * (4 - Kib) *
                #                u2.flat[ib]) / (1 + lf * (4 - Kib))

        u0[inx, iny] = u0[inx, iny] + u_in[nt]

        u2 = u1.copy()
        u1 = u0.copy()

        img = cv2.normalize(u0, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img[~in_mask] = 255
        video.write(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))

    video.release()

    print(f"last: u0={u0[inx, iny]} u1={u1[inx, iny]} u2={u2[inx, iny]}")

    # fig = plt.figure()

    # def draw_func(i):
    #     plt.cla()
    #     plt.clf()
    #     plt.xlim([np.min(xv), np.max(xv)])
    #     plt.ylim([np.min(yv), np.max(yv)])
    #     img = plt.imshow(
    #         (frames[i] * draw_mask).T,
    #         extent=(np.min(xv), np.max(xv), np.min(yv), np.max(yv)),
    #         cmap="bone",
    #         aspect="equal",
    #         origin="lower"
    #         # vmin=0.0,
    #         # vmax=0.05
    #     )
    #     # Add minorticks on the colorbar to make
    #     # it easy to read the values off the colorbar.
    #     color_bar = fig.colorbar(img, extend = 'both')

    #     color_bar.minorticks_on()

    # ani = FuncAnimation(fig, draw_func, frames=len(frames), interval=16)
    # plt.show()


if __name__ == "__main__":
    main()
