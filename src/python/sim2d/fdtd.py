import argparse
import pathlib

import cv2
import h5py
import numpy as np
import numba as nb
from tqdm import tqdm

from pffdtd.diffusor.design import diffusor_bandwidth
from pffdtd.diffusor.qrd import quadratic_residue_diffuser
from pffdtd.diffusor.prd import primitive_root_diffuser


def to_ixy(x, y, Nx, Ny, order="row"):
    if order == "row":
        return x*Ny+y
    return y*Nx+x


def point_on_circle(center, radius: float, angle: float):
    """
    Calculate the coordinates of a point on a circle arc.

    Parameters:
    center (tuple): (x, y) coordinates of the center of the circle.
    radius: Radius of the circle.
    angle: Angle in radians.

    Returns:
    tuple: (p_x, p_y) coordinates of the point on the circle arc.
    """
    x, y = center
    p_x = x + radius * np.cos(angle)
    p_y = y + radius * np.sin(angle)
    return (p_x, p_y)


def quantize_point(pos, dx, ret_err=True, scale=False):
    def quantize(i):
        low = np.floor(i/dx)
        high = np.ceil(i/dx)
        err_low = np.fabs(i-low*dx)
        err_high = np.fabs(i-high*dx)
        err = err_low if err_low < err_high else err_high
        i_q = low if err_low < err_high else high
        if scale:
            return i_q*dx, err
        else:
            return i_q, err

    x, y = pos
    xq, yq = quantize(x), quantize(y)
    if ret_err:
        return xq, yq
    return xq[0], yq[0]


def add_diffusor(prime, well_width, max_depth, room, in_mask, X, Y, dx, c, verbose=False):
    print('--SIM-SETUP: Quantize diffusor')
    dim = (well_width, max_depth)
    width = well_width
    depth = max_depth
    fmin_t, fmax_t = diffusor_bandwidth(dim[0], dim[1], c=c)

    (width_q, werr_q), (depth_q, derr_q) = quantize_point(dim, dx)
    width_q *= dx
    depth_q *= dx
    print(dim, width_q, depth_q)
    fmin_q, fmax_q = diffusor_bandwidth(width_q, depth_q, c=c)

    radius = 5
    total_width = 5
    pos = (room[0]/2-total_width/2, room[1]/2-radius-depth)
    pos_q = quantize_point(pos, dx)
    n = int(total_width/width_q)

    if verbose:
        print(f"  {pos_q=}")
        print(f"  {width=:.4f}")
        print(f"  {depth=:.4f}")
        print(f"  {fmin_t=:.4f}")
        print(f"  {fmax_t=:.4f}")
        print(f"  {width_q=:.4f}")
        print(f"  {depth_q=:.4f}")
        print(f"  {fmin_q=:.4f}")
        print(f"  {fmax_q=:.4f}")
        print(f"  error_w={werr_q/width*100:.2f}%")
        print(f"  error_d={derr_q/depth*100:.2f}%")

    print('--SIM-SETUP: Locate diffusor')
    depths, g = primitive_root_diffuser(prime, g=None, depth=depth_q)
    depths = quadratic_residue_diffuser(prime, depth_q)
    prime = depths.shape[0]
    for w in range(n):
        xs = (room[0]/2-total_width/2)+w*width_q
        xe = xs+width_q
        ys = room[1]/2-5-depth_q
        ye = ys+depths[w % prime]+0.05
        in_mask[(X >= xs) & (Y >= ys) & (X < xe) & (Y < ye)] = False

    return in_mask


def make_receiver_arc(count, center, radius, dx, Nx, Ny):
    x, y = center
    angles = np.linspace(0.0, 180.0, count, endpoint=True)
    out_ixy = []
    out_cart = []
    for i in range(angles.shape[0]):
        x, y = point_on_circle(center, radius, np.deg2rad(angles[i]))
        xq, yq = quantize_point((x, y), dx, ret_err=False)
        idx = to_ixy(xq, yq, Nx, Ny)
        out_ixy.append(idx)
        out_cart.append((xq, yq))
    return out_ixy, out_cart


@nb.njit(parallel=True)
def stencil_air(u0, u1, u2, mask):
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


@nb.njit(parallel=True)
def stencil_boundary_rigid(u0, u1, u2, bn_ixy, adj_bn):
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


@nb.njit(parallel=True)
def stencil_boundary_loss(u0, u2, bn_ixy, adj_bn, loss_factor):
    Nb = bn_ixy.size
    for i in nb.prange(Nb):
        ib = bn_ixy[i]
        K = adj_bn[i]
        lf = loss_factor
        prev = u2.flat[ib]
        current = u0.flat[ib]

        u0.flat[ib] = (current + lf * (4 - K) * prev) / (1 + lf * (4 - K))


def main():
    bool_action = argparse.BooleanOptionalAction
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--duration', type=float, default=0.1)
    parser.add_argument('--fmax', type=float, default=1000.0)
    parser.add_argument('--ppw', type=float, default=10.5)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--apply_rigid', action=bool_action, default=True)
    parser.add_argument('--apply_loss', action=bool_action, default=True)

    args = parser.parse_args()
    verbose = args.verbose
    if not args.data_dir:
        raise RuntimeError("--data_dir not given")

    data_dir = pathlib.Path(args.data_dir)
    if not data_dir.exists():
        data_dir.mkdir(parents=True)

    c = 343  # speed of sound m/s (20degC)
    room = (30, 30)
    fmax = args.fmax  # Hz
    PPW = args.ppw  # points per wavelength at fmax
    duration = args.duration  # seconds
    refl_coeff = 0.99  # reflection coefficient

    apply_rigid = args.apply_rigid
    apply_loss = args.apply_loss
    if apply_loss:
        assert apply_rigid

    Lx, Ly = room[0], room[1]  # box dims (with lower corner at origin)
    x_in, y_in = Lx*0.5, Ly*0.5  # source input position

    # calculate grid spacing, time step, sample rate
    dx = c/fmax/PPW  # grid spacing
    dt = np.sqrt(0.5)*dx/c
    fs = 1/dt

    print('--SIM-SETUP: Generate mesh & mask')
    Nx = int(np.ceil(Lx/dx)+2)  # number of points in x-dir
    Ny = int(np.ceil(Ly/dx)+2)  # number of points in y-dir
    Nt = int(np.ceil(duration/dt))  # number of time-steps to compute

    # x and y sampling points
    xv = np.arange(0, Nx) * dx - 0.5 * dx
    yv = np.arange(0, Ny) * dx - 0.5 * dx
    X, Y = np.meshgrid(xv, yv, indexing='ij')

    # Mask for 'interior' points
    in_mask = np.zeros((Nx, Ny), dtype=bool)
    in_mask[(X >= 0) & (Y >= 0) & (X < Lx) & (Y < Ly)] = True

    # Diffusor
    print('--SIM-SETUP: Generate diffusor')
    prime = 17
    depth = 0.35
    width = 0.048
    in_mask = add_diffusor(prime, width, depth, room,
                           in_mask, X, Y, dx, c, verbose)

    # Receiver linear index
    print('--SIM-SETUP: Generate receivers')
    arc_radius = 5.0
    arc_center = (x_in, y_in-arc_radius)
    out_ixy, _ = make_receiver_arc(180, arc_center, arc_radius, dx, Nx, Ny)

    if apply_rigid:
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

    # Grid forcing points
    inx = int(np.round(x_in / dx + 0.5) + 1)
    iny = int(np.round(y_in / dx + 0.5) + 1)
    assert in_mask[inx, iny]

    print('--SIM-SETUP: Calculate loss factor')
    loss_factor = 0
    if apply_loss:
        # calculate specific admittance γ (g)
        assert abs(refl_coeff) <= 1.0
        g = (1-refl_coeff)/(1+refl_coeff)
        loss_factor = 0.5*np.sqrt(0.5)*g  # a loss factor

    # Set up an excitation signal
    src_sig = np.zeros(Nt, dtype=np.float64)
    src_sig[0] = 1.0

    # Nh = int(np.ceil(5 * fs / fmax))
    # n = np.arange(Nh)
    # src_sig[:Nh] = 0.5 - 0.5 * np.cos(2 * np.pi * n / Nh)
    # src_sig[:Nh] *= np.sin(2 * np.pi * n / Nh)

    print('--SIM-SETUP: Allocate python memory')
    u0 = np.zeros((Nx, Ny), dtype=np.float64)
    u1 = np.zeros((Nx, Ny), dtype=np.float64)
    u2 = np.zeros((Nx, Ny), dtype=np.float64)

    sps30 = dt*30
    target_sps = 0.115
    fps = int(min(120, target_sps/sps30))

    if args.video:
        video_name = data_dir/'output_video.avi'
        print(f'--SIM-SETUP: Create python video file: {video_name}')
        height, width = 1000, 1000  # u0.shape
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        video = cv2.VideoWriter(video_name, fourcc, fps,
                                (width, height), isColor=False)

    print('--SIM-SETUP: Writing dataset to h5 file')
    h5f = h5py.File(data_dir / pathlib.Path('sim.h5'), 'w')
    h5f.create_dataset('fmax', data=np.float64(fmax))
    h5f.create_dataset('fs', data=np.float64(fs))
    h5f.create_dataset('video_fps', data=np.float64(fps))
    h5f.create_dataset('dx', data=np.float64(dx))
    h5f.create_dataset('dt', data=np.float64(dt))
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
    h5f.create_dataset('src_sig', data=src_sig)
    h5f.close()

    print(f'  fmax = {fmax:.3f} Hz')
    print(f'  fs   = {fs:.3f} Hz')
    print(f'  Δx   = {dx*100:.5f} cm / {dx*1000:.2f} mm')
    print(f'  fps  = {fps}')
    print(f'  Nb   = {bn_ixy.shape[0]}')
    print(f'  Nt   = {int(Nt)}')
    print(f'  Nx   = {int(Nx)}')
    print(f'  Ny   = {int(Ny)}')
    print(f'  N    = {int(Nx)*int(Ny)}')

    model_img = np.zeros((Nx, Ny), dtype=np.uint8)
    model_img[~in_mask] = 255
    model_img.flat[out_ixy] = 255

    model_img = cv2.rotate(model_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(data_dir / 'model.png', model_img.astype(np.uint8))

    if args.video:
        for nt in tqdm(range(Nt)):
            stencil_air(u0, u1, u2, in_mask)
            if apply_rigid:
                stencil_boundary_rigid(u0, u1, u2, bn_ixy, adj_bn)
                if apply_loss:
                    stencil_boundary_loss(u0, u2, bn_ixy, adj_bn, loss_factor)

            u0[inx, iny] = u0[inx, iny] + src_sig[nt]

            u2 = u1.copy()
            u1 = u0.copy()

            img = np.abs(u0)
            img = cv2.normalize(img, None, 0, 255,
                                cv2.NORM_MINMAX).astype(np.uint8)
            img[~in_mask] = 255
            img = cv2.resize(img, (1000, 1000))
            video.write(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))

        video.release()

        print(f"last: u0={u0[inx, iny]} u1={u1[inx, iny]} u2={u2[inx, iny]}")


if __name__ == "__main__":
    main()
