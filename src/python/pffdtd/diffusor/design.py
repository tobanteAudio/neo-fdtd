import cv2
import numpy as np

from pffdtd.diffusor.qrd import quadratic_residue_diffuser

# fmax calculation matches multiple online calculators. Not sure about fmin


def diffusor_bandwidth(well_width, max_depth, c=343.0):
    fmin = c/(max_depth*4)
    fmax = c/(well_width*2)
    return fmin, fmax


def diffusor_dimensions(fmin, fmax, c=343.0):
    max_depth = c/(fmin*4)
    well_width = c/(fmax*2)
    return max_depth, well_width


def fractal_diffusor(primes, long_well, long_depth, short_depth, repeats=1, c=343.0):
    long_fmin, long_fmax = diffusor_bandwidth(long_well, long_depth, c)
    long_wells = quadratic_residue_diffuser(primes[0], long_depth)
    print(f"long_well={long_well*100:.2f}cm")
    print(f"long_depth={long_depth*100:.2f}cm")
    print(f"{long_fmin=:.1f}Hz")
    print(f"{long_fmax=:.1f}Hz")
    print("")

    short_well = long_well/primes[1]
    short_fmin, short_fmax = diffusor_bandwidth(short_well, short_depth, c)
    short_wells = quadratic_residue_diffuser(primes[1], short_depth)
    print(f"short_well={short_well*100:.2f}cm")
    print(f"short_depth={short_depth*100:.2f}cm")
    print(f"{short_fmin=:.1f}Hz")
    print(f"{short_fmax=:.1f}Hz")
    print("")

    total_depth = long_depth+short_depth
    total_width = long_well*primes[0]*int(repeats)
    print(f"{total_depth=:.2f}m")
    print(f"{total_width=:.2f}m")

    ratio = total_width/total_depth
    img_width = 1000
    dx = total_width/img_width
    img_size = (int((img_width+50)/ratio), img_width)
    print(dx)
    print(img_size)

    img = np.zeros(img_size, dtype=np.uint8)
    for r in range(int(repeats)):
        xo = int(long_well*primes[0]*r/dx)
        for i in range(primes[0]):
            p1 = (int(xo+i*(long_well/dx)), 0)
            p2 = (int(p1[0]+long_well/dx), int(long_wells[i % primes[0]]/dx))
            cv2.rectangle(img, p1, p2, (100, 100, 100), -1)
            for j in range(primes[1]):
                ps1 = (int(p1[0]+j*(short_well/dx)), p2[1])
                ps2 = (int(ps1[0]+short_well/dx), p2[1] +
                       int(short_wells[j % primes[1]]/dx))
                cv2.rectangle(img, ps1, ps2, (255, 255, 255), -1)

    img = cv2.flip(img, 0)
    cv2.imwrite('fractal.png', img.astype(np.uint8))


def main():
    primes = (13, 11)
    long_well = 0.12
    long_depth = 0.24
    short_depth = long_well*0.7
    repeats = 2
    fractal_diffusor(primes, long_well, long_depth, short_depth, repeats)


if __name__ == "__main__":
    main()
