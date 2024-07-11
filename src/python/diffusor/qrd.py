import numpy as np


def quadratic_residue_diffuser(prime, depth):
    n = np.mod(np.arange(0, prime, 1)**2, prime)
    n = n / np.max(n)
    return n*depth


def main():
    print(quadratic_residue_diffuser(7, 1.0))
    print(quadratic_residue_diffuser(17, 1.0))


if __name__ == "__main__":
    main()
