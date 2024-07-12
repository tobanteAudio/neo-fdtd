import numpy as np


def main():
    c = 343

    tweeter = np.array([0, 0, 1.2])
    midrange = np.array([0, 0, 1.1])
    woofer = np.array([0, 0, 0.9])
    subwoofer = np.array([0, 0, 0.5])

    listener = np.array([1.0, 2.0, 1.2])

    t_tweeter = np.linalg.norm(listener - tweeter)/c
    t_midrange = np.linalg.norm(listener - midrange)/c
    t_woofer = np.linalg.norm(listener - woofer)/c
    t_subwoofer = np.linalg.norm(listener - subwoofer)/c

    rd_midrange = t_midrange-t_tweeter
    rd_woofer = t_woofer-t_tweeter
    rd_subwoofer = t_subwoofer-t_tweeter

    print(f"t_tweeter={t_tweeter*1000:.2f} ms")
    print(f"t_midrange={t_midrange*1000:.2f} ms")
    print(f"t_woofer={t_woofer*1000:.2f} ms")
    print(f"t_subwoofer={t_subwoofer*1000:.2f} ms")
    print(f"rd_midrange={rd_midrange*1000:.3f} ms")
    print(f"rd_woofer={rd_woofer*1000:.3f} ms")
    print(f"rd_subwoofer={rd_subwoofer*1000:.3f} ms")


if __name__ == "__main__":
    main()
