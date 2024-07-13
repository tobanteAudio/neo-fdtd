import numpy as np


def main():
    c = 343
    fs = 192000

    tweeter = np.array([0, 0, 1.2])
    midrange = np.array([0, 0, 1.05])
    woofer = np.array([0, 0, 0.9])
    subwoofer = np.array([0, 0, 0.5])

    listener = np.array([1.0, 2.0, 1.2])

    distance_tweeter = np.linalg.norm(listener - tweeter)
    distance_midrange = np.linalg.norm(listener - midrange)
    distance_woofer = np.linalg.norm(listener - woofer)
    distance_subwoofer = np.linalg.norm(listener - subwoofer)

    delay_tweeter = distance_tweeter/c
    delay_midrange = distance_midrange/c
    delay_woofer = distance_woofer/c
    delay_subwoofer = distance_subwoofer/c

    relative_distance_midrange = distance_midrange-distance_tweeter
    relative_distance_woofer = distance_woofer-distance_tweeter
    relative_distance_subwoofer = distance_subwoofer-distance_tweeter

    relative_delay_midrange = delay_midrange-delay_tweeter
    relative_delay_woofer = delay_woofer-delay_tweeter
    relative_delay_subwoofer = delay_subwoofer-delay_tweeter

    print("General")
    print(f"{c=}")
    print(f"{fs=}")
    print("")

    print("Distance to listener:")
    print(f"tweeter={distance_tweeter*100:.2f} cm")
    print(f"midrange={distance_midrange*100:.2f} cm")
    print(f"woofer={distance_woofer*100:.2f} cm")
    print(f"subwoofer={distance_subwoofer*100:.2f} cm")
    print("")

    print("Relative-distance to listener:")
    print(f"midrange={relative_distance_midrange*100:.2f} cm")
    print(f"woofer={relative_distance_woofer*100:.2f} cm")
    print(f"subwoofer={relative_distance_subwoofer*100:.2f} cm")
    print("")

    print("Delay to listener:")
    print(f"tweeter={delay_tweeter*1000:.2f} ms")
    print(f"midrange={delay_midrange*1000:.2f} ms")
    print(f"woofer={delay_woofer*1000:.2f} ms")
    print(f"subwoofer={delay_subwoofer*1000:.2f} ms")
    print("")

    print("Relative-delay to tweeter:")
    print(f"midrange={relative_delay_midrange*1000000:.3f} us")
    print(f"woofer={relative_delay_woofer*1000000:.3f} us")
    print(f"subwoofer={relative_delay_subwoofer*1000000:.3f} us")
    print("")

    print(f"midrange={fs*relative_delay_midrange:.2f} samples")
    print(f"woofer={fs*relative_delay_woofer:.2f} samples")
    print(f"subwoofer={fs*relative_delay_subwoofer:.2f} samples")
    print("")


if __name__ == "__main__":
    main()
