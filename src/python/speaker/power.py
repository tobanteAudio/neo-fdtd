import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def max_sound_pressure(SPL_ref, P_rms, P_ref=1):
    """Maximum sound pressure level (SPL dB)
    """
    return SPL_ref + 10*np.log10(P_rms/P_ref)


def power_for_target_spl(SPL_target, SPL_ref,  P_ref=1):
    """Required power for given max SPL
    """
    return P_ref * 10**((SPL_target-SPL_ref)/10)


def driver_spl_report(df, drivers, SPL_target=None):
    from common.plot import plot_styles
    plt.rcParams.update(plot_styles)
    plt.title("SPL vs. Power")
    plt.xlabel("SPL [dB]")
    plt.ylabel("Power [W]")
    plt.grid(which="minor", color='#222222', linestyle=':', linewidth=0.5)
    plt.vlines(SPL_target, 0, 600, linestyles="--", label="Target SPL")

    print(f"---- DRIVER SPL REPORT {SPL_target=} dB ----")
    for name in drivers:
        tweeter = df[df["Name"] == name]
        V_ref = float(tweeter["V_ref"].iloc[0])
        Z_ref = float(tweeter["Z_ref"].iloc[0])
        P_rms = float(tweeter["P_rms"].iloc[0])
        P_max = float(tweeter["P_max"].iloc[0])
        SPL_ref = float(tweeter["SPL_ref"].iloc[0])

        P_ref = (V_ref**2)/Z_ref
        SPL_rms = max_sound_pressure(SPL_ref, P_rms, P_ref)
        SPL_max = max_sound_pressure(SPL_ref, P_max, P_ref)
        if SPL_target:
            P_required = power_for_target_spl(SPL_target, SPL_ref, P_ref)

        print(f"- {name}:")
        print(f"    {SPL_max=:.2f} dB")
        print(f"    {SPL_rms=:.2f} dB")
        if SPL_target:
            print(f"    {P_required=:.2f} W")
        print("")

        desired = np.linspace(SPL_ref, SPL_rms, 1024)
        required = power_for_target_spl(desired, SPL_ref, P_ref)
        plt.plot(desired, required, label=name)

    plt.legend()
    plt.show()


def main():
    df = pd.read_csv(sys.argv[1])
    drivers = [
        # "Alcone AC 15",
        # "AMT U60W1.1-C",
        "AMT U160W1.1-R",
        # "Dayton Audio AMTHR-4",
        # "Dayton Audio AMTPRO-4",
        "Morel CAT 328-110",
        # "Morel EM 1308",
        # "Morel ET 448",
        "Dayton Audio RSS315HF-4",
        "Dayton Audio RSS390HF-4",
        "ScanSpeak Discovery 15M/4624G00",
    ]
    driver_spl_report(df, drivers, SPL_target=106)


if __name__ == "__main__":
    main()
