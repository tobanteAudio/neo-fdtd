import numpy as np

from speaker.driver import DRIVERS
from speaker.enclosure import sealed_enclosure_volume, ported_enclosure_volume


def max_sound_pressure(SPL_ref, P_max, P_ref=1):
    return SPL_ref + 10*np.log10(P_max/P_ref)


def required_power(SPL_desired, SPL_ref,  P_ref=1):
    """Required power for given max SPL
    """
    return P_ref * 10**((SPL_desired-SPL_ref)/10)


def main():
    drivers = [
        # 'ScanSpeak_Discovery_15M_4624G00',
        'ScanSpeak_Discovery_26W_4558T00',
        'ScanSpeak_Discovery_30W_4558T00',
        'ScanSpeak_Revelator_32W_4878T01',
        'ScanSpeak_Revelator_32W_4878T00',
    ]

    for name in drivers:
        driver = DRIVERS[name]

        Fs = driver["Fs"]
        P_ref = (driver["V_ref"]**2)/driver["Z_ref"]
        P_max = driver["P_max"]
        Qts = driver["Qts"]
        SPL_ref = driver["SPL_ref"]
        Vas = driver["Vas"]

        Fb = Fs*1.2
        Qtc = 0.707
        SPL_desired = 105

        P_required = required_power(SPL_desired, SPL_ref, P_ref)
        SPL_max = max_sound_pressure(SPL_ref, P_max, P_ref)
        Vs = sealed_enclosure_volume(Vas, Qtc, Qts)
        Vp = ported_enclosure_volume(Vas, Fs, Fb)

        print(f"--- {name} ---")
        print(f"{Fs=:.2f} Hz")
        print(f"{Fb=:.2f} Hz")
        print(f"{SPL_ref=:.2f} dB")
        print(f"{SPL_max=:.2f} dB")
        print(f"{SPL_desired=:.2f} dB")
        print(f"{P_max=:.2f} W")
        print(f"{P_required=:.2f} W")
        print(f"{Vs=:.2f} litre")
        print(f"{Vp=:.2f} litre")
        print(f"{Fb/Fs=:.2f}")
        print("")


if __name__ == "__main__":
    main()
