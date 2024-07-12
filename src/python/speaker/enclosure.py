
def sealed_enclosure_volume(Vas, Qtc, Qts):
    """
    Vas: Equivalent compliance volume
    Qtc: Desired system Q, typically around 0.7 for a good balance between transient response and efficiency
    Qts: Total Q factor of the driver
    """
    return Vas/((Qtc/Qts)**2-1)


def ported_enclosure_volume(Vas, Fs, Fb):
    """
    Vas: Equivalent compliance volume
    Fs: Driver resonant frequency
    Fb: Tuning frequency
    """
    return (Vas*(0.7*Fs))/(Fb-0.7*Fs)


def port_length(Vb, Fb, D):
    """
    Port length for a ported speaker enclosure

    Vb: Enclosure volume
    Fb: Tuning frequency
    D: Port diameter
    """
    return (23562.5*(D**2))/((Fb**2) * Vb)


def main():
    Vas = 50
    Qtc = 0.707
    Qts = 0.4
    Fs = 30
    Fb = 35
    D = 10

    Vas = 197
    Qtc = 0.707
    Qts = 0.32
    Fs = 17
    Fb = 25
    D = 12

    Vas = 103.61
    Qtc = 0.707
    Qts = 0.35
    Fs = 28
    Fb = 32
    D = 10

    Vs = sealed_enclosure_volume(Vas, Qtc, Qts)
    Vp = ported_enclosure_volume(Vas, Fs, Fb)
    Lp = port_length(Vp, Fb, D)

    print(f"sealed: {Vs:.2f} l")
    print(f"ported: {Vp:.2f} l")
    print(f"port:   {Lp:.2f} cm")


if __name__ == "__main__":
    main()
