import electrical_signal as els
import numpy as np
from matplotlib import pyplot as plt
from scipy import fft
import random as rm
from scipy.optimize import curve_fit
import electrical_signal as els

def LPF(n_dots, signal, slots=100, tau=1):
    signal_w = fft.fft(signal)
    dT = slots * tau / n_dots
    W_max = 1 / dT
    freqs = fft.fftfreq(n_dots, dT)
    freqs = freqs
    alpha = 0.07 * W_max
    return (alpha + 1j * freqs) / (alpha ** 2 + freqs ** 2) / np.sqrt(2)


if __name__ == "__main__":
    n_dots = 2 ** 10
    seq = els.gen_sequence(2 ** 10)
    dots = np.linspace(0, 100, 2 ** 10)
    gen_flat = els.flat_signal(seq, 1)
    slots = 100
    tau = 1
    dT = slots * tau / n_dots
    freqs = fft.fftfreq(n_dots, dT)
    freqs = freqs
    signal_flat = np.array([gen_flat(x) for x in dots])
    signal_flat_w = fft.fft(signal_flat)

    lpf = LPF(2 ** 10, signal_flat)
    signal_flat_w *= lpf

    signal_back = fft.ifft(signal_flat_w)
    plt.plot(signal_back)
    plt.show()





