import generation as sg
import numpy as np
import visualization as vs

from scipy import fft


def fourier(signal, slots, tau, n_dots):
    signal_w = fft.fftshift(fft.fft(signal))
    w = fft.fftshift(fft.fftfreq(n_dots, tau * slots / n_dots))
    return w, signal_w


def inv_fft(signal_w):
    return fft.ifft(fft.ifftshift(signal_w))


# def LPF(freqs, slots=100, tau=100, n_dots=2**10):
#     dT = slots * tau / n_dots
#     W_max = 1 / dT
#
#     alpha = 0.07 * W_max
#     return (alpha + 1j * freqs) / (alpha ** 2 + freqs ** 2) / np.sqrt(2)

def LPF(freqs, tau_c=0.05):
    H = 1 / np.sqrt(1 + (2 * np.pi * freqs * tau_c)**2)
    return H









