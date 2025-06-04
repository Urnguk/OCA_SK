import generation as sg
import numpy as np
import visualization as vs

from scipy import fft
from scipy.stats import truncnorm


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

def LPF(freqs, tau_c=0.02):
    H = 1 / np.sqrt(1 + (2 * np.pi * freqs * tau_c)**2)
    return H


def mzm_modulate(V_signal, V_pi=1.0, alpha=0.5, P0=1.0):
    V1 = V_signal / 2
    V2 = V1

    E_out = (alpha * np.cos(np.pi * V1 / V_pi) +
             (1 - alpha) * np.cos(np.pi * V2 / V_pi))

    return P0 * np.abs(E_out) ** 2


def photodetector(P_optical, R=0.9, BW=10e9, R_load=50):
    I_pd = R * P_optical  # Photocurrent in mA

    # Shot noise calculation
    q = 1.6e-19  # Electron charge
    sigma = np.sqrt(2 * q * I_pd * BW) * 1e3  # mA scale
    noise = np.zeros_like(I_pd)

    for i in range(len(I_pd)):
        if I_pd[i] <= 0:
            noise[i] = 0
        else:
            a = -I_pd[i] / sigma[i]
            noise[i] = truncnorm.rvs(a, np.inf, scale=sigma[i])

    I_noisy = I_pd + noise
    return I_pd * R_load, I_noisy * R_load



