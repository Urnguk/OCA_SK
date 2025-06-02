import generation as sg
import numpy as np
from scipy import fft


# def LPF(n_dots, signal, slots=100, tau=1):
#     signal_w = fft.fft(signal)
#     dT = slots * tau / n_dots
#     W_max = 1 / dT
#     freqs = fft.fftfreq(n_dots, dT)
#     freqs = freqs
#     alpha = 0.07 * W_max
#     return (alpha + 1j * freqs) / (alpha ** 2 + freqs ** 2) / np.sqrt(2)







