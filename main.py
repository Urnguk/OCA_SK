import electrical_signal as els
import numpy as np
from matplotlib import pyplot as plt
from scipy import fft
import random as rm
from scipy.optimize import curve_fit


def gauss_func(t, mean, sigma):
    return 1 / (100 * sigma * (2 * np.pi) ** 0.5) * np.exp(-0.5 * (t - mean) ** 2 / sigma ** 2)


if __name__ == "__main__":
    tau = 1
    slots = 100
    n_dots = 2 ** 10
    dots = np.linspace(0, slots * tau, n_dots)

    seq = els.gen_sequence(slots)
    get_flat = els.flat_signal(seq, tau)
    get_nrz = els.nrz_signal(seq, tau)

    flat_y = [get_flat(x) for x in dots]
    nrz_y = [get_nrz(x) for x in dots]

    # plt.show()

    # delta_y = np.array([rm.gauss(0.5, 0.1) for i in range(n_dots)])
    # nrz_y += delta_y * 0.1





    nrz_w = fft.fft(nrz_y) ** 2
    freqs = fft.fftfreq(n_dots, 1/n_dots)
    plt.plot(freqs[:len(freqs) // 2], nrz_w[:len(freqs) // 2])
    plt.show()
    #

    # mean = slots // 2
    # sigma = 1
    #
    # get_gauss = els.gauss_signal(mean, sigma)
    # gauss_y = np.array([get_gauss(x) for x in dots])
    # gauss_y *= np.sin(100 * np.pi * dots)
    # gauss_w = fft.fft(gauss_y)

    # plt.subplot(2, 2, 1)
    # plt.plot(dots, gauss_y)
    # plt.subplot(2, 2, 2)
    # plt.plot(freqs, np.abs(gauss_w) ** 2)
    # plt.subplot(2, 2, 3)
    # plt.plot(dots, np.arctan(np.imag(gauss_y) / np.real(gauss_y)))
    # plt.subplot(2, 2, 4)
    # plt.plot(freqs, np.arctan(np.imag(gauss_w) / np.real(gauss_w)))
    # eye_data = []
    # for i in range(slots - 2):
    #     X = []
    #     Y = []
    #     for j in range(len(dots)):
    #         if i <= dots[j] < i + 3:
    #             X.append(dots[j] - i)
    #             Y.append(nrz_y[j])
    #     eye_data.append((X, Y))
    # for X, Y in eye_data:
    #     plt.plot(X, Y)
    # plt.show()
    #
    # n_bins = 100
    # bins = [0 for i in range(n_bins)]
    #
    #
    # for X, Y in eye_data:
    #     for i in range(len(X)):
    #         if 1.25 < X[i] < 1.75 and Y[i] > 0.55:
    #             for j in range(n_bins):
    #                 if 0.1 * j / n_bins <= Y[i] - 1 < 0.1 * (j + 1) / n_bins:
    #                     bins[j] += 1
    #
    # bins = np.array(bins) / np.sum(bins)
    # bin_steps = np.array([i / n_bins for i in range(n_bins)])
    # popt, pcov = curve_fit(gauss_func, bin_steps, bins, bounds = (0, 1))
    # gauss_fit = [gauss_func(t, *popt) for t in bin_steps]
    # plt.plot(bin_steps, bins)
    # plt.plot(bin_steps, gauss_fit, color="red")
    # print(*popt)
    # plt.show()

    # for i in range(len(delta_y)):
    #     for j in range(n_bins):
    #         if j / n_bins <= delta_y[i] < (j + 1) / n_bins:
    #             bins[j] += 1
    # bins = np.array(bins) / 1024
    # bin_steps = np.array([i / n_bins for i in range(n_bins)])
    # popt, pcov = curve_fit(gauss_func, bin_steps, bins, bounds = (0, 1))
    # gauss_fit = [gauss_func(t, *popt) for t in bin_steps]
    # plt.plot(bin_steps, bins)
    # plt.plot(bin_steps, gauss_fit, color="red")
    # print(*popt)
    # plt.show()








