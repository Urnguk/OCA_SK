import generation as sg
import visualization as vs
import processing as pc
import analisys as an

import numpy as np
from scipy import fft


if __name__ == "__main__":
    tau = 0.1
    slots = 100
    n_dots = 2 ** 10

    timescale, signal = sg.gen_signal(slots, tau, n_dots)
    omega, signal_omega = pc.fourier(signal, slots, tau, n_dots)
    lpf = pc.LPF(omega)
    new_signal = np.real(pc.inv_fft(signal_omega * lpf))
    data = [
        (timescale, signal, "TD"),
        (omega, np.abs(signal_omega), "FD"),
        (timescale, new_signal, "TD"),
        (omega, np.abs(signal_omega * lpf), "FD")
    ]
    vs.multi_plot(data, n_rows=2, n_cols=2)

    vs.fast_plot(omega, lpf)
    data = [
        an.eye_data(timescale, signal, slots, tau),
        an.eye_data(timescale, new_signal, slots, tau)
    ]
    vs.multi_plot(data, 1, 2, subroutine=vs.eye_plot)
    signal = new_signal







    # delta_y = np.array([rm.gauss(0.5, 0.1) for i in range(n_dots)])
    # nrz_y += delta_y * 0.1





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








