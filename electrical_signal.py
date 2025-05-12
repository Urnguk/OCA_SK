import random as rm
import numpy as np
from matplotlib import pyplot as plt


def gen_sequence(length):
    return [rm.randint(0, 1) for i in range(length)]


def flat_signal(seq, tau):
    t_full = tau * len(seq)

    def curr_signal(t):
        if t < 0 or t >= t_full:
            return 0
        return seq[int(t // tau)]

    return curr_signal


def nrz_signal(seq, tau):
    seq += [0]
    t_full = tau * len(seq)
    seq += [0]


    def curr_signal(t):
        if t < 0 or t >= t_full:
            return 0
        i = int(t // tau)
        step = t % tau
        if tau / 4 <= step <= 3 * tau / 4:
            return seq[i]
        if step < tau / 4 and seq[i] == seq[i - 1] or 3 * tau / 4 < step and seq[i] == seq[i + 1]:
            return seq[i]
        alpha = 0.01276
        k = (4 * 2 * np.pi / tau) * np.cos(2 * np.pi * alpha - np.pi / 2)
        b = 0.5 - k * tau / 4
        if 3 * tau / 4 < step:
            step -= 3 * tau / 4
        else:
            step = tau / 4 - step
        res = 1 + np.sin(2 * np.pi * step / (tau / 4) - np.pi / 2) if step < (alpha * (tau / 4)) else k * step + b
        if seq[i] == 0:
            return res
        return 1 - res

    return curr_signal


if __name__ == "__main__":
    # alpha = 0.01276
    # tau = 25
    # k = (2 * np.pi / tau) * np.cos(2 * np.pi * alpha - np.pi / 2)
    # b = 0.5 - k * tau
    # X = np.linspace(0, tau, 1000)
    # Y = [1 + np.sin(2 * np.pi * x / tau - np.pi / 2) if x < alpha * tau else k * x + b for x in X]
    # plt.plot(X, Y)
    # plt.show()

    s = [0, 1, 0, 1]
    signal = nrz_signal(s, 100)
    t = np.linspace(0, 600, 10000)
    y = [signal(x) for x in t]
    plt.plot(t, y)
    plt.show()
