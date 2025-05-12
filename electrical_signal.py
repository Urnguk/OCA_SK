import random as rm
import numpy as np
from matplotlib import pyplot as plt


def gen_sequence(length):
    return [rm.randint(0, 1) for i in range(length)]

def gauss_signal(mean, sigma):

    def curr_signal(t):
        return 1 / (sigma * (2 * np.pi) ** 0.5) * np.exp(-0.5 * (t - mean) ** 2 / sigma ** 2)

    return curr_signal

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
        local_w = 2 * np.pi / tau
        if 3 * tau / 4 < step:
            step -= 3 * tau / 4
        else:
            step = tau / 4 - step
        res = 0.5 - 0.5 * np.cos(local_w * step)
        if seq[i] == 0:
            return res
        return 1 - res

    return curr_signal


if __name__ == "__main__":
    s = [0, 1, 0, 1, 1, 0, 0]
    signal = nrz_signal(s, 100)
    t = np.linspace(0, 800, 10000)
    y = [signal(x) for x in t]
    plt.plot(t, y)
    plt.show()
