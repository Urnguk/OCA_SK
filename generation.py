import random as rm
import numpy as np
from matplotlib import pyplot as plt


def gen_sequence(length):
    if length < 2:
        raise ValueError(f"sequence length {length} < 2")
    return [0] + [rm.randint(0, 1) for i in range(length - 2)] + [0]


def gauss_signal(t, mean, sigma):
    return 1 / (sigma * (2 * np.pi) ** 0.5) * np.exp(-0.5 * (t - mean) ** 2 / sigma ** 2)


def flat_signal(t, seq, tau):
    t_full = tau * len(seq)
    if t < 0 or t >= t_full:
        return 0
    return seq[int(t // tau)]



def nrz_signal(t, seq, tau):
    t_full = tau * len(seq)
    seq = seq + [0]
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



if __name__ == "__main__":
    seq = gen_sequence(100)
    tau = 20
    timescale = np.linspace(0, len(seq) * tau, 10 ** 6)
    y = [nrz_signal(t, seq, tau) for t in timescale]
    plt.plot(timescale, y)
    plt.show()
