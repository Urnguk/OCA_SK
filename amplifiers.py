import numpy as np
from functools import cache
from matplotlib import pyplot as plt


class RamanAmp:
    def __init__(self, lambdas=None, gamma=0.2, delta_lambda=10, num_channels=100, left_border=None, right_border=None):
        if lambdas is None:
            lambdas = [1530, 1540, 1550, 1565]  # mkm
        if left_border is None:
            left_border = min(lambdas)
            right_border = max(lambdas)
        self._len = len(lambdas)
        self.lambdas = np.array(lambdas, dtype=np.float64)
        self.powers = np.array([300 for i in range(self._len)], dtype=np.float64)  # mW
        self.delta_lambda = delta_lambda  # mkm

        self.gamma_db = gamma  # db / km
        self.gamma = 10 ** (self.gamma_db / 10)  # 1 / km
        self.dist = 30  # km

        self.num_channels = num_channels
        self.channels = np.linspace(left_border, right_border, self.num_channels)  # mkm

        self.g_0 = self.gamma ** 2 * self.dist / (self.powers[0] * (1 - np.exp(-self.gamma * self.dist)))
        self.m = np.zeros(shape=(self.num_channels, self._len), dtype=np.float64)
        self.m_is_set = False  # init already overloaded, let us init M separately
        self.saved_G = None

    @cache
    def g(self, lambd, channel):
        return self.g_0 / (1 + ((channel - lambd) / self.delta_lambda) ** 2)

    def set_m(self):
        for k in range(self.num_channels):
            for i in range(self._len):
                self.m[k, i] = self.g(self.lambdas[i], self.channels[k])
        self.m_is_set = True

    def G(self):
        if not self.m_is_set:
            self.set_m()
        self.saved_G = self.m @ self.powers
        return self.saved_G

    def tune(self, G_ideal=65):
        G = self.G() if self.saved_G is None else self.saved_G
        dG = G - G_ideal
        m = self.m
        mT = self.m.T
        dP = np.linalg.solve(mT @ m, mT @ dG)
        self.powers -= dP



if __name__ == "__main__":
    raman = RamanAmp()
    plt.plot(raman.channels, raman.G(), label="initial")
    G_lim = 65
    raman.tune(G_lim)
    plt.plot(raman.channels, raman.G(), color="red", label="ideal")
    plt.plot(raman.channels, [G_lim for i in range(raman.num_channels)], color="green", linestyle="dashed", label="tuned")
    plt.grid()
    plt.legend()
    plt.show()













