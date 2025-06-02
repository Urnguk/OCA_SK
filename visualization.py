from matplotlib import pyplot as plt
import numpy as np


def fast_plot(x, y, domain=None, show=True):
    plt.plot(x, y)
    if domain == "TD":
        plt.xlabel("t, ps")
    if domain == "FD":
        plt.xlabel(r"$\omega$, GHz")
    plt.grid()
    if show:
        plt.show()


def multi_plot(data, domains=None, n_rows=1, n_cols=2, show=True):
    if domains is None:
        domains = [None] * len(data)
    for i in range(len(data)):
        plt.subplot(n_rows, n_cols, i + 1)
        fast_plot(*data[i], domains[i], show=False)
    if show:
        plt.show()


def eye_plot(eye_data):
    for X, Y in eye_data:
        plt.plot(X, Y, color="blue")
    plt.grid()
    plt.show()



