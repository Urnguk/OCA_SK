from matplotlib import pyplot as plt
import numpy as np


def fast_plot(x, y, domain=None, show=True):
    if domain == "TD":
        plt.xlabel("t, ns")
        plt.ylabel("I, a.u.")
    if domain == "FD":
        plt.xlabel(r"$\omega$, GHz")
        plt.ylabel("I, dB.")
        x = x[len(x) // 2:]
        y = y[len(y) // 2:]
        y = 10 * np.log10(y)
    plt.plot(x, y)
    plt.grid()
    if show:
        plt.show()


def multi_plot(data, n_rows=1, n_cols=2,
               show=True, subroutine=fast_plot):
    for i in range(len(data)):
        plt.subplot(n_rows, n_cols, i + 1)
        if subroutine == fast_plot:
            subroutine(*data[i], show=False)
        else:
            subroutine(data[i], show=False)
    if show:
        plt.show()


def eye_plot(eye_data, show=True):
    for X, Y in eye_data:
        plt.plot(X, Y, color="blue")
        plt.xlabel("t, ns")
        plt.ylabel("I, a.u.")
    plt.grid()
    if show:
        plt.show()



