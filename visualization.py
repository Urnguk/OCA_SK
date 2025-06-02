from matplotlib import pyplot as plt


def fast_plot(x, y, domain=None, show=True):
    plt.plot(x, y)
    if domain == "TD":
        plt.xlabel("t, ps")
    if domain == "FD":
        plt.xlabel(r"$\omega$, GHz")
    plt.grid()
    if show:
        plt.show()


def eye_plot(eye_data):
    for X, Y in eye_data:
        plt.plot(X, Y, color="blue")
    plt.grid()
    plt.show()



