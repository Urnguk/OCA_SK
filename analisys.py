import numpy as np
from scipy.special import erfc


def eye_data(t, y, slots=100, tau=100):
    res = []
    for slot in range(slots - 2):
        X, Y = [], []
        for i in range(len(t)):
            x = t[i] // tau
            if slot <= x < slot + 3:
                X.append(t[i] - slot * tau)
                Y.append(y[i])
            elif x >= slot + 3:
                break
        res.append((X, Y))
    return res


def calculate_ber(signal, bits, timescale, tau):


    # Separate 0s and 1s
    sig0 = signal[bit_idx == 0]
    sig1 = signal[bit_idx == 1]

    # Calculate statistics
    V0, V1 = np.median(sig0), np.median(sig1)
    sigma0, sigma1 = np.std(sig0), np.std(sig1)

    Q = abs(V1 - V0) / (sigma0 + sigma1)
    return 0.5 * erfc(Q / np.sqrt(2))



