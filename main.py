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

    original_seq, timescale, signal = sg.gen_signal(slots, tau, n_dots)
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

    V_pi = 2.0  # Typical MZM parameter
    P_optical = pc.mzm_modulate(signal)
    vs.eye_plot(an.eye_data(timescale, P_optical, slots, tau))

    V_clean, V_noisy = pc.photodetector(P_optical)

    vs.eye_plot(an.eye_data(timescale, V_noisy, slots, tau))
    
    # bits = original_seq
    # samples_per_bit = n_dots // slots
    # ber = an.calculate_ber(V_noisy, bits, samples_per_bit)
    # print(f"Estimated BER: {ber:.2e}")
    #
    # # 4. Visualization
    # data = [
    #     (timescale, signal, "TD"),
    #     (timescale, P_optical, "TD"),
    #     (timescale, V_noisy, "TD"),
    #     an.eye_data(timescale, V_noisy, slots, tau)
    # ]
    # vs.multi_plot(data[:3], n_rows=3, n_cols=1)
    # vs.eye_plot(data[3])








