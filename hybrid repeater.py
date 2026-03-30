import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from math import comb
import math


@dataclass
class HybridParams:
    # topology
    NQPU: int
    Ntemp: int
    Nfreq: int

    # times [seconds]
    t_spdc: float
    t_atom: float
    t_cnot: float
    t_meas: float
    d_qpu: float

    # probabilities
    p_click_freq: float
    p_load: float
    p_ed: float

    # defaults
    c_fiber: float = 2.0e5

    # optional physical parameters
    q: float = 0.5
    lam: float = 0.15
    eta_qfc: float = 0.99
    eta_qm: float = 0.99
    eta_fiber: float = 0.3
    eta_afc: float = 0.3
    eta_dec: float = 0.9


def tsignal(d_qpu_km: float, c_fiber_km_per_s: float = 2.0e5) -> float:
    return d_qpu_km / c_fiber_km_per_s


def t_ed(params: HybridParams) -> float:
    return params.t_cnot + params.t_meas + tsignal(params.d_qpu, params.c_fiber)


def t_qm_try(params: HybridParams) -> float:
    return params.Ntemp * params.t_spdc + tsignal(params.d_qpu, params.c_fiber)


def t_load(params: HybridParams) -> float:
    return params.Ntemp * (params.t_atom + tsignal(params.d_qpu, params.c_fiber))


def t_hybrid_el_try(params: HybridParams) -> float:
    return t_qm_try(params) + t_load(params)


def t_hybrid_el_d(params: HybridParams) -> float:
    return t_hybrid_el_try(params) + t_ed(params)


def t_merge(params: HybridParams) -> float:
    #QPUが二つの時はマージする必要なし
    if params.NQPU==2:
        return 0
    return params.t_cnot + params.t_meas + (params.NQPU - 1) * tsignal(params.d_qpu, params.c_fiber)
    #return 1.0


def p_click_temp(params: HybridParams) -> float:
    return 1.0 - (1.0 - params.p_click_freq) ** params.Nfreq


def p_hybrid_el(params: HybridParams) -> float:
    return p_click_temp(params) * params.p_load


def p_hybrid_el_2(params: HybridParams) -> float:
    p = p_hybrid_el(params)
    n = params.Ntemp
    return 1.0 - (1.0 - p) ** n - n * p * (1.0 - p) ** (n - 1)


def p_hybrid_el_d(params: HybridParams) -> float:
    return p_hybrid_el_2(params) * params.p_ed


def expected_time_all_distilled_links(params: HybridParams) -> float:
    p = p_hybrid_el_d(params)
    m = params.NQPU - 1

    if p <= 0 or m <= 0:
        return np.inf

    s = 0.0
    for j in range(1, m + 1):
        denom = 1.0 - (1.0 - p) ** j
        if abs(denom) < 1e-15:
            return np.inf
        s += ((-1) ** (j + 1)) * comb(m, j) / denom

    return t_hybrid_el_d(params) * s


def expected_time_end_to_end(params: HybridParams) -> float:
    return expected_time_all_distilled_links(params) + t_merge(params)


def repetition_rate(params: HybridParams) -> float:
    t = expected_time_end_to_end(params)
    if not np.isfinite(t) or t <= 0:
        return 0.0
    return 1.0 / t


def toy_p_click_freq(distance_km: float, Nfreq: int, eta_dec: float, eta_fiber: float,
                     alpha_db_per_km: float = 0.2) -> float:
    x = 10**(-alpha_db_per_km * distance_km/10) * eta_dec 
    return x


def toy_p_load(eta_qfc: float, eta_qm: float, eta_dec: float) -> float:
    return eta_qfc * eta_qm * eta_dec


def toy_p_ed(const: float = 0.05) -> float:
    return const


def plot_repetition_rate_vs_distance():
    distances = np.linspace(0, 1000, 100)
    NQPU_list=[2,3,4]
    colors=["red","yellow","blue"]
    Ntemp = 10
    Nfreq = 100
    plt.figure(figsize=(7, 5))
    for idx,NQPU in enumerate(NQPU_list):
        rates=[]
        times=[]
        print(f"QPU={NQPU}")
        for d in distances:
            d_qpu = d / (NQPU - 1)
            p_click_freq = toy_p_click_freq(
                distance_km=d_qpu,
                Nfreq=Nfreq,
                eta_dec=0.9,
                eta_fiber=0.3
                )
            p_load = toy_p_load(
                eta_qfc=0.99,
                eta_qm=0.99,
                eta_dec=0.9
                )
            p_ed = toy_p_ed(0.05)
            
            params = HybridParams(
                NQPU=NQPU,
                Ntemp=Ntemp,
                Nfreq=Nfreq,
                t_spdc=1e-6,
                t_atom=1e-4,
                t_cnot=1e-4,
                t_meas=1e-4,
                d_qpu=d_qpu,
                p_click_freq=p_click_freq,
                p_load=p_load,
                p_ed=p_ed
                )
            t_end=expected_time_all_distilled_links(params)
            r_rep=repetition_rate(params)
            rates.append(repetition_rate(params))
            times.append(expected_time_end_to_end(params))
            print(f"t_end={t_end}")
        plt.plot(
            distances,
            rates,
            color=colors[idx],
            label=f"NQPU={NQPU}"
        )
    plt.xlabel("Distance (km)")
    plt.ylabel("Repetition rate (Hz)")
    plt.title("Hybrid repeater repetition rate vs distance")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_repetition_rate_vs_distance()