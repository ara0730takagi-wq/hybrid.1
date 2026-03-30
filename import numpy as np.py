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

    # default / optional parameters
    c_fiber: float = 2.0e5
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
    if params.NQPU==2:
        return 0
    return params.t_cnot + params.t_meas + (params.NQPU - 1) * tsignal(params.d_qpu, params.c_fiber)


def p_click_temp(params: HybridParams) -> float:
    # Eq. (29): temporal-mode success probability after frequency multiplexing
    if params.p_click_freq < 0 or params.p_click_freq > 1:
        raise ValueError("p_click_freq must be in [0,1].")
    return 1.0 - (1.0 - params.p_click_freq) ** params.Nfreq


def p_hybrid_el(params: HybridParams) -> float:
    # Eq. (30)
    p = p_click_temp(params) * params.p_load
    return min(max(p, 0.0), 1.0)


def p_hybrid_el_2(params: HybridParams) -> float:
    # Eq. (31): probability that at least two raw ELs are obtained
    p = p_hybrid_el(params)
    n = params.Ntemp
    val = 1.0 - (1.0 - p) ** n - n * p * (1.0 - p) ** (n - 1)
    return min(max(val, 0.0), 1.0)


def p_hybrid_el_d(params: HybridParams) -> float:
    # Eq. (32)
    val = p_hybrid_el_2(params) * params.p_ed
    return min(max(val, 0.0), 1.0)


# ------------------------------------------------------------
# Appendix D implementation
# ------------------------------------------------------------
def expected_trials_until_all_links_ready_appendix_d(num_links: int, p_link: float) -> float:
    """
    Appendix D:
    Each link succeeds independently with probability p_link per trial round.
    Let T_i ~ Geometric(p_link), then T_all = max(T_1,...,T_num_links).
    This function returns E[T_all].

    E[T_all]] = sum_{j=1}^{num_links} (-1)^(j+1) C(num_links,j) / (1-(1-p_link)^j)
    """
    if num_links <= 0:
        return 0.0
    if p_link <= 0:
        return np.inf
    if p_link >= 1:
        return 1.0

    s = 0.0
    for j in range(1, num_links + 1):
        denom = 1.0 - (1.0 - p_link) ** j
        if abs(denom) < 1e-15:
            return np.inf
        s += ((-1) ** (j + 1)) * comb(num_links, j) / denom
    return s


def expected_time_all_distilled_links_appendix_d(params: HybridParams) -> float:
    """
    Eq. (33) derived in Appendix D
    """
    num_links = params.NQPU - 1
    p_link = p_hybrid_el_d(params)
    return t_hybrid_el_d(params) * expected_trials_until_all_links_ready_appendix_d(num_links, p_link)


def expected_time_end_to_end(params: HybridParams) -> float:
    # Eq. (34)
    return expected_time_all_distilled_links_appendix_d(params) + t_merge(params)


def repetition_rate(params: HybridParams) -> float:
    # Eq. (35)
    t = expected_time_end_to_end(params)
    if not np.isfinite(t) or t <= 0:
        return 0.0
    return 1.0 / t


# ------------------------------------------------------------
# Temporary placeholder probability models
# NOTE:
# p_click_freq should be "single frequency mode" click probability.
# Do not include Nfreq here, because Eq. (29) already handles Nfreq.
# ------------------------------------------------------------
def toy_p_click_freq_single_mode(distance_km: float,
                                 eta_dec: float,
                                 eta_fiber: float) -> float:
    """
    Placeholder for per-frequency-mode remote-click probability.
    """
    fiber_trans = 10**(-eta_fiber * distance_km/10 )
    p = fiber_trans * eta_dec 
    return min(max(p, 0.0), 1.0)


def toy_p_load(eta_qfc: float, eta_qm: float, eta_dec: float) -> float:
    p = eta_qfc * eta_qm * eta_dec
    return min(max(p, 0.0), 1.0)


def toy_p_ed(const: float = 0.25) -> float:
    return min(max(const, 0.0), 1.0)


# ------------------------------------------------------------
# Optional: Monte Carlo check of Appendix D
# ------------------------------------------------------------
def mc_expected_trials_until_all_links_ready(num_links: int, p_link: float,
                                             num_samples: int = 10000,
                                             rng_seed: int = 0) -> float:
    """
    Monte Carlo validation of Appendix D formula.
    Each T_i ~ Geometric(p_link), and we estimate E[max_i T_i].
    """
    if num_links <= 0:
        return 0.0
    if p_link <= 0:
        return np.inf

    rng = np.random.default_rng(rng_seed)
    trials = []
    for _ in range(num_samples):
        # numpy geometric returns support {1,2,3,...}
        t_i = rng.geometric(p_link, size=num_links)
        trials.append(np.max(t_i))
    return float(np.mean(trials))


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
def plot_repetition_rate_vs_distance():
    distances = np.linspace(0, 1000, 100)
    NQPU_list = [2, 3, 4]
    colors = ["red", "green", "blue"]

    Ntemp = 10
    Nfreq = 100

    plt.figure(figsize=(8, 6))

    for color, NQPU in zip(colors, NQPU_list):
        rates = []

        for d_total in distances:
            d_qpu = d_total / (NQPU - 1)

            p_click_freq = toy_p_click_freq_single_mode(
                distance_km=d_qpu,
                eta_dec=0.9,
                eta_fiber=0.3
            )

            p_load = toy_p_load(
                eta_qfc=0.99,
                eta_qm=0.99,
                eta_dec=0.9
            )

            p_ed = toy_p_ed(0.25)

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

            rates.append(repetition_rate(params))

        plt.plot(distances, rates, color=color, label=f"NQPU = {NQPU}")

    plt.xlabel("Total distance (km)")
    plt.ylabel("Repetition rate (Hz)")
    plt.title("Hybrid repeater repetition rate vs total distance")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_repetition_rate_vs_distance()