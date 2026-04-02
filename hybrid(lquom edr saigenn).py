import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from math import comb

#LQUOMの再現
"""
LQUOMの再現をするために、
ARC(QR-QST-EL-QST-QR)にたいして、１回の試行で成功する確率
P_{gen}^{ARC}(n)=(\eta_QST)^2 *(p_{gen}^{EL})
一回のラウンドで全リンク準備完了する確率
p_{all}=(p_{gen}^{ARC})^N
各ラウンド数 T_i~Geom(p_{gen}^{ARC})とする。ARC-R全体が準備完了するまでのラウンド数は
T_{max}=max(T_1,...,T_N)
E[T_{max}]=\sum_{j=1}^{N}(-1)^{j+1} comb(N,j) 1/(1-(1-p_{gen}^{ARC})^j)
E[t_end]=t_{round}E[T_max]+t_merge,(t_round=t_QST+t_ARC,t_merge=t_cnot)
repition rate=1/E[t_end]
"""
@dataclass
class LQUOMExpectedTimeParams:
    # topology
    N: int                 # number of ARCs in ARC-R
    n: int = 1             # number of ELs inside one ARC (blue-box reproduction uses n=1)

    # distance
    l_km: float = 20.0     # EL length [km]

    # device parameters
    eta_BSM: float = 0.32
    eta_DET: float = 0.90
    Gamma_f: int = 50
    eta_AFC: float = 0.30
    eta_QST: float = 0.90 * 0.70 * 0.30 * 0.90   # eta_loss * eta_shift * eta_interface * eta_AOM
    alpha_db_per_km: float = 0.2

    # timing
    t_QST: float = 10e-6
    t_AFC: float = 100e-6
    t_CNOT: float = 10e-6

    # optional
    c_fiber: float = 2.0e5   # km/s

    # protocol choice
    x: int = 2               # bell-pair generation sideなら x=2 をまず試す


# ------------------------------------------------------------
# utilities
# ------------------------------------------------------------
def clamp01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


def eta_fiber_db(distance_km: float, alpha_db_per_km: float = 0.2) -> float:
    """
    Fiber transmission for alpha given in dB/km:
        eta = 10^(- alpha * distance / 10)
    """
    return np.exp(-alpha_db_per_km * distance_km )


# ------------------------------------------------------------
# LQUOM probability model
# ------------------------------------------------------------
def p_el_gen(params: LQUOMExpectedTimeParams) -> float:
    """
    LQUOM report style elementary-link generation probability:
        p_EL = [1 - (1 - e^{-alpha l} * eta_BSM^(x-1) * eta_DET^x)^Gamma_f] * eta_AFC^2

    Here fiber is implemented in dB/km form:
        eta_fiber = 10^(-alpha*l/10)
    """
    eta_fiber = eta_fiber_db(params.l_km, params.alpha_db_per_km)
    p_single_mode = eta_fiber * (params.eta_BSM ** (params.x - 1)) * (params.eta_DET ** params.x)
    p_single_mode = clamp01(p_single_mode)

    p = (1.0 - (1.0 - p_single_mode) ** params.Gamma_f) * (params.eta_AFC ** 2)
    return clamp01(p)


def eta_ec(params: LQUOMExpectedTimeParams) -> float:
    """
    EC efficiency.
    Minimal toy choice for Ca-trap/LQUOM-style simplified chain:
        eta_EC = eta_BSM * eta_DET * eta_shift
    If you already used a different EC expression in your LQUOM-only code,
    replace only this function body with that exact expression.
    """
    eta_shift = 0.70
    return clamp01(params.eta_BSM * params.eta_DET * eta_shift)


def p_arc_gen(params: LQUOMExpectedTimeParams) -> float:
    """
    ARC success probability:
        p_ARC = (eta_QST)^2 * (p_EL)^n * (eta_EC)^(n-1)
    For blue-box reproduction, n=1, so:
        p_ARC = (eta_QST)^2 * p_EL
    """
    p_el = p_el_gen(params)
    p = (params.eta_QST ** 2) * (p_el ** params.n) * (eta_ec(params) ** (params.n - 1))
    return clamp01(p)


# ------------------------------------------------------------
# expected-trial formula (keep Appendix D skeleton)
# ------------------------------------------------------------
def expected_trials_until_all_links_ready_appendix_d(num_links: int, p_link: float) -> float:
    """
    T_i ~ Geometric(p_link), T_all = max(T_1, ..., T_num_links)
    E[T_all]] = sum_{j=1}^{num_links} (-1)^(j+1) C(num_links,j) / (1-(1-p_link)^j)
    """
    if num_links <= 0:
        return 0.0
    if p_link <= 0.0:
        return np.inf
    if p_link >= 1.0:
        return 1.0
    nattempts=9091
    p_EL1=1-(1-p_link)**nattempts
    s = 0.0
    for j in range(1, num_links + 1):
        denom = 1.0 - (1.0 - p_link) ** j
        if abs(denom) < 1e-15:
            return np.inf
        s += ((-1) ** (j + 1)) * comb(num_links, j) / denom
    return s


# ------------------------------------------------------------
# LQUOM-style time model for "1 / expected time"
# ------------------------------------------------------------
def t_round_arc(params: LQUOMExpectedTimeParams) -> float:
    """
    Use one ARC trial duration as the round time.
    Simplest consistent choice for expected-time method:
        t_round = t_QST + t_AFC + t_CNOT
    If your LQUOM-only code uses another exact t_gen expression,
    replace only this function.
    """
    return params.t_QST + params.t_AFC 


def t_merge_arc_r(params: LQUOMExpectedTimeParams) -> float:
    """
    End correction after all ARCs are ready.
    Borrow the same structural idea as your current code:
        t_merge ~ t_CNOT + signal propagation over ARC-R
    Since total distance is N*l for n=1 blue-box case, use that.
    """
    total_distance_km = params.N * params.l_km
    return params.t_CNOT #+ total_distance_km / params.c_fiber


def expected_time_all_arcs_ready(params: LQUOMExpectedTimeParams) -> float:
    """
    Replace distilled-link block by ARC block.
    """
    p_link = p_arc_gen(params)
    num_links = params.N
    return t_round_arc(params) * expected_trials_until_all_links_ready_appendix_d(num_links, p_link)


def expected_time_end_to_end(params: LQUOMExpectedTimeParams) -> float:
    return expected_time_all_arcs_ready(params) + t_merge_arc_r(params)


def repetition_rate(params: LQUOMExpectedTimeParams) -> float:
    """
    "repetition_rate" name is intentionally kept,
    but now means: 1 / expected end-to-end time
    with LQUOM ARC probability model.
    """
    t = expected_time_end_to_end(params)
    if not np.isfinite(t) or t <= 0.0:
        return 0.0
    return 1.0 / t


# ------------------------------------------------------------
# plot: fixed EL length = 20 km, vary number of ARCs N
# ------------------------------------------------------------
def plot_repetition_rate_vs_N_fixed_el():
    N_list = np.arange(1, 11)   # N = 1,...,10
    rates = []
    total_distances = []

    for N in N_list:
        params = LQUOMExpectedTimeParams(
            N=N,
            n=1,               # important for blue-box comparison
            l_km=20.0,
            eta_BSM=0.32,
            eta_DET=0.90,
            Gamma_f=50,
            eta_AFC=0.30,
            eta_QST=0.90 * 0.70 * 0.30 * 0.90,
            alpha_db_per_km=0.2,
            t_QST=10e-6,
            t_AFC=100e-6,
            t_CNOT=10e-6,
            x=2,
        )

        rates.append(repetition_rate(params))
        total_distances.append(N * params.l_km)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(total_distances, rates, marker="x", label="1 / expected time (LQUOM probability)")
    ax.set_xlabel("ARC-R Distance (km)")
    ax.set_ylabel("Entanglement Distribution Rate (1/s)")
    ax.set_title("Expected-time version of LQUOM ARC-R rate (n = 1, varying N)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

    print("N, total_distance_km, rate_Hz")
    for N, d, r in zip(N_list, total_distances, rates):
        print(f"{N:2d}, {d:6.1f}, {r:.6e}")


if __name__ == "__main__":
    plot_repetition_rate_vs_N_fixed_el()