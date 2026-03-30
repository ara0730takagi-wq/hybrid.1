import numpy as np
import pandas
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import os
def trial_methodB_timeline(p_EL, p_EC, p_QST, rng=None):
    """
    Method B (N=1, n=2) の 1回試行を
    t = 0, t_AFC/4, t_AFC/2, t_AFC で記録。
    EC と QST の「判定したかどうか(QST_judged)」も持つ。
    """
    if rng is None:
        rng = np.random.default_rng()

    tl = {"t0": {}, "t_quarter": {}, "t_half": {}, "t_full": {}}

    # t = 0 : EPPS 発射（ニュートラル）
    tl["t0"]["EPPS_fired"] = True

    # t = t_AFC/4 : EL 内 BSM で成功/失敗決定（乱数 p_EL）
    EL1_success = (rng.random() < p_EL)
    EL2_success = (rng.random() < p_EL)
    tl["t_quarter"]["EL1_success"] = EL1_success
    tl["t_quarter"]["EL2_success"] = EL2_success

    # t = t_AFC/2 : ヘラルドが EC と隣接 QST に届く
    # EC が受け取るヘラルド（True: 成功, False: 失敗）
    EC_msg_from_EL1 = EL1_success
    EC_msg_from_EL2 = EL2_success
    tl["t_half"]["EC_msg_from_EL1"] = EC_msg_from_EL1
    tl["t_half"]["EC_msg_from_EL2"] = EC_msg_from_EL2

    # QST1_1 は EL1_1 から, QST1_2 は EL1_2 から
    QST1_msg_from_EL1 = EL1_success
    QST2_msg_from_EL2 = EL2_success
    tl["t_half"]["QST1_msg_from_EL1"] = QST1_msg_from_EL1
    tl["t_half"]["QST2_msg_from_EL2"] = QST2_msg_from_EL2

    # EL の状態も t_half にコピー
    tl["t_half"]["EL1_success"] = EL1_success
    tl["t_half"]["EL2_success"] = EL2_success

    # EC の判定：両方成功ヘラルドが来たときだけ判定
    if EC_msg_from_EL1 and EC_msg_from_EL2:
        EC_judged = True
        EC_success = (rng.random() < p_EC)
    else:
        EC_judged = False
        EC_success = False  # 成功していない扱い

    tl["t_half"]["EC_judged"] = EC_judged
    tl["t_half"]["EC_success"] = EC_success

    # t = t_AFC : 反対側の QST にもヘラルドが届く
    QST1_msg_from_EL2 = EL2_success  # EL1_2 → QST1_1
    QST2_msg_from_EL1 = EL1_success  # EL1_1 → QST1_2
    tl["t_full"]["QST1_msg_from_EL2"] = QST1_msg_from_EL2
    tl["t_full"]["QST2_msg_from_EL1"] = QST2_msg_from_EL1

    # 便利のため色々コピー
    tl["t_full"]["EL1_success"] = EL1_success
    tl["t_full"]["EL2_success"] = EL2_success
    tl["t_full"]["EC_judged"] = EC_judged
    tl["t_full"]["EC_success"] = EC_success
    tl["t_full"]["QST1_msg_from_EL1"] = QST1_msg_from_EL1
    tl["t_full"]["QST2_msg_from_EL2"] = QST2_msg_from_EL2

    # QST 判定条件：
    #   両 EL のヘラルド成功 & EC 成功 のときだけ判定
    pre_QST = EC_success and EL1_success and EL2_success
    tl["t_full"]["QST_judged"] = pre_QST

    if pre_QST:
        QST1_success = (rng.random() < p_QST)
        QST2_success = (rng.random() < p_QST)
    else:
        QST1_success = False
        QST2_success = False

    tl["t_full"]["QST1_success"] = QST1_success
    tl["t_full"]["QST2_success"] = QST2_success

    # ARC 全体の成功：QST1, QST2 両方成功
    ARC_success = pre_QST and QST1_success and QST2_success
    tl["t_full"]["ARC_success"] = ARC_success

    return tl,ARC_success
def estimate_p_connect_given_tau(
    tau_total: float,
    *,
    omega_epps: float,     # omega = eta_EPPS * R_EPPS [1/s]
    t_gen: float,          # t_gen = t_QST + t_ARC + t_CNOT [s]
    p_EL: float,
    p_EC: float,
    p_QST: float,
    num_big_trials: int = 20000,
    seed: int = 0,
):
    """
    tau_total 秒の待ち時間で「少なくとも1回」QR接続に成功する確率 p を Monte Carlo 推定する。
    """
    # 試行できる attempt 回数（整数）
    budget = tau_total - t_gen
    if budget <= 0:
        return 0.0, 0.0, 0  # p, se, M

    M = int(np.floor(omega_epps * budget))
    if M <= 0:
        return 0.0, 0.0, 0

    rng = np.random.default_rng(seed)

    successes = 0
    for _ in range(num_big_trials):
        # 1 big trial: 最大 M attempts のうち1回でも成功したら成功
        for _ in range(M):
            _, ARC_success = trial_methodB_timeline(p_EL=p_EL, p_EC=p_EC, p_QST=p_QST, rng=rng)
            if ARC_success:
                successes += 1
                break

    p_hat = successes / num_big_trials
    se = np.sqrt(p_hat * (1.0 - p_hat) / num_big_trials)
    return p_hat, se, M


def find_tau_for_target_prob(
    target_p: float = 0.95,
    *,
    omega_epps: float,
    t_gen: float,
    p_EL: float,
    p_EC: float,
    p_QST: float,
    num_big_trials: int = 20000,
    seed: int = 0,
    tau_max: float | None = None,  # 例: QRメモリで上限があるなら t_QR を入れる
    tol_tau: float = 1e-4,         # τの許容誤差[s]
    max_iter: int = 40,
):
    """
    p(tau)=target_p となる最小の tau を二分探索で探す（MC推定）。
    """
    # まず上限側を見つける（倍々で増やす）
    tau_lo = t_gen
    tau_hi = t_gen + 1e-6  # ほぼ t_gen から開始

    # tau_hi を増やして p>=target を満たす点を探す
    while True:
        if tau_max is not None and tau_hi > tau_max:
            tau_hi = tau_max
            p_hi, se_hi, M_hi = estimate_p_connect_given_tau(
                tau_hi,
                omega_epps=omega_epps, t_gen=t_gen,
                p_EL=p_EL, p_EC=p_EC, p_QST=p_QST,
                num_big_trials=num_big_trials, seed=seed
            )
            # 上限でも届かないなら「達成不能」
            if p_hi < target_p:
                return None, (p_hi, se_hi, M_hi)
            break

        p_hi, se_hi, M_hi = estimate_p_connect_given_tau(
            tau_hi,
            omega_epps=omega_epps, t_gen=t_gen,
            p_EL=p_EL, p_EC=p_EC, p_QST=p_QST,
            num_big_trials=num_big_trials, seed=seed
        )
        if p_hi >= target_p:
            break
        # まだ足りない → 倍に伸ばす
        tau_hi = t_gen + 2.0 * (tau_hi - t_gen) if (tau_hi > t_gen) else (t_gen + 1e-3)

    # 二分探索（MCはノイズがあるので、iter回数で止める）
    for it in range(max_iter):
        if tau_hi - tau_lo <= tol_tau:
            break

        tau_mid = 0.5 * (tau_lo + tau_hi)

        # 乱数相関を減らしたいなら、tauごとに seed を変える
        p_mid, se_mid, M_mid = estimate_p_connect_given_tau(
            tau_mid,
            omega_epps=omega_epps, t_gen=t_gen,
            p_EL=p_EL, p_EC=p_EC, p_QST=p_QST,
            num_big_trials=num_big_trials,
            seed=seed + 1000 + it
        )

        if p_mid >= target_p:
            tau_hi = tau_mid
        else:
            tau_lo = tau_mid

    # 最終推定
    p_final, se_final, M_final = estimate_p_connect_given_tau(
        tau_hi,
        omega_epps=omega_epps, t_gen=t_gen,
        p_EL=p_EL, p_EC=p_EC, p_QST=p_QST,
        num_big_trials=num_big_trials, seed=seed + 99999
    )
    return tau_hi, (p_final, se_final, M_final)
def main():
    """
    This python script calculates the minimum rate of entanglement distribution between two QRs connected by N ARC chains, each chain composed of
    n individual elementary links. The proposed scheme includes ARCs, buffers and QRs, as detailed in the paper "Entanglement distribution in multi
    platform buffered-router-assisted frequency-multiplexed automated repeater chains", Askarani et al.

    Inputs:
    - "Parameters.xlsx" containing all efficiencies and time durations that are required.

    Outputs:
    - Minimum entanglement distribution rate given a chosen scheme.

    Planned functionality:
    - ARC-R (N length ARC chain) EDR calculation [Calcium ion trap setup]
    - Comparison with no buffer case
    - A and B configuration comparison
    - Other QM comparison (e.g. Rubidium and Yb)

    Current functionality:
    - ARC-R (N length ARC chain) EDR calculation [Askarani proposed NV and GEM setup]


    This simulation is written for the sake of LQUOM.
    """

    def load_params_dict(arch, arc_method, param_set=None):
        """
        Looks at the file named "Parameters.xlsx" and loads the set of parameters designated by param_set
        into a dictionary for use in the simulation.

        Calling the function with no designated param set returns a list of implemented parameter sets.
        """

        ##### Load presets from excel files
        if arch == "askarani":
            df_data = pandas.read_excel(r'C:\Users\ara07\Desktop\Parameters_askarani.xlsx')

            if param_set == 1:
                column_key = "askarani near term"
            elif param_set == 2:
                column_key = "askarani long term"
            else:
                raise ValueError("The parameter set you indicated does not exist.")
            
            param_list = df_data[column_key].to_list()
            dictkeys = ["eta_NV", "t_NV", "t_13C", "t_CNOT", "eta_13C", "eta_QFC_1588", "Gamma_t", "eta_BSM", "eta_DET", "Gamma_f", \
                        "eta_AFC", "t_AFC", "l", "eta_EPPS", "R_EPPS", "eta_BUFF", "eta_MAP", "t_BUFF_spin", "eta_shift", "eta_pol", \
                        "eta_QFC_637", "alpha", "t_QR"]#パラメータの説明はレポートに記載あり
            
        elif arch == "ca":
            df_data = pandas.read_excel(r'C:\Users\ara07\Desktop\Parameters_ca_trap.xlsx')

            if param_set == 1:
                column_key = "lquom"
            elif param_set == 2:
                column_key = "askarani_nearterm"
            else:
                raise ValueError("The parameter set you indicated does not exist.")

            param_list = df_data[column_key].to_list()
            dictkeys = ["t_Ca43", "t_CNOT", "eta_BSM", "eta_DET", "Gamma_f", "eta_AFC", "t_AFC", "l", "eta_EPPS", "R_EPPS", "eta_BUFF", "t_buff", "eta_shift", "alpha" \
                        ,"eta_aom", "eta_interface", "t_QR"]
        
        loaded_dict = dict(zip(dictkeys, param_list))

        
        ##### Load ARC variables depending on chosen arc_method

        if arc_method == "A":
            loaded_dict["n"] = np.arange(1,11,1) 
            #loaded_dict["n"] = 1
            loaded_dict["big_N"] = 1
            #loaded_dict["big_N"] = np.arange(1,11,1)
            loaded_dict["eps"] = 0.05#entanngle/second(目標失敗率\epsilon)
            loaded_dict["eta_loss"] = 0.9                        # Generic fiber loss coefficient
            
        elif arc_method == "B":
            loaded_dict["n"] = np.arange(1,11,1) 
            #loaded_dict["n"] = 1
            loaded_dict["big_N"] = 1
            #loaded_dict["big_N"] = np.arange(1,11,1)
            loaded_dict["eps"] = 0.05
            loaded_dict["eta_loss"] = 0.9                        # Generic fiber loss coefficient

            ##### Method B makes some adjustments to ARC parameters, perform that adjustment here.
            loaded_dict["big_N"] = loaded_dict["big_N"] + loaded_dict["big_N"] * (loaded_dict["n"] - 1)#methodBの基本リンクを分割した後のARCの総数
            print(f'===========================================> Check now we have big N equal to {loaded_dict["big_N"]} ')
            # loaded_dict["n"] = np.arange(2,7,1)              # We look at the effect of dividing each EL into 2/3/4/5/6 segments
            loaded_dict["n"] = 2                               # This number is actually xi and not n, but they both represent the same idea in this method.#分割数2であるのはレポートの結果から分かる
            loaded_dict["l"] = loaded_dict["l"]/loaded_dict["n"]#基本リンク数が1個のときでも分割できるの？ECが一つも無い場合は分割した後にQRを挿入できないから無理じゃね？
#卒論:methodBにおいてはパラメータ(分配率の成功or失敗の乱数)をふって各状態を比較(各機能がエンタングルメントが成功しているかどうかを視覚的に見えるようにする)
#methodA:N=1,n=3で固定して、乱数ふって値を出してプロット
        else:
            raise NotImplementedError

        return loaded_dict


    def calculate_edr(param_dict, arcmethod):
        """
        Calculates the entanglement distribution rate given an input parameter dictionary. It is expected that the parameter dictionary contains
        among system parameters, the variables l_half (half of the length of an elementary link) and the n number of elementary links per
        ARC.

        Entanglement distribution rate is calculated as follows:
        - The elements involved in the connection of two QRs are: the efficiency eta_qr between the ARC chain and the QR itself, the probability
        p_arc_gen of generating entanglement in each individual elementary link, and the efficiency of the elements connecting each individual
        elementary link eta_el_connections.
        - Once the efficiencies are obtained, the conversion to an entanglement distribution rate is handled by the function calc_timing_thresholds.
        This function takes into account operation times, transmission times etc.
        #効果が得られたら、エンタングルメント分配率への変換はcalc_timing_thresholdsによって扱われる
        """

        def calc_p_arc_gen(param_dict, required_single_photon_measurements=2):
            """
            Calculates the probability for entanglement to be generated within a single elementary link, given system parameters

            The optional parameter required_single_photon_measurements is defaulted to 2, meaning we only need 2 single photon detections to
            successfully generate entanglement. In the case of a DLCZ protocol, we can change this to 1.
            """
            eta_bsm = param_dict["eta_BSM"]
            alpha = param_dict["alpha"]
            l = param_dict["l"]
            eta_det = param_dict["eta_DET"]
            gammaf = param_dict["Gamma_f"]
            eta_afc = param_dict["eta_AFC"]
            eta_shift = param_dict["eta_shift"]
            eta_buff = param_dict["eta_BUFF"]
            n = param_dict["n"]

            p_single_link = (1-(1-np.exp(-alpha*l)*(eta_bsm**(required_single_photon_measurements-1))*(eta_det)**required_single_photon_measurements)**gammaf) * (eta_afc)**2
            # p_single_link = (1-(1-np.exp(-alpha*l)*eta_bsm*(eta_det)**required_single_photon_measurements)**gammaf) * (eta_afc * eta_shift)**2

            # print(f'DEBUG SECTION -=-=-=-=-==--=-=-=-=-=-=-=-=-=-=-=-=-=-==--=-=-=-=-=-=-=-=-=-=-=-=-=-==--=-=-=-=-=-=-=-=-=-=-=-=-=-==--=-=-=-=-=-=-=-=-=')
            # print(f'We perform the calculation for gammaf = 20 {(1-np.exp(-alpha*l)*eta_bsm*(eta_det)**required_single_photon_measurements)**20}')
            # print(f'We perform the calculation for gammaf = 50 {(1-np.exp(-alpha*l)*eta_bsm*(eta_det)**required_single_photon_measurements)**50}')
            # print(f'We perform the calculation for gammaf = 100 {(1-np.exp(-alpha*l)*eta_bsm*(eta_det)**required_single_photon_measurements)**100}')
            # print(f'DEBUG SECTION -=-=-=-=-==--=-=-=-=-=-=-=-=-=-=-=-=-=-==--=-=-=-=-=-=-=-=-=-=-=-=-=-==--=-=-=-=-=-=-=-=-=-=-=-=-=-==--=-=-=-=-=-=-=-=-=')

            # p_single_link = ((1-(1-eta_bsm*(np.exp(-alpha*l)) * (eta_det)**2)**gammaf)**n )*((eta_afc)**2 *(eta_shift)**2)**n

            # print(f'Debug single link {p_single_link}')
            # total_p_arc_gen = p_single_link * (eta_bsm)**(n-1) * (eta_qr)**2 * (eta_afc)**(2*(n-1))
            # print(f'Debug total p arc gen {total_p_arc_gen}')
            # return total_p_arc_gen
            return p_single_link

        def calc_eta_qr(param_dict, sync_method = "A", required_single_photon_measurements=2):#sync_method<- 同期方式
            """
            Calculates the efficiency of the router, eta_qr, depending on which scheme we are looking at. The scheme can either be "a" or "b".

            Scheme "NV" matches the scheme presented in the paper of Askarani
            Scheme "Ca40" is the ion trap adapted version we are looking at.

            Method "a" uses multiple QMs at either ends of an ARC chain to ensure the message reaches both routers at the same time.
            Method "b" achieves the same effect by halving the distances between nodes.
            """

            # print(f'lets check: {param_dict}')

            if architecture == "askarani":
                
                eta_shift = param_dict["eta_shift"]
                eta_qfc637 = param_dict["eta_QFC_637"]
                eta_pol = param_dict["eta_pol"]
                eta_map = param_dict["eta_MAP"]
                eta_13c = param_dict["eta_13C"]
                eta_buff = param_dict["eta_BUFF"]

                eta_afc = param_dict["eta_AFC"]
                n = param_dict["n"]

                if sync_method == "A":
                    eta_qr = eta_shift * eta_buff * eta_qfc637 * eta_pol * eta_map * eta_13c * (eta_afc **(n-1))
                    # eta_qr = eta_buff * eta_qfc637 * eta_pol * eta_map * eta_13c * (eta_afc **(n-1))
                elif sync_method == "B":
                    eta_qr = eta_shift * eta_buff * eta_qfc637 * eta_pol * eta_map * eta_13c
                    # eta_qr = eta_buff * eta_qfc637 * eta_pol * eta_map * eta_13c
                else:
                    raise ValueError("You have not picked one of the two Q network methods")
                
            elif architecture == "ca":
                
                eta_loss = param_dict["eta_loss"]
                eta_afc = param_dict["eta_AFC"]
                eta_shift = param_dict["eta_shift"]
                eta_aom = param_dict["eta_aom"]
                eta_bsm = param_dict["eta_BSM"]
                eta_det = param_dict["eta_DET"]
                eta_interface = param_dict["eta_interface"]
                #eta_interface = eta_aom * eta_bsm * (eta_det) ** required_single_photon_measurements

                n = param_dict["n"]

                if sync_method == "A":
                    eta_qr = eta_loss * (eta_afc **(n-1)) * eta_shift * eta_aom * eta_interface#レポートのQST式と同じじゃね？
                elif sync_method == "B":
                    eta_qr = eta_loss * eta_shift * eta_aom * eta_interface
                else:
                    raise ValueError("You have not picked one of the two Q network methods")          

            else:
                raise NotImplementedError

            return eta_qr

        def calc_el_connections(param_dict, required_single_photon_measurements=2):
            """
            Each elementary link produces one successful frequency mode. To mediate between these successful modes between different ELs, there is an
            FFSMM operation that connects each EL, which also has an attributed efficiency.

            Again, the optional parameter required_single_photon_measurements describes the number of single photon measurements required to successfully
            distribute entanglement. This can be 1 in the case of a DLCZ type scheme.
            """
            eta_bsm = param_dict["eta_BSM"]
            eta_det = param_dict["eta_DET"]
            eta_shift = param_dict["eta_shift"]

            if architecture == "askarani":
                eta_connection = eta_bsm
            elif architecture == "ca":
                eta_connection = ((eta_bsm)**(required_single_photon_measurements-1)) * eta_shift * (eta_det) ** required_single_photon_measurements
            else:
                raise NotImplementedError

            return eta_connection
        
        def calc_p_arc_gen_n(param_dict, qr_eff, conn_eff, el_gen_prob):
            """
            Constructs the probability of generating entanglement between two QRs separated by n elementary links.
            """
            n = param_dict["n"]

            return (qr_eff)**2* (el_gen_prob)** n * (conn_eff)**(n-1)
        def calc_p_arc_gen_1(qr_eff, el_gen_prob):
            return (qr_eff)**2*el_gen_prob

        def calc_timings(param_dict, prob_arcgen_n):
            """
            Function calculates the timings and probabilities between two QRs connected by N ARC-Rs. 
            
            Outputs of this function are tau_arc and t_trans:
            - t_trans, which is the time taken for the ARC, QR and CNOT operations.
            - t_arc, which is the minimum between: the time it takes for entanglement to have been generated between two QRs connected by M
            ARCs with a probability given by 100% * (1-epsilon), and the maximum storage time of the QR itself.
            """

            big_N = param_dict["big_N"]
            etaepps = param_dict["eta_EPPS"]
            repps = param_dict["R_EPPS"]
            omega_epps = etaepps*repps
            tCNOT = param_dict["t_CNOT"]
            tAFC = param_dict["t_AFC"]
            eps = param_dict["eps"]
            n = param_dict["n"]

            tQR = param_dict["t_QR"]
            tARC = tAFC * (n-1)

            ttrans = tQR + tARC + tCNOT
            # ttrans = tARC + tCNOT + param_dict["t_13C"]

            tau = (1/omega_epps * np.log(1-(1-eps)**(1/big_N))/np.log(1-prob_arcgen_n)) + ttrans
            
            if architecture == "askarani":
                tau_arc = np.minimum(param_dict["t_NV"], tau)

            elif architecture == "ca":
                tau_arc = np.minimum(param_dict["t_Ca43"], tau)

            else:
                raise NotImplementedError

            return tau_arc, ttrans
        def calc_timings_11(param_dict):
            etaepps = param_dict["eta_EPPS"]
            repps = param_dict["R_EPPS"]
            omega_epps = etaepps*repps
            tCNOT = param_dict["t_CNOT"]
            tAFC = param_dict["t_AFC"]
            eps = param_dict["eps"]
            tQR = param_dict["t_QR"]

            ttrans = tQR + tCNOT
            # ttrans = tARC + tCNOT + param_dict["t_13C"]
            prob_arcgen_1=calc_p_arc_gen_1(eta_QR,p_arc_gen)
            tau = (1/omega_epps * np.log(1-(1-eps))/np.log(1-prob_arcgen_1)) + ttrans
            
            if architecture == "askarani":
                tau_arc = np.minimum(param_dict["t_NV"], tau)

            elif architecture == "ca":
                tau_arc = np.minimum(param_dict["t_Ca43"], tau)

            else:
                raise NotImplementedError

            return tau_arc, ttrans

        # def calc_timing_thresholds(param_dict):
        #     """
        #     Function calculates two quantities given a dictionary of parameters:
        #     - t_trans, which is the time taken for the ARC, QR and CNOT operations.
        #     - t_arc, which is the minimum between: the time it takes for entanglement to have been generated between two QRs connected by M
        #     ARCs with a probability given by 100% * (1-epsilon), and the maximum storage time of the QR itself.
        #     """
        #     eps = param_dict["eps"]
        #     parcgen = param_dict["p_arc_gen"]
        #     etaepps = param_dict["eta_EPPS"]
        #     repps = param_dict["R_EPPS"]
        #     n = param_dict["n"]
            
        #     t13C = param_dict["t_13C"]
        #     tafc = param_dict["t_AFC"]

        #     # Two adjustable parameters, no function at the moment
        #     lenARCR = 1
        #     numARC = 1

        #     tau = np.log10((1-(1-eps)**(1/(lenARCR*numARC)))) / (np.log10(1-parcgen)) * 1/(etaepps * repps)
        #     print(f'tau is {tau}')


        #     x = np.log10(1-parcgen)
        #     y = np.log10((1-(1-eps)**(1/(lenARCR*numARC))))
        #     z = (1/(etaepps*repps))
        #     print(f'>>>>>>>>>>>>>>>> DEBUG 1 x is calculated as {x}')
        #     print(f'this shouldnt be zero {x[3]}')
        #     print(f'>>>>>>>>>>>>>>>> DEBUG 2 y is calculated as {y}')
        #     print(f'>>>>>>>>>>>>>>>> DEBUG 3 y second step is calculated as {(y/x)}')
        #     print(f'>>>>>>>>>>>>>>>> DEBUG 4 z is calculated as {z}')
        #     print(f'FINAL DEBUG Then tau should be calculated as: {z*y}')

        #     # x = log10(1.-vpa(Pgen))
        #     # y = np.log10((1-(1-eps)**(1/(lenARCR*numARC))))
        #     # y = (y./x);
        #     # z = (1/(EPPSEfficiency*EPPSRate));
        #     # tau = z*y;



        #     transsol = t13C + tafc * (n-1)

        #     # print(f'temp {temp}')
        #     # print(f'Debug {tau} and {transsol} and parcgen {parcgen}')
        #     print(f'Debug comparison units {tau + transsol} and t nv {param_dict["t_NV"]}')
        #     tau_arc = np.minimum(param_dict["t_NV"], tau + transsol)
        #     print(f'After comparison is {tau_arc}')
        #     return tau_arc, transsol
        
        ################################################################### Codebase

        # print(f'New Debug ==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=---=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
        # print(f'Before calculating anything, we have method {arcmethod} and architecture {architecture}')
        # print(f'Our N is {param_dict["big_N"]}, our n is {param_dict["n"]}, our l is {param_dict["l"]}')
        # print(f'New Debug ==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=---=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')

        eta_QR = calc_eta_qr(param_dict, sync_method=arcmethod)
        p_arc_gen = calc_p_arc_gen(param_dict)
        eta_conn = calc_el_connections(param_dict)

        ################### MAking some plots
        # eta_QR = np.arange(0.1,1.1,0.1)
        # p_arc_gen = np.arange(0.1,1.1,0.1)
        # eta_conn = np.arange(0.1,1.1,0.1)
        # eta_QR = 0.9
        # p_arc_gen = 0.9
        # eta_conn = 0.9
        ################### MAking some plots
        big_N = param_dict["big_N"]
        n=param_dict["n"]
        bigN_is_array=isinstance(big_N,(list,tuple,np.ndarray))
        n_is_array=isinstance(n,(list,tuple,np.ndarray))
        etaepps = param_dict["eta_EPPS"]
        repps = param_dict["R_EPPS"]
        omega_epps = etaepps*repps
        qr_entanglement_distribution_prob = calc_p_arc_gen_n(param_dict, eta_QR, eta_conn, p_arc_gen)
        tauARC, ttrans = calc_timings(param_dict, qr_entanglement_distribution_prob)
        print(f'Checks for {architecture}: eta_QR {eta_QR}, p_arc_gen {p_arc_gen}, eta_conn {eta_conn}')
        
        #どのケースを可視化するかを指定
        #"case1":n=1,N=1,...,10
        #"case2":N=1,n=1,...,10
        #"both":両方
        # print(f'Debug marker -----------------------------------------> We are using method {arcmethod} for architecture {architecture}')
        # p_arc_gen = 1
        # eta_conn = 1
        # # eta_QR = np.divide(eta_QR,eta_QR)
        # eta_QR = np.array([1,0.9,0.8,0.7,0.6])
        # param_dict["t_AFC"] = 1e-8
        # print(f'After modifying for {architecture}: eta_QR {eta_QR}, p_arc_gen {p_arc_gen}, eta_conn {eta_conn}')

        edr = 1/tauARC * (1-(1-qr_entanglement_distribution_prob)**(omega_epps*(tauARC - ttrans)))**big_N
        #edr_11=edr[0]
        #qr_entanglement_distribution_prob_11=calc_p_arc_gen_1(eta_QR,p_arc_gen)
        #tauARC_11,ttrans_11=calc_timings_11(param_dict)
        #n_attemps=round(np.log(1-edr_11*tauARC_11)/np.log(1-qr_entanglement_distribution_prob_11))
        #n_attemps=round(np.log(0.05)/np.log(1-qr_entanglement_distribution_prob_11))
        if bigN_is_array and not n_is_array:#N=[1,2,3,...],n=1
            print(f'tau(N=1,n=1)theory={tauARC[0]}')
            #n_attemps=omega_epps*(tauARC[0]-ttrans)
        elif n_is_array and not bigN_is_array:#n=[1,2,3,...],N=1
            print(f'tau(N=1,n=1)theory={tauARC[0]}')
            #n_attemps=omega_epps*(tauARC[0]-ttrans[0])
        if arcmethod == "B":
            N_val = int(np.atleast_1d(param_dict["big_N"])[0])
            n_val = int(np.atleast_1d(param_dict["n"])[0])
            if N_val == 1 and n_val == 2:
        # t_gen = t_QST + t_ARC + t_CNOT（あなたのコードでは ttrans がそれ
              t_gen = float(ttrans)
              # tau の上限（メモリ時間の上限でクリップするなら）
              if architecture == "askarani":
                  tau_max = float(param_dict["t_NV"])
              elif architecture == "ca":
                  tau_max = float(param_dict["t_Ca43"])
              else:
                  tau_max = None
              tau_star, info = find_tau_for_target_prob(
                  target_p=0.95,
                  omega_epps=float(omega_epps),
                  t_gen=t_gen,
                  p_EL=float(p_arc_gen),
                  p_EC=float(eta_conn),
                  p_QST=float(eta_QR),
                  num_big_trials=20000,  # まずは5000〜、精度欲しければ増やす
                  seed=0,
                  tau_max=tau_max,
                  tol_tau=1e-4,
                  max_iter=40,
                  )

              if tau_star is None:
                  p_hi, se_hi, M_hi = info
                  print("=== MC result ===")
                  print("tau_max でも 0.95 に到達しませんでした")
                  print(f"tau_max={tau_max}, p={p_hi:.6f}, se={se_hi:.6f}, M={M_hi}")
              else:
                  p_fin, se_fin, M_fin = info
                  print("=== MC tau for p=0.95 (MethodB, N=1,n=2) ===")
                  print(f"tau* = {tau_star:.6e} [s]")
                  print(f"p(tau*) = {p_fin:.6f} (se={se_fin:.6f}), M={M_fin}")    
        """"
        for l,tauART in enumerate(tau_ART):
            n_attemps_TKG=int(omega_epps*tauART)
            successes=0
            for j in range(50):
                for i in range(n_attemps_TKG):#1s=100μs(t_AFC)*10000,論文では取りあえず10000回
                    tl,ARC_success = trial_methodB_timeline(p_EL=p_arc_gen, p_EC=eta_conn, p_QST=eta_QR, rng=rng)
                    if ARC_success==True:
                        successes +=1
                        break
            p[l]=successes/50
            stderror[l]=np.sqrt(p[l]*(1.0-p[l])/50)
        """       
        #plt.figure()
        #plt.errorbar(x, y, marker="o", capsize=3)
        #plt.axhline(0.95, linestyle="--")   # ★ y=0.95 に点線（破線）
        #plt.xlabel("tauART")
        #plt.ylabel("p (success probability)")
        #plt.ylim(0, 1.05)
        #plt.grid(True, alpha=0.3)
        #plt.show()     
        #print(f"Monte Carlo p={p:.6f}")
        #print(f"std_error={stderror:.6f}")

        # eta_QR = calc_eta_qr(param_dict)
        # param_dict["eta_QR"] = eta_QR
        # p_arc_gen = calc_p_arc_gen(param_dict)
        # param_dict["p_arc_gen"] = p_arc_gen
        # tau_arc, transsol = calc_timing_thresholds(param_dict)

        # # edr = (1-(1-p_arc_gen**(param_dict["eta_EPPS"] * param_dict["R_EPPS"] * (tau_arc - transsol))))/tau_arc
        # edr = (1-(1-p_arc_gen)**((tau_arc-transsol)*(param_dict["eta_EPPS"] * param_dict["R_EPPS"])))/tau_arc
        return edr
    
    def option_select(options, input_message):
        """
        Generic QOL function that allows user input to choose what code to run
        """

        user_input = ''

        # input_message = "Pick an option:\n"

        for index, item in enumerate(options):
            input_message += f'{index+1}) {item}\n'

        input_message += 'Your choice: '

        while user_input not in map(str, range(1, len(options) + 1)):
            user_input = input(input_message)

        return options[int(user_input) - 1], int(user_input)
    
    def plotter(xdata, ydata, xlabel, dlabels=None):
        """
        Generic function that handles plotting sequence.
        
        Input ydata can be a list, in which case the function will plot multiple curves.

        Labels input should be 1 entry per set of ydata. It should always be a list.
        """

        if ydata.ndim == 1:
            plt.plot(xdata, ydata,'x', label = dlabels[0])
        else:
            for dataset in range(ydata.shape[0]):
                plt.plot(xdata, ydata[dataset], 'x', label = dlabels[dataset])

        plt.yscale("log")
        plt.grid()
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel("Entanglement Distribution Rate")
        plt.show()

    ############################################### Code base Begins here ###############################################

    # Choose which methods we would like to view
    options_list = ["askarani", "ca", "both"]
    architecture, arch_index = option_select(options_list, "Please pick an architecture:\n")
    if architecture == "both":
        options_list = ["A A", "A B", "B A", "B B"]
    else:
        options_list = ["A", "A B", "B"]
    method_choice, method_index = option_select(options_list, "Please pick a method:\n")
    options_list = [1, 2]#choise Askarani:nearterm or longterm,Ca:lquom,nearterm
    param_choice, param_index = option_select(options_list, "Please pick a set of parameters:\n")

    print(f'You have selected the architecture "{architecture}" with method "{method_choice}" and parameter set given by index {param_choice}') 


    # Run the code depending on user inputs
    if architecture == "both":

        ############################################# First for askarani
        if method_index == 1 or method_index == 2:
            method = "A"
        elif method_index == 3 or method_index == 4:
            method = "B"
        else:
            raise NotImplementedError
        
        architecture = "askarani"
        param_dict = load_params_dict(arch=architecture, arc_method=method, param_set = param_choice)

        distribution_rates_askarani = calculate_edr(param_dict, arcmethod=method)
        print(f'distribution rates askarani: {distribution_rates_askarani}')
        print(f'>>>>>>>> Lets have a look at what is being varied. N: {param_dict["big_N"]} n: {param_dict["n"]} l: {param_dict["l"]}')
        # print(f'Distances being varied are {param_dict["big_N"] * param_dict["n"] * param_dict["l"]}')

        ############################################# Next for ca trap
        if method_index == 1 or method_index == 3:
            method = "A"
        elif method_index == 2 or method_index == 4:
            method = "B"
        else:
            raise NotImplementedError

        architecture = "ca"
        param_dict = load_params_dict(arch=architecture, arc_method=method, param_set = param_choice)

        distribution_rates_ca = calculate_edr(param_dict, arcmethod=method)
        print(f'distribution rates ca: {distribution_rates_ca}')
        print(f'Lets have a look at what is being varied. N: {param_dict["big_N"]} n: {param_dict["n"]} l: {param_dict["l"]}')
        print(f'Distances being varied are {param_dict["big_N"] * param_dict["n"] * param_dict["l"]}')

        ############################################# Plot results
        plotter(param_dict["big_N"] * param_dict["n"] * param_dict["l"], np.array([distribution_rates_askarani, distribution_rates_ca]), xlabel="ARC-R distance in km", dlabels=["Askarani","Ca"])


    else:
        if method_index == 1:
            method_1 = "A"
            dlab = ["Method A"]
        elif method_index == 2:
            method_1 = "A"
            method_2 = "B"
            dlab = ["Method A","Method B"]
        elif method_index == 3:
            method_1 = "B"
            dlab = ["Method B"]
        else:
            raise NotImplementedError
        
        param_dict_1 = load_params_dict(arch=architecture, arc_method=method_1, param_set = param_choice)
        distribution_rates_1 = calculate_edr(param_dict_1,arcmethod=method_1)
        print(f'Distribution rates for the first method: {distribution_rates_1}')

        if method_index == 2:
            param_dict_2 = load_params_dict(arch=architecture, arc_method=method_2, param_set = param_choice)
            distribution_rates_2 = calculate_edr(param_dict_2,arcmethod=method_2)
            print(f'Distribution rates for the second method: {distribution_rates_2}')
            ydat = np.array([distribution_rates_1, distribution_rates_2])

        else:
            ydat = distribution_rates_1

        plotter(param_dict_1["big_N"]*param_dict_1["n"]*param_dict_1["l"], ydat, xlabel="ARC-R distance in km", dlabels=dlab)


if __name__ == '__main__':
    main()