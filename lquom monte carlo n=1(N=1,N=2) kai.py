import numpy as np
import pandas
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
#N=1,n=1のとき
def trial_timeline11(p_EL, p_QST, t_AFC, t_QST, rng=None):
    """
    N=1, n=1 専用
    1回の試行で
      - ELヘラルド待ちまで t_AFC
      - EL成功なら QST を行い追加で t_QST
    を消費する

    Returns
    -------
    tl : dict
        タイムライン情報
    ARC_success : bool
        その試行で ARC が成功したか
    dt : float
        その試行で消費した時間 [s]
    """
    if rng is None:
        rng = np.random.default_rng()

    tl = {"t0": {}, "t_half": {}, "t_full": {}}

    # t = 0
    tl["t0"]["EPPS_fired"] = True

    # まず EL 試行
    EL_success = (rng.random() < p_EL)
    tl["t_half"]["EL_success"] = EL_success

    # 少なくとも t_AFC は使う
    dt = t_AFC

    # t = t_AFC でヘラルド到着
    tl["t_full"]["QST1_msg_from_EL"] = EL_success
    tl["t_full"]["QST2_msg_from_EL"] = EL_success
    tl["t_full"]["QST_judged"] = EL_success

    if EL_success:
        # 成功したときだけ QST 時間を追加
        dt += t_QST

        QST1_success = (rng.random() < p_QST)
        QST2_success = (rng.random() < p_QST)
    else:
        QST1_success = False
        QST2_success = False

    tl["t_full"]["QST1_success"] = QST1_success
    tl["t_full"]["QST2_success"] = QST2_success

    ARC_success = EL_success and QST1_success and QST2_success
    tl["t_full"]["ARC_success"] = ARC_success

    return tl, ARC_success, dt
def simulate_one_second_N1_n1(p_EL, p_QST, t_AFC, t_QST, rng=None, T=1.0):
    """
    N=1, n=1 の系で、経過時間が T 秒を超えるまで試行を繰り返す。
    成功した ARC の回数を返す。
    """
    if rng is None:
        rng = np.random.default_rng()

    t_elapsed = 0.0
    success_count = 0
    n_trials = 0

    while True:
        tl, ARC_success, dt = trial_timeline11(
            p_EL=p_EL,
            p_QST=p_QST,
            t_AFC=t_AFC,
            t_QST=t_QST,
            rng=rng
        )

        # この試行を入れると T 秒を超えるなら終了
        if t_elapsed + dt > T:
            break

        t_elapsed += dt
        n_trials += 1

        if ARC_success:
            success_count += 1

    return success_count, n_trials, t_elapsed
def monte_carlo_edr_N1_n1(p_EL, p_QST, t_AFC, t_QST, n_mc=5000, seed=0):
    rng = np.random.default_rng(seed)

    success_list = np.zeros(n_mc, dtype=float)
    trial_list = np.zeros(n_mc, dtype=float)
    elapsed_list = np.zeros(n_mc, dtype=float)

    for k in tqdm(range(n_mc)):
        success_count, n_trials, t_elapsed = simulate_one_second_N1_n1(
            p_EL=p_EL,
            p_QST=p_QST,
            t_AFC=t_AFC,
            t_QST=t_QST,
            rng=rng,
            T=1.0
        )
        success_list[k] = success_count
        trial_list[k] = n_trials
        elapsed_list[k] = t_elapsed

    edr_mean = np.mean(success_list)              # 1秒あたり成功数の平均
    edr_std = np.std(success_list, ddof=1)
    edr_se = edr_std / np.sqrt(n_mc)

    return {
        "edr_mean": edr_mean,
        "edr_std": edr_std,
        "edr_se": edr_se,
        "success_list": success_list,
        "trial_list": trial_list,
        "elapsed_list": elapsed_list,
    }

import numpy as np

def trial_arc_n1_single(p_EL, p_QST, t_AFC, t_QST, rng=None):
    """
    N=1, n=1 の 1 ARC を 1 回試行する。
    Returns
    -------
    arc_success : bool
        このARC試行が成功したか
    dt : float
        この試行に要した時間 [s]
    success_time_offset : float or None
        試行開始から成功が確定した時刻オフセット [s]
        成功しなければ None
    """
    if rng is None:
        rng = np.random.default_rng()

    # EL 成功判定
    EL_success = (rng.random() < p_EL)

    # EL失敗なら t_AFC で終了
    if not EL_success:
        return False, t_AFC, None

    # EL成功なら QST を行う
    QST1_success = (rng.random() < p_QST)
    QST2_success = (rng.random() < p_QST)

    arc_success = QST1_success and QST2_success
    dt = t_AFC + t_QST

    if arc_success:
        return True, dt, dt
    else:
        return False, dt, None
    
def simulate_one_second_N2_n1(
    p_EL,
    p_QST,
    t_AFC,
    t_QST,
    t_hold,
    rng=None,
    T=1.0
):
    """
    N=2, n=1:
    ARC1, ARC2 の2本を並列運転する。
    片方だけ成功したら、その成功状態を t_hold 秒だけ保持し、
    その間にもう片方の成功を待つ。
    両方そろったら ARC_success を1回加算し、両方リセット。
    """
    if rng is None:
        rng = np.random.default_rng()

    t_elapsed = 0.0
    pair_success_count = 0

    # ARCごとの保持状態
    arc1_ready = False
    arc2_ready = False
    arc1_expire = -np.inf
    arc2_expire = -np.inf

    n_global_steps = 0

    while True:
        # まず期限切れを処理
        if arc1_ready and t_elapsed >= arc1_expire:
            arc1_ready = False
            arc1_expire = -np.inf

        if arc2_ready and t_elapsed >= arc2_expire:
            arc2_ready = False
            arc2_expire = -np.inf

        # もし両方 ready なら1回成功として数えてリセット
        if arc1_ready and arc2_ready:
            pair_success_count += 1
            arc1_ready = False
            arc2_ready = False
            arc1_expire = -np.inf
            arc2_expire = -np.inf
            continue

        # 未成功の ARC だけ試行する
        do_arc1 = not arc1_ready
        do_arc2 = not arc2_ready

        results = []

        t_start = t_elapsed

        if do_arc1:
            arc1_success, dt1, succ_offset1 = trial_arc_n1_single(
                p_EL=p_EL, p_QST=p_QST, t_AFC=t_AFC, t_QST=t_QST, rng=rng
            )
            results.append(("arc1", arc1_success, dt1, succ_offset1))
        else:
            dt1 = 0.0

        if do_arc2:
            arc2_success, dt2, succ_offset2 = trial_arc_n1_single(
                p_EL=p_EL, p_QST=p_QST, t_AFC=t_AFC, t_QST=t_QST, rng=rng
            )
            results.append(("arc2", arc2_success, dt2, succ_offset2))
        else:
            dt2 = 0.0

        # このステップで進む壁時計時間
        dt_step = max(dt1, dt2)

        # 1秒を超えるなら、このステップは入れず終了
        if t_elapsed + dt_step > T:
            break

        # 各ARCの成功を反映
        for name, success, dt_arc, succ_offset in results:
            if success:
                success_time = t_start + succ_offset
                expire_time = success_time + t_hold

                if name == "arc1":
                    arc1_ready = True
                    arc1_expire = expire_time
                elif name == "arc2":
                    arc2_ready = True
                    arc2_expire = expire_time

        # 壁時計を進める
        t_elapsed += dt_step
        n_global_steps += 1

        # 時間を進めた結果、期限切れしていれば落とす
        if arc1_ready and t_elapsed >= arc1_expire:
            arc1_ready = False
            arc1_expire = -np.inf

        if arc2_ready and t_elapsed >= arc2_expire:
            arc2_ready = False
            arc2_expire = -np.inf

        # この時点で両方そろっていれば1回成功
        if arc1_ready and arc2_ready:
            pair_success_count += 1
            arc1_ready = False
            arc2_ready = False
            arc1_expire = -np.inf
            arc2_expire = -np.inf

    return pair_success_count, n_global_steps, t_elapsed
from tqdm import tqdm

def monte_carlo_edr_N2_n1(
    p_EL,
    p_QST,
    t_AFC,
    t_QST,
    t_hold,
    n_mc=5000,
    seed=0
):
    rng = np.random.default_rng(seed)

    success_list = np.zeros(n_mc, dtype=float)
    step_list = np.zeros(n_mc, dtype=float)
    elapsed_list = np.zeros(n_mc, dtype=float)

    for k in tqdm(range(n_mc)):
        pair_success_count, n_steps, t_elapsed = simulate_one_second_N2_n1(
            p_EL=p_EL,
            p_QST=p_QST,
            t_AFC=t_AFC,
            t_QST=t_QST,
            t_hold=t_hold,
            rng=rng,
            T=1.0
        )

        success_list[k] = pair_success_count
        step_list[k] = n_steps
        elapsed_list[k] = t_elapsed

    edr_mean = np.mean(success_list)
    edr_std = np.std(success_list, ddof=1)
    edr_se = edr_std / np.sqrt(n_mc)

    return {
        "edr_mean": edr_mean,
        "edr_std": edr_std,
        "edr_se": edr_se,
        "success_list": success_list,
        "step_list": step_list,
        "elapsed_list": elapsed_list,
    }
"""
def trial_timeline1(p_EL,p_EC, p_QST, rng=None):
    
    if rng is None:
        rng = np.random.default_rng()

    tl = {"t0": {}, "t_half": {}, "t_full": {}}

    # t = 0 : EPPS 発射（ニュートラル）
    tl["t0"]["EPPS_fired"] = True

    # t = t_AFC/2 : EL 内 BSM で成功/失敗決定（乱数 p_EL）
    EL_success = (rng.random() < p_EL)
    tl["t_half"]["EL_success"] = EL_success

    # t = t_AFC : 反対側の QST にもヘラルドが届く
    QST1_msg_from_EL = EL_success  # EL → QST1
    QST2_msg_from_EL = EL_success  # EL → QST2
    tl["t_full"]["QST1_msg_from_EL"] = QST1_msg_from_EL
    tl["t_full"]["QST2_msg_from_EL"] = QST2_msg_from_EL

    # 便利のため色々コピー
    tl["t_full"]["EL1_success"] = EL_success
    tl["t_full"]["EL2_success"] = EL_success
    tl["t_full"]["QST1_msg_from_EL1"] = QST1_msg_from_EL
    tl["t_full"]["QST2_msg_from_EL2"] = QST2_msg_from_EL

    # QST 判定条件：
    #   両 EL のヘラルド成功 & EC 成功 のときだけ判定
    pre_QST = EL_success 
    tl["t_full"]["QST_judged"] = pre_QST
    # QST duration time t_QST
    if pre_QST==True:
        QST1_success = (rng.random() < p_QST)
        QST2_success = (rng.random() < p_QST)
    else:
        QST1_success = False
        QST2_success = False

    tl["t_full"]["QST1_success"] = QST1_success
    tl["t_full"]["QST2_success"] = QST2_success

    # ARC 全体の成功：QST1, QST2 両方成功,合計時間t_AFC+t_QST
    ARC_success = pre_QST and QST1_success and QST2_success
    tl["t_full"]["ARC_success"] = ARC_success

    return tl,ARC_success

#N=2,n=1のとき
def trial_timeline2(p_EL,p_EC, p_QST, rng=None):
    
    if rng is None:
        rng = np.random.default_rng()

    tl = {"t0": {}, "t_half": {}, "t_full": {}}

    # t = 0 : EPPS 発射（ニュートラル）
    tl["t0"]["EPPS_fired"] = True

    # t = t_AFC/4 : EL 内 BSM で成功/失敗決定（乱数 p_EL）
    EL1_success = (rng.random() < p_EL)
    EL2_success = (rng.random() < p_EL)

    tl["t_half"]["EL1_success"] = EL1_success
    tl["t_half"]["EL2_success"] = EL2_success

    # t = t_AFC : 反対側の QST にもヘラルドが届く
    QST1_1_msg_from_EL = EL1_success  # EL → QST1_1
    QST1_2_msg_from_EL = EL1_success  # EL → QST1_2

    QST2_1_msg_from_EL = EL2_success
    QST2_2_msg_from_EL = EL2_success
    tl["t_full"]["QST1_1_msg_from_EL"] = QST1_1_msg_from_EL
    tl["t_full"]["QST1_2_msg_from_EL"] = QST1_2_msg_from_EL

    tl["t_full"]["QST2_1_msg_from_EL"] = QST2_1_msg_from_EL
    tl["t_full"]["QST2_2_msg_from_EL"] = QST2_2_msg_from_EL
    # 便利のため色々コピー
    tl["t_full"]["EL1_success"] = EL1_success
    tl["t_full"]["EL2_success"] = EL2_success
    tl["t_full"]["QST1_1_msg_from_EL"] = QST1_1_msg_from_EL
    tl["t_full"]["QST1_2_msg_from_EL"] = QST1_2_msg_from_EL
    tl["t_full"]["QST2_1_msg_from_EL"] = QST2_1_msg_from_EL
    tl["t_full"]["QST2_2_msg_from_EL"] = QST2_2_msg_from_EL

    # QST 判定条件：
    #   両 EL のヘラルド成功 & EC 成功 のときだけ判定
    pre1_QST = EL1_success
    pre2_QST = EL2_success 
    tl["t_full"]["QST_judged"] = pre1_QST
    tl["t_full"]["QST_judged"] = pre2_QST

    if pre1_QST==True:
        QST1_1_success = (rng.random() < p_QST)
        QST1_2_success = (rng.random() < p_QST)
    else:
        QST1_1_success = False
        QST1_2_success = False
    tl["t_full"]["QST1_1_success"] = QST1_1_success
    tl["t_full"]["QST1_2_success"] = QST1_2_success
    if pre2_QST==True:
        QST2_1_success = (rng.random() < p_QST)
        QST2_2_success = (rng.random() < p_QST)
    else:
        QST2_1_success = False
        QST2_2_success = False    
    tl["t_full"]["QST2_1_success"] = QST2_1_success
    tl["t_full"]["QST2_2_success"] = QST2_2_success

    # ARC 全体の成功：QST1, QST2 両方成功
    ARC_success = pre1_QST and pre2_QST and QST1_1_success and QST1_2_success and QST2_1_success and QST2_2_success
    tl["t_full"]["ARC_success"] = ARC_success

    return tl,ARC_success
"""
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
            df_data = pandas.read_excel(r'C:\Users\ara07\Desktop\LQUOM書類\Parameters_ca_trap.xlsx')

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
            #loaded_dict["n"] = np.arange(1,11,1) 
            loaded_dict["n"] = 1
            #loaded_dict["big_N"] = 1
            loaded_dict["big_N"] = np.arange(1,11,1)
            loaded_dict["eps"] = 0.05#entanngle/second(目標失敗率\epsilon)
            loaded_dict["eta_loss"] = 0.9                        # Generic fiber loss coefficient
            
        elif arc_method == "B":
            #loaded_dict["n"] = np.arange(1,11,1) 
            loaded_dict["n"] = 1
            #loaded_dict["big_N"] = 1
            loaded_dict["big_N"] = np.arange(1,11,1)
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
            print(f"EL={p_single_link:.4f}")
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
                    print(f"qst={eta_qr:4f}")
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
                print(f"EC={eta_connection:.4f}")
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
            #eps=0.5
            n = param_dict["n"]

            tQR = param_dict["t_QR"]
            tARC = tAFC * (n-1)

            ttrans = tQR + tARC + tCNOT
            # ttrans = tARC + tCNOT + param_dict["t_13C"]

            tau = (1/omega_epps * np.log(1-(1-eps)**(1/big_N))/np.log(1-prob_arcgen_n)) + ttrans
            print(f'tau={tau[0]}')
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
        etaepps = param_dict["eta_EPPS"]
        repps = param_dict["R_EPPS"]
        omega_epps = etaepps*repps
        qr_entanglement_distribution_prob = calc_p_arc_gen_n(param_dict, eta_QR, eta_conn, p_arc_gen)
        tauARC, ttrans = calc_timings(param_dict, qr_entanglement_distribution_prob)
        print(f'Checks for {architecture}: eta_QR {eta_QR}, p_arc_gen {p_arc_gen}, eta_conn {eta_conn}')
        print(f"arc={((eta_QR)**2)*((p_arc_gen)**2)*(eta_conn):.9f}")
        # print(f'Debug marker -----------------------------------------> We are using method {arcmethod} for architecture {architecture}')
        # p_arc_gen = 1
        # eta_conn = 1
        # # eta_QR = np.divide(eta_QR,eta_QR)
        # eta_QR = np.array([1,0.9,0.8,0.7,0.6])
        # param_dict["t_AFC"] = 1e-8
        # print(f'After modifying for {architecture}: eta_QR {eta_QR}, p_arc_gen {p_arc_gen}, eta_conn {eta_conn}')

        edr = 1/tauARC * (1-(1-qr_entanglement_distribution_prob)**(omega_epps*(tauARC - ttrans)))**big_N

         #====Monte Carlo probability====#
        """"
        p_EL=p_arc_gen
        p_EC=eta_conn
        p_QST=eta_QR
        tau_ART=np.array([2.0,4.0,6.0,8.0,10.0,12.0,14.0],dtype=float)
        n_art=3000
        rng = np.random.default_rng(0)
        p_list = np.zeros_like(tau_ART, dtype=float)
        err_list = np.zeros_like(tau_ART, dtype=float)
        for k,tauART in tqdm(enumerate(tau_ART)):
            n_attempts=int(round(omega_epps*tauART))#omega_epps*tauART-ttrans
            theta=0
            for i in range(n_art):#(8)式のプログラム
                successes=0
                for j in range(n_attempts):
                    _, ARC_success = trial_methodB_timeline(p_EL=p_EL,p_EC=p_EC,p_QST=p_QST,rng=rng)
                    if ARC_success==True:
                        successes += 1
                if successes>0:
                    theta += 1                            
            p=theta/(n_art)
            error=np.sqrt(p * (1.0 - p) / n_art)
            p_list[k]=p
            err_list[k]=error
        fig, ax = plt.subplots()
        ax.errorbar(tau_ART, p_list, yerr=err_list, fmt="o", capsize=4)
        ax.axhline(0.95, linestyle="--")   # p=0.95 の横点線
        ax.set_xlabel("tau_ART")
        ax.set_ylabel("p")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True)
        plt.show()
        target = 0.95
        idx = np.where((p_list[:-1] - target) * (p_list[1:] - target) <= 0)[0]
        if len(idx) > 0:
            i0 = idx[0]
            tau_star = np.interp(target, [p_list[i0], p_list[i0+1]], [tau_ART[i0], tau_ART[i0+1]])
            print(f"Estimated tau_ART where p≈0.95 (linear interp): {tau_star:.4f}")
        """
        #===============================#
        #====Monte Carlo EDR===#
        
        print("===Monte Carlo check for n=1,N=1===")
        rng=np.random.default_rng(0)
        
        successes=0
        t_AFC=param_dict["t_AFC"]
        tQST=10e-6
        mc_result=monte_carlo_edr_N1_n1(
            p_EL=p_arc_gen,
            p_QST=eta_QR,
            t_AFC=t_AFC,
            t_QST=tQST,
            n_mc=5000,
            seed=0
        )
        print(f"Monte Carlo EDR mean   = {mc_result['edr_mean']:.6f} [1/s]")
        print(f"Monte Carlo EDR std    = {mc_result['edr_std']:.6f}")
        print(f"Monte Carlo EDR stderr = {mc_result['edr_se']:.6f}")
        print(f"Mean trials in 1 sec   = {np.mean(mc_result['trial_list']):.2f}")
        print(f"Mean elapsed time      = {np.mean(mc_result['elapsed_list']):.6f} [s]")
        
        print("===Monte Carlo check for n=1,N=2===")
        t_AFC=param_dict["t_AFC"]
        t_QST=10e-6
        t_hold=1 #t^ca=1s
        mc_result = monte_carlo_edr_N2_n1(
            p_EL=p_arc_gen,
            p_QST=eta_QR,
            t_AFC=t_AFC,
            t_QST=t_QST,
            t_hold=t_hold,
            n_mc=5000,
            seed=0)
        print(f"Monte Carlo EDR mean   = {mc_result['edr_mean']:.6f} [1/s]")
        print(f"Monte Carlo EDR std    = {mc_result['edr_std']:.6f}")
        print(f"Monte Carlo EDR stderr = {mc_result['edr_se']:.6f}")
        print(f"Mean global steps      = {np.mean(mc_result['step_list']):.2f}")
        print(f"Mean elapsed time      = {np.mean(mc_result['elapsed_list']):.6f} [s]")
        
        expected_value_N1_n1=(p_arc_gen*(eta_QR)**2)/(t_AFC+p_arc_gen*t_QST)#分母は一回の試行辺りの平均時間:μ=t_AFC(1-P_EL)+(t_AFC+t_QST)p_EL=t_AFC+t_QST * p_EL
        print(f"expected_value(N=1,n=1)={expected_value_N1_n1:.6f}")
        #expected_value_N2_n1=expected_value_N1_n1*(1-np.exp(-expected_value_N1_n1*t_hold))
        def edr_expected_N2_n1_markov(p_EL, p_QST, t_AFC, t_QST, t_Ca):
            q = p_EL * (p_QST ** 2)
            mu = t_AFC + p_EL * t_QST
            lam = q / mu
            edr = lam * (2.0 * (1.0 - np.exp(-lam * t_Ca))) / (3.0 - 2.0 * np.exp(-lam * t_Ca))
            return edr, lam, q, mu
        edr_markov, lam, q, mu = edr_expected_N2_n1_markov(
            p_EL=p_arc_gen,
            p_QST=eta_QR,
            t_AFC=param_dict["t_AFC"],
            t_QST=10e-6,
            t_Ca=1.0
            )
        print(f"EDR_expected_markov = {edr_markov:.6f}")
        #===================================================================#
        #n_attemps=round(np.log(1-edr_11*tauARC_11)/np.log(1-qr_entanglement_distribution_prob_11))
        #n_attemps=round(np.log(0.05)/np.log(1-qr_entanglement_distribution_prob_11))

        # eta_QR = calc_eta_qr(param_dict)
        # param_dict["eta_QR"] = eta_QR
        # p_arc_gen = calc_p_arc_gen(param_dict)
        # param_dict["p_arc_gen"] = p_arc_gen
        # tau_arc, transsol = calc_timing_thresholds(param_dict)

        # # edr = (1-(1-p_arc_gen**(param_dict["eta_EPPS"] * param_dict["R_EPPS"] * (tau_arc - transsol))))/tau_arc
        # edr = (1-(1-p_arc_gen)**((tau_arc-transsol)*(param_dict["eta_EPPS"] * param_dict["R_EPPS"])))/tau_arc

        return edr,p_arc_gen,eta_conn,eta_QR
    
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
        distribution_rates_1,p_EL,p_EC,p_QST = calculate_edr(param_dict_1,arcmethod=method_1)
        print(f'Distribution rates for the first method: {distribution_rates_1}')

        if method_index == 2:
            param_dict_2 = load_params_dict(arch=architecture, arc_method=method_2, param_set = param_choice)
            distribution_rates_2,p_EL,p_EC,p_QST = calculate_edr(param_dict_2,arcmethod=method_2)
            print(f'Distribution rates for the second method: {distribution_rates_2}')
            ydat = np.array([distribution_rates_1, distribution_rates_2])

        else:
            ydat = distribution_rates_1

        plotter(param_dict_1["big_N"]*param_dict_1["n"]*param_dict_1["l"], ydat, xlabel="ARC-R distance in km", dlabels=dlab)        
        return p_EL,p_EC,p_QST
if __name__ == "__main__":
    #main()
    p_EL,p_EC,p_QST=main()
    """"
    ani = animate_flipbook(
        num_trials=10,  # 10試行ぶん流す
        #p_EL=p_EL,
        p_EL=0.6,
        #p_EC=p_EC,
        p_EC=0.8,
        #p_QST=p_QST,
        p_QST=0.5,
        interval=1000,  # 1.0秒ごとに次のコマへ
        )
    plt.show()
    """
    """"
    omega_epps=1*(10**4)
    tau_ART=np.array([2.0,4.0,6.0,8.0,10.0,12.0,14.0],dtype=float)
    n_art=3000
    rng = np.random.default_rng(0)
    p_list = np.zeros_like(tau_ART, dtype=float)
    err_list = np.zeros_like(tau_ART, dtype=float)
    for k,tauART in enumerate(tau_ART):
        n_attempts=int(round(omega_epps*tauART))
        successes=0
        for i in range(n_art):#(8)式のプログラム
            for j in range(n_attempts):
                _, ARC_success = trial_methodB_timeline(p_EL=p_EL,p_EC=p_EC,p_QST=p_QST,rng=rng)
                if ARC_success==True:
                    successes += 1
                    break
        p=successes/(n_art)
        error=np.sqrt(p * (1.0 - p) / n_art)
        p_list[k]=p
        err_list[k]=error
    fig, ax = plt.subplots()
    ax.errorbar(tau_ART, p_list, yerr=err_list, fmt="o", capsize=4)
    ax.axhline(0.95, linestyle="--")   # p=0.95 の横点線

    ax.set_xlabel("tau_ART")
    ax.set_ylabel("p")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True)
    plt.show()

    # （任意）0.95 を跨いでいたら線形補間で tau* を推定
    target = 0.95
    idx = np.where((p_list[:-1] - target) * (p_list[1:] - target) <= 0)[0]
    if len(idx) > 0:
        i0 = idx[0]
        tau_star = np.interp(target, [p_list[i0], p_list[i0+1]], [tau_ART[i0], tau_ART[i0+1]])
        print(f"Estimated tau_ART where p≈0.95 (linear interp): {tau_star:.4f}")    
        """