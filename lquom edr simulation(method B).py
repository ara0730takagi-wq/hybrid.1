import numpy as np
import pandas
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
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
#2. その時刻の色をまとめて決める関数
# =========================================
def compute_colors(tl, time_key):
    """
    time_key:
      "t0",
      "t_quarter",
      "t_half_preEC",
      "t_half_postEC",
      "t_full_preQST",
      "t_full_postQST"
    に応じて色を決める。
    """

    # デフォルト色
    EL1_color = "lightgray"
    EL2_color = "lightgray"
    EC_left = "lightgray"
    EC_right = "lightgray"
    QST1_left = "lightgray"
    QST1_right = "lightgray"
    QST2_left = "lightgray"
    QST2_right = "lightgray"
    QR1_color = "lightgray"
    QR2_color = "lightgray"

    # --- t = 0 ---
    if time_key == "t0":
        pass  # 何も起きてないのでニュートラル

    # --- t = t_AFC/4 : EL 内 BSM 判定直後 ---
    elif time_key == "t_quarter":
        EL1 = tl["t_quarter"]["EL1_success"]
        EL2 = tl["t_quarter"]["EL2_success"]
        EL1_color = "blue" if EL1 else "lightgray"
        EL2_color = "blue" if EL2 else "lightgray"

    # --- t = t_AFC/2 : EC 判定「前」 ---
    elif time_key == "t_half_preEC":
        EL1 = tl["t_half"]["EL1_success"]
        EL2 = tl["t_half"]["EL2_success"]
        EL1_color = "blue" if EL1 else "lightgray"
        EL2_color = "blue" if EL2 else "lightgray"

        # EC が受け取ったヘラルド（判定はまだしない）
        msg1 = tl["t_half"]["EC_msg_from_EL1"]
        msg2 = tl["t_half"]["EC_msg_from_EL2"]
        EC_left = "lightgray" if msg1 else "lightgray"
        EC_right = "lightgray" if msg2 else "lightgray"

        # QST は片側からのヘラルドだけ受け取っている状態
        q1 = tl["t_half"]["QST1_msg_from_EL1"]
        q2 = tl["t_half"]["QST2_msg_from_EL2"]
        QST1_left = "lightgray" if q1 else "lightgray"
        QST2_right = "lightgray" if q2 else "lightgray"
        # 右半分はまだ未到達なので gray のまま

    # --- t = t_AFC/2 : EC 判定「後」 ---
    elif time_key == "t_half_postEC":
        EL1 = tl["t_half"]["EL1_success"]
        EL2 = tl["t_half"]["EL2_success"]
        EL1_color = "blue" if EL1 else "lightgray"
        EL2_color = "blue" if EL2 else "lightgray"

        msg1 = tl["t_half"]["EC_msg_from_EL1"]
        msg2 = tl["t_half"]["EC_msg_from_EL2"]
        EC_judged = tl["t_half"]["EC_judged"]
        EC_success = tl["t_half"]["EC_success"]

        if not EC_judged:
            # 判定しなかった場合は preEC と同じ
            EC_left = "lightgray" if msg1 else "lightgray"
            EC_right = "lightgray" if msg2 else "lightgray"
        else:
            if EC_success:
                # 成功：EC 全体 Yellow（左右とも yellow）
                EC_left = "yellow"
                EC_right = "yellow"
            else:
                # 失敗：EC 全体 茶色＋EL も茶色
                EC_left = "lightgray"
                EC_right = "lightgray"
                EL1_color = "lightgray"
                EL2_color = "lightgray"

        # QST 左半分は t_half_preEC と同じ
        q1 = tl["t_half"]["QST1_msg_from_EL1"]
        q2 = tl["t_half"]["QST2_msg_from_EL2"]
        QST1_left = "lightgray" if q1 else "lightgray"
        QST2_right = "lightgray" if q2 else "lightgray"

    # --- t = t_AFC : QST 判定「前」
    #      → 2本の EL からのヘラルドは受け取っているが、まだ判定していない ---
    elif time_key == "t_full_preQST":
        # まず EC までの結果を反映（t_full には EC 情報もコピー済）
        EL1 = tl["t_full"]["EL1_success"]
        EL2 = tl["t_full"]["EL2_success"]
        EL1_color = "blue" if EL1 else "lightgray"
        EL2_color = "blue" if EL2 else "lightgray"

        msg1 = tl["t_half"]["EC_msg_from_EL1"]
        msg2 = tl["t_half"]["EC_msg_from_EL2"]
        EC_judged = tl["t_full"]["EC_judged"]
        EC_success = tl["t_full"]["EC_success"]

        if not EC_judged:
            EC_left = "lightgray" if msg1 else "lightgray"
            EC_right = "lightgray" if msg2 else "lightgray"
        else:
            if EC_success:
                EC_left = "yellow"
                EC_right = "yellow"
            else:
                EC_left = "lightgray"
                EC_right = "lightgray"
                EL1_color = "lightgray"
                EL2_color = "lightgray"

        # QST は 2本の EL からのヘラルドを両方受信済み
        q1_L = tl["t_full"]["QST1_msg_from_EL1"]
        q1_R = tl["t_full"]["QST1_msg_from_EL2"]
        q2_L = tl["t_full"]["QST2_msg_from_EL2"]
        q2_R = tl["t_full"]["QST2_msg_from_EL1"]

        QST1_left  = "blue" if q1_L else "lightgray"
        QST1_right = "blue" if q1_R else "lightgray"
        QST2_right  = "blue" if q2_L else "lightgray"
        QST2_left = "blue" if q2_R else "lightgray" 

        # QST 判定前なので QR はまだ gray のまま

    # --- t = t_AFC : QST 判定「後」 ---
    elif time_key == "t_full_postQST":
        EL1 = tl["t_full"]["EL1_success"]
        EL2 = tl["t_full"]["EL2_success"]
        EL1_color = "blue" if EL1 else "lightgray"
        EL2_color = "blue" if EL2 else "lightgray"

        msg1 = tl["t_half"]["EC_msg_from_EL1"]
        msg2 = tl["t_half"]["EC_msg_from_EL2"]
        EC_judged = tl["t_full"]["EC_judged"]
        EC_success = tl["t_full"]["EC_success"]

        # まず EC の色
        if not EC_judged:
            EC_left = "lightgray" if msg1 else "lightgray"
            EC_right = "lightgray" if msg2 else "lightgray"
        else:
            if EC_success:
                EC_left = EC_right = "yellow"
            else:
                EC_left = EC_right = "lightgray"
                EL1_color = EL2_color = "lightgray"

        # QST に来ているヘラルド
        q1_L = tl["t_full"]["QST1_msg_from_EL1"]
        q1_R = tl["t_full"]["QST1_msg_from_EL2"]
        q2_L = tl["t_full"]["QST2_msg_from_EL2"]
        q2_R = tl["t_full"]["QST2_msg_from_EL1"]

        QST1_left  = "blue" if q1_L else "lightgray"
        QST1_right = "blue" if q1_R else "lightgray"
        QST2_right  = "blue" if q2_L else "lightgray"
        QST2_left = "blue" if q2_R else "lightgray"

        # ここから QST 判定結果を反映
        QST_judged = tl["t_full"]["QST_judged"]
        ARC_success = tl["t_full"]["ARC_success"]

        if QST_judged:
            if ARC_success:
                # 成功：QST は green で維持し、QR を赤に
                QST1_left = QST1_right = "green"
                QST2_left = QST2_right = "green"
                QR1_color = QR2_color = "red"
            else:
                # 失敗：QR 以外の全てを茶色
                QR1_color = QR2_color = "lightgray"
                EL1_color = EL2_color = "lightgray"
                EC_left = EC_right = "lightgray"
                QST1_left = QST1_right = "lightgray"
                QST2_left = QST2_right = "lightgray"
        else:
            # 判定していない場合は preQST と同じ
            pass

    else:
        raise ValueError(f"unknown time_key: {time_key}")

    return dict(
        EL1=EL1_color, EL2=EL2_color,
        EC_left=EC_left, EC_right=EC_right,
        QST1_left=QST1_left, QST1_right=QST1_right,
        QST2_left=QST2_left, QST2_right=QST2_right,
        QR1=QR1_color, QR2=QR2_color,
    )


# =========================================
# 3. 図形のパッチを一度だけ作っておく
# =========================================
def create_patches(ax,W_QR=0.55,R_QR=0.25,W_QST_HALF=0.32,H_QST=0.8,W_EL=0.85,H_EL=0.3,W_EC_HALF=0.36,H_EC=0.6,GAP=0.18,LABEL_DX_QST = 0.08 ):
    # 10個ノードを左→右に配置
    x_positions = np.linspace(0, 9.0, 10)
    y_center = 1.3
    label_y = 0.4

    # 内部キー（参照用）と表示ラベル（図に出す文字）を分ける
    nodes = [
        ("QR_L",     "QR(L)","QR",W_QR,None,False),
        ("QST_L_1",  "QST(L1)","QST",W_QST_HALF,H_QST,False),
        ("QST_L_2",  "QST(L2)","QST",W_QST_HALF,H_QST,True),
        ("EL_1",     "EL(1)","EL",W_EL,H_EL,False),
        ("EC_L",     "EC(L)","EC_L",W_EC_HALF,H_EC,False),
        ("EC_R",     "EC(R)","EC_R",W_EC_HALF,H_EC,True),
        ("EL_2",     "EL(2)","EL", W_EL, H_EL, False),
        ("QST_R_2",  "QST(R2)","QST", W_QST_HALF, H_QST, False),
        ("QST_R_1",  "QST(R1)","QST", W_QST_HALF, H_QST, True),
        ("QR_R",     "QR(R)","QR",  W_QR,None,False),
    ]

    patches_dict = {}
    cursor=0.0
    x_centers={}
    x_edges={}
    for i, (key, label, kind, w, h, fuse_prev) in enumerate(nodes):
        gap = 0.0 if (i > 0 and fuse_prev) else GAP
        cursor += gap
        x0 = cursor
        x1 = cursor + w
        xc = (x0 + x1) / 2.0
        x_centers[key] = xc
        x_edges[key] = (x0, x1)
        cursor = x1

    # 描画
    for key, label, kind, w, h, fuse_prev in nodes:
        xc = x_centers[key]
        x0, x1 = x_edges[key]

        if kind == "QR":
            circ = patches.Circle((xc, y_center), radius=R_QR,
                                  facecolor="lightgray", edgecolor="black")
            ax.add_patch(circ)
            patches_dict[key] = circ

        elif kind == "QST":
            rect = patches.Rectangle((x0, y_center - h/2), w, h,
                                     facecolor="lightgray", edgecolor="black")
            ax.add_patch(rect)
            patches_dict[key] = rect

        elif kind == "EL":
            rect = patches.Rectangle((x0, y_center - h/2), w, h,
                                     facecolor="lightgray", edgecolor="black")
            ax.add_patch(rect)
            patches_dict[key] = rect

        elif kind == "EC_L":
            # EC分割は境界線を共有させる（くっつきの本体）
            boundary = x1  # EC_L の右端 = EC_R の左端（gap=0だから一致）
            top = (boundary, y_center + h/2)
            mid_bottom = (boundary, y_center - h/2)
            left_bottom = (x0, y_center - h/2)
            tri = patches.Polygon([top, mid_bottom, left_bottom], closed=True,
                                  facecolor="lightgray", edgecolor="black")
            ax.add_patch(tri)
            patches_dict[key] = tri

        elif kind == "EC_R":
            boundary = x0  # EC_R の左端 = EC_L の右端
            top = (boundary, y_center + h/2)
            mid_bottom = (boundary, y_center - h/2)
            right_bottom = (x1, y_center - h/2)
            tri = patches.Polygon([top, right_bottom, mid_bottom], closed=True,
                                  facecolor="lightgray", edgecolor="black")
            ax.add_patch(tri)
            patches_dict[key] = tri

        # ラベル（下）
        #ax.text(xc, label_y, label, ha="center", va="center", fontsize=9)
        dx = 0.0
        if key == "QST_L_1":
            dx = -LABEL_DX_QST
        elif key == "QST_L_2":
            dx = +LABEL_DX_QST
        elif key == "QST_R_2":
            dx = -LABEL_DX_QST
        elif key == "QST_R_1":
            dx = +LABEL_DX_QST
        ax.text(xc + dx, label_y, label, ha="center", va="center", fontsize=9)

    ax.set_ylim(0,2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.2,cursor+0.2)

    return patches_dict



# =========================================
# 4. パラパラ漫画アニメーション
# =========================================
import os
import matplotlib.pyplot as plt

def animate_flipbook(num_trials=5, p_EL=0.3, p_EC=0.9, p_QST=0.8, interval=600):
    """
    num_trials 回ぶんの試行を連続して流すアニメーション。

    1試行ごとに：
      t0
      t_quarter
      t_half_preEC
      （EC_judged=True のときだけ t_half_postEC）
      t_full_preQST
      （QST_judged=True のときだけ t_full_postQST）
    を順番に表示する。

    左上には「試行◯回目, 時刻: ...」を表示。
    """

    rng = np.random.default_rng()

    # 各 time_key に対応するラベル（左上表示用）
    time_labels = {
        "t0": "t = 0",
        "t_quarter": "t = t_AFC/4",
        "t_half_preEC":  "t = t_AFC/2 (EC before judgement)",
        "t_half_postEC": "t = t_AFC/2 (EC after judgement)",
        "t_full_preQST":  "t = t_AFC (QST before judgement)",
        "t_full_postQST": "t = t_AFC (QST after judgement)",
    }

    # まず num_trials 回ぶんのタイムラインを生成
    timelines = [
        trial_methodB_timeline(p_EL, p_EC, p_QST, rng=rng)
        for _ in range(num_trials)
    ]

    # ===== ここがポイント：trialごとのステップ列を条件付きで組み立てる =====
    # schedule は (trial_index, time_key) のリスト
    schedule = []

    for trial_idx, tl in enumerate(timelines):
        # 1試行の中のステップリスト
        steps = []

        # 常に表示する基本3ステップ
        steps.append("t0")
        steps.append("t_quarter")
        steps.append("t_half_preEC")

        # EC 判定前に「2つの成功信号が揃っているかどうか」で postEC を入れるか決める
        EC_judged = tl["t_half"]["EC_judged"]  # 両方成功ヘラルドのときのみ True
        if EC_judged:
            steps.append("t_half_postEC")

        # その後、t = t_AFC へ
        steps.append("t_full_preQST")

        # QST 判定が実際に行われるなら postQST も出す
        QST_judged = tl["t_full"]["QST_judged"]
        if QST_judged:
            steps.append("t_full_postQST")

        # 今の trial のステップを (trial_idx, time_key) 形式で schedule に追加
        for skey in steps:
            schedule.append((trial_idx, skey))

    total_frames = len(schedule)

    # ======= 描画準備 =======
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.set_xlim(-0.5, 7.7)
    ax.set_ylim(0, 2)
    ax.set_aspect("equal")
    ax.axis("off")

    patches_dict = create_patches(ax)

    # 上のタイトル（固定文言でもOK）
    title = ax.set_title("Method B timeline animation")

    # 左上に「試行◯回目, 時刻: ...」
    info_text = ax.text(
        0.02, 0.95, "",#0.02,0.95,"" 元の値
        transform=ax.transAxes,
        fontsize=11,
        ha="left",
        va="top"
    )

    fig.subplots_adjust(bottom=0.15)

    def init():
        for p in patches_dict.values():
            p.set_facecolor("lightgray")
        info_text.set_text("")
        return list(patches_dict.values()) + [title, info_text]

    def update(frame):
        # schedule から「何回目の試行」「どの時刻か」を取り出す
        trial_index, tkey = schedule[frame]
        tl = timelines[trial_index]

        colors = compute_colors(tl, tkey)
        patches_dict["EL_1"].set_facecolor(colors["EL1"])
        patches_dict["EL_2"].set_facecolor(colors["EL2"])
        patches_dict["EC_L"].set_facecolor(colors["EC_left"])
        patches_dict["EC_R"].set_facecolor(colors["EC_right"])
        patches_dict["QST_L_1"].set_facecolor(colors["QST1_left"])
        patches_dict["QST_L_2"].set_facecolor(colors["QST1_right"])
        patches_dict["QST_R_2"].set_facecolor(colors["QST2_left"])
        patches_dict["QST_R_1"].set_facecolor(colors["QST2_right"])
        patches_dict["QR_L"].set_facecolor(colors["QR1"])
        patches_dict["QR_R"].set_facecolor(colors["QR2"])


        # 左上に「試行◯回目, 時刻: ...」を表示
        info_text.set_text(
            f"trial {trial_index+1} ,  {time_labels[tkey]}"
        )

        return list(patches_dict.values()) + [title, info_text]

    ani = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=total_frames,
        interval=interval,
        blit=True,
        repeat=True,
    )

    return ani
def build_schedule_from_timelines(timelines):
    """
    animate_flipbook と同じルールで
    schedule = [(trial_idx, time_key), ...] を作る。
    """
    schedule = []

    for trial_idx, tl in enumerate(timelines):
        steps = []

        # 常に出すステップ
        steps.append("t0")
        steps.append("t_quarter")
        steps.append("t_half_preEC")

        # EC 判定が行われる trial のみ postEC を追加
        EC_judged = tl["t_half"]["EC_judged"]
        if EC_judged:
            steps.append("t_half_postEC")

        # t_AFC へ
        steps.append("t_full_preQST")

        # QST 判定が行われる trial のみ postQST を追加
        QST_judged = tl["t_full"]["QST_judged"]
        if QST_judged:
            steps.append("t_full_postQST")

        for skey in steps:
            schedule.append((trial_idx, skey))

    return schedule

import os
import matplotlib.pyplot as plt

def save_flipbook_snapshots(
    num_trials=5,
    p_EL=0.3,
    p_EC=0.9,
    p_QST=0.8,
    out_dir="snapshots",
    seed=0
):
    """
    アニメと同じ schedule に従って、
    各フレームを png 画像として保存する。

    例:
      snapshots/frame000_trial01_t0.png
      snapshots/frame001_trial01_t_quarter.png
      ...
    """

    rng = np.random.default_rng(seed)

    # ラベル（タイトルに使う用）
    time_labels = {
        "t0": "t = 0",
        "t_quarter": "t = t_AFC/4",
        "t_half_preEC":  "t = t_AFC/2 (EC before judgement)",
        "t_half_postEC": "t = t_AFC/2 (EC after judgement)",
        "t_full_preQST":  "t = t_AFC (QST before judgement)",
        "t_full_postQST": "t = t_AFC (QST after judgement)",
    }

    # 1. 全 trial のタイムラインを生成
    timelines = [
        trial_methodB_timeline(p_EL, p_EC, p_QST, rng=rng)
        for _ in range(num_trials)
    ]

    # 2. アニメと同じルールで schedule を構築
    schedule = build_schedule_from_timelines(timelines)

    # 出力フォルダ作成
    os.makedirs(out_dir, exist_ok=True)

    # 3. schedule を1フレームずつ回して保存
    for frame_idx, (trial_idx, tkey) in enumerate(schedule):
        tl = timelines[trial_idx]

        # --- 1枚ぶんの Figure を作る ---
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.set_xlim(-0.5, 7.7)
        ax.set_ylim(0, 2)
        ax.set_aspect("equal")
        ax.axis("off")

        patches_dict = create_patches(ax)

        # 色を計算して反映
        colors = compute_colors(tl, tkey)

        patches_dict["EL_1"].set_facecolor(colors["EL1"])
        patches_dict["EL_2"].set_facecolor(colors["EL2"])
        patches_dict["EC_L"].set_facecolor(colors["EC_left"])
        patches_dict["EC_R"].set_facecolor(colors["EC_right"])
        patches_dict["QST_L_1"].set_facecolor(colors["QST1_left"])
        patches_dict["QST_L_2"].set_facecolor(colors["QST1_right"])
        patches_dict["QST_R_2"].set_facecolor(colors["QST2_left"])
        patches_dict["QST_R_1"].set_facecolor(colors["QST2_right"])
        patches_dict["QR_L"].set_facecolor(colors["QR1"])
        patches_dict["QR_R"].set_facecolor(colors["QR2"])
        # タイトルに「試行◯回目 & 時刻」を表示
        ax.set_title(
            f"frame {frame_idx+1}, Trial {trial_idx+1}/{num_trials}, {time_labels[tkey]}"
        )

        # ファイル名：フレーム番号 + trial番号 + time_key
        fname = f"frame{frame_idx:03d}_trial{trial_idx+1:02d}_{tkey}.png"
        fpath = os.path.join(out_dir, fname)

        fig.tight_layout()
        fig.savefig(fpath, dpi=150)
        plt.close(fig)

        print("saved:", fpath)
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
            loaded_dict["n"] = np.arange(1,11,1) 
            #loaded_dict["n"] = 1
            loaded_dict["big_N"] = 1
            #loaded_dict["big_N"] = np.arange(1,11,1)
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
        tAFC=param_dict["t_AFC"]
        n_attempts=10000
        N=5000
        for j in tqdm(range(N)):
            for i in range(n_attempts):#1s=100μs(t_AFC)*10000,論文では取りあえず10000回
                tl,ARC_success = trial_methodB_timeline(p_EL=p_arc_gen, p_EC=eta_conn, p_QST=eta_QR, rng=rng)
                if ARC_success==True:
                    successes +=1      
        p=successes/N
        error=np.sqrt(p*(1.0-p)/N) 
        print(f'p(N=1,n=1)={p:.6f},error(N=1,n=1)={error:.6f}')

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
    save_flipbook_snapshots(
        num_trials=10,
        #p_EL=p_EL,
        p_EL=0.6,
        #p_EC=p_EC,
        p_EC=0.8,
        #p_QST=p_QST,
        p_QST=0.5,
        out_dir="snapshots_2",
        seed=0,
    )
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