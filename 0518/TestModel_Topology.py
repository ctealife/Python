# -*- coding: utf-8 -*-
import os
import sys
import time
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import h5py
from keras import Input, Model
from keras.layers import Conv1D, MaxPooling1D, Flatten, BatchNormalization, Dense, ReLU, Dropout
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Lambda
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
)
import pandas as pd
import json


sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------- 多模型測試路徑設定 ----------------
# 本測試架構假設：四個 model 都測試同一份固定 2Tx 同時發射的測試資料。
# 實驗意義是 processing ablation：
# 2PN2R = 使用 PN1+PN2，使用 Rx1+Rx2
# 2PN1R = 使用 PN1+PN2，使用 Rx1
# 1PN2R = 只使用 PN1，使用 Rx1+Rx2
# 1PN1R = 只使用 PN1，使用 Rx1

TEST_DATA_DIR = r"C:\鈞論文\paper_2026\CNN\training_result\0518\SCATTER 0800\2T2R"
MODEL_ROOT    = r"C:\鈞論文\paper_2026\CNN\training_result\0518\SCATTER 0800"

TEST_RX1_PATH = os.path.join(TEST_DATA_DIR, "testing_data_rx1.mat")
TEST_RX2_PATH = os.path.join(TEST_DATA_DIR, "testing_data_rx2.mat")

out_dir = os.path.join(MODEL_ROOT, "TestResult_AllModels")
fig_dir = os.path.join(out_dir, "figures")
pred_dir = os.path.join(out_dir, "predictions")
diag_dir = os.path.join(out_dir, "diagnostics")

for _d in [out_dir, fig_dir, pred_dir, diag_dir]:
    os.makedirs(_d, exist_ok=True)

# 若你的訓練資料夾仍使用 2T2R / 2T1R / 1T2R / 1T1R 命名，
# 這裡保留訓練時的檔名，但測試輸出使用 2PN2R / 2PN1R / 1PN2R / 1PN1R 命名。
PROCESSING_CONFIG = {
    "2T2R": {
        "active_rx": [1, 2],
        "mf_tx": [1, 2],
        "train_name": "2T2R",
        "model_dir": os.path.join(MODEL_ROOT, "Training_Result_2T2R"),
        "weights_name": "2T2R_det_pos_geo_weights.h5",
        "norm_name": "2T2R_norm_params.pkl",
    },
    "2T1R": {
        "active_rx": [1],
        "mf_tx": [1, 2],
        "train_name": "2T1R",
        "model_dir": os.path.join(MODEL_ROOT, "Training_Result_2T1R"),
        "weights_name": "2T1R_det_pos_geo_weights.h5",
        "norm_name": "2T1R_norm_params.pkl",
    },
    "1T2R": {
        "active_rx": [1, 2],
        "mf_tx": [1],
        "train_name": "1T2R",
        "model_dir": os.path.join(MODEL_ROOT, "Training_Result_1T2R"),
        "weights_name": "1T2R_det_pos_geo_weights.h5",
        "norm_name": "1T2R_norm_params.pkl",
    },
    "1T1R": {
        "active_rx": [1],
        "mf_tx": [1],
        "train_name": "1T1R",
        "model_dir": os.path.join(MODEL_ROOT, "Training_Result_1T1R"),
        "weights_name": "1T1R_det_pos_geo_weights.h5",
        "norm_name": "1T1R_norm_params.pkl",
    },
}

FIXED_THRESHOLD = 0.5
FIXED_PFA_LIST = [0.05, 0.01]

log_path = os.path.join(
    out_dir,
    f"{os.path.splitext(os.path.basename(__file__))[0]}_terminal_log.txt"
)


class TeeLogger:
    def __init__(self, *streams):
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8")

    def write(self, message):
        for stream in self.streams:
            stream.write(message)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def load_mat73_selected(mat_path, required_keys=None, optional_keys=None, verbose=True):
    """
    讀取 MATLAB -v7.3 MAT 檔案。
    h5py 讀出的 MATLAB array 維度通常與 MATLAB / scipy.loadmat 相反，
    因此對 ndarray 使用 .T 還原成 Python 端原本訓練程式使用的 shape。

    例如：
    MATLAB feature_4d: (N, P, L, 2)
    h5py raw shape    : (2, L, P, N)
    after .T          : (N, P, L, 2)
    """
    if required_keys is None:
        required_keys = []

    if optional_keys is None:
        optional_keys = []

    data = {}

    with h5py.File(mat_path, "r") as f:
        for key in required_keys:
            if key not in f:
                raise KeyError(f"{mat_path} missing required field: {key}")

        load_keys = list(required_keys) + [
            key for key in optional_keys
            if key in f and key not in required_keys
        ]

        for key in load_keys:
            obj = f[key]

            if isinstance(obj, h5py.Dataset):
                arr = np.array(obj)

                # MATLAB char / string 類型通常不會在此測試流程中用到。
                # 若是數值 array，轉置以恢復 MATLAB 維度順序。
                if arr.ndim >= 2:
                    arr = arr.T

                data[key] = arr

                if verbose:
                    print(f"[load_mat73] {key}: shape={arr.shape}, dtype={arr.dtype}")
            else:
                raise TypeError(f"{key} is not a dataset. Current loader only supports numeric datasets.")

    return data

def os_cfar_ratio_layer(x, win_len=31, guard_len=3, rank_ratio=0.75, eps=1e-8, name="os_cfar_ratio"):
    def _layer(t):
        t = tf.squeeze(t, axis=-1)

        pad = win_len // 2
        t_pad = tf.pad(t, [[0, 0], [pad, pad]], mode="REFLECT")
        frames = tf.signal.frame(t_pad, frame_length=win_len, frame_step=1, axis=1)

        center = win_len // 2
        idx = tf.range(win_len)
        valid_mask = tf.logical_or(idx < (center - guard_len), idx > (center + guard_len))
        valid_mask = tf.cast(valid_mask, t.dtype)

        large_val = tf.constant(1e9, dtype=t.dtype)
        masked_frames = frames + (1.0 - valid_mask)[tf.newaxis, tf.newaxis, :] * large_val
        sorted_vals = tf.sort(masked_frames, axis=-1, direction="ASCENDING")

        ref_len = win_len - (2 * guard_len + 1)
        k = tf.cast(tf.round(rank_ratio * tf.cast(ref_len - 1, tf.float32)), tf.int32)
        k = tf.clip_by_value(k, 0, ref_len - 1)

        z_os = sorted_vals[:, :, k]
        y = t / (z_os + eps)

        return tf.expand_dims(y, axis=-1)

    return Lambda(_layer, name=name)(x)


def log_layer(x):
    return tf.math.log(x + 1e-6)


def build_mimo_model(
    num_tx,
    num_rx,
    num_pulses,
    sequence_length,
    num_features,
    pnSeq_active,
    pos_min,
    pos_max,
    tx_pos_active,
    rx_pos_active,
    rbi_min,
    rbi_max,
    cfar_win_len=31,
    cfar_guard_len=3,
    topology_name="MIMO",
):
    input_signal = Input(
        shape=(num_rx, num_pulses, sequence_length, num_features),
        name="mimo_feature_input",
    )

    pn_mf_list = [
        pn[::-1].copy().astype("float32")
        for pn in pnSeq_active
    ]

    pn_len_set = set(len(pn) for pn in pn_mf_list)
    if len(pn_len_set) != 1:
        raise ValueError("active PN length mismatch.")

    kernel_size = len(pn_mf_list[0])

    # 每個 PN branch 對應 I/Q 兩個 filter，因此 filters = 2 * num_tx
    init_weight = np.zeros((kernel_size, 2, 2 * num_tx), dtype=np.float32)

    for k, pn_mf in enumerate(pn_mf_list):
        init_weight[:, 0, 2 * k] = pn_mf
        init_weight[:, 1, 2 * k + 1] = pn_mf

    x = Lambda(
        lambda t: tf.reshape(t, (-1, sequence_length, num_features)),
        name="merge_batch_rx_pulse",
    )(input_signal)

    x = Conv1D(
        filters=2 * num_tx,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer=Constant(init_weight),
        use_bias=False,
        trainable=True,
        name="pn_matched_filter_bank",
    )(x)

    x = Lambda(
        lambda t: tf.reshape(t, (-1, sequence_length, num_tx, 2)),
        name="split_tx_iq",
    )(x)

    x = Lambda(
        lambda t: tf.reduce_sum(tf.square(t), axis=-1),
        name="per_tx_power",
    )(x)

    def to_virtual_channels_for_cfar(t):
        batch_size_dyn = tf.shape(t)[0] // (num_rx * num_pulses)

        t = tf.reshape(
            t,
            (batch_size_dyn, num_rx, num_pulses, sequence_length, num_tx),
        )
        t = tf.transpose(t, perm=[0, 4, 1, 2, 3])
        t = tf.reshape(t, (-1, sequence_length, 1))
        return t

    x = Lambda(
        to_virtual_channels_for_cfar,
        name="to_virtual_channels_for_cfar",
    )(x)

    x = os_cfar_ratio_layer(
        x,
        win_len=cfar_win_len,
        guard_len=cfar_guard_len,
        rank_ratio=0.85,
        eps=1e-8,
        name="os_cfar_ratio",
    )

    x = Lambda(log_layer, name="log_power")(x)

    def restore_mimo_map(t):
        batch_size_dyn = tf.shape(t)[0] // (num_tx * num_rx * num_pulses)

        t = tf.reshape(
            t,
            (batch_size_dyn, num_tx, num_rx, num_pulses, sequence_length, 1),
        )
        t = tf.squeeze(t, axis=-1)
        t = tf.transpose(t, perm=[0, 4, 1, 2, 3])
        t = tf.reshape(
            t,
            (batch_size_dyn, sequence_length, num_tx * num_rx * num_pulses)
        )
        return t

    x = Lambda(
        restore_mimo_map,
        name="mimo_virtual_pulse_as_channel",
    )(x)

    x = Conv1D(filters=32, kernel_size=5, padding="same", name="conv1d_2")(x)
    x = BatchNormalization(name="bn_1")(x)
    x = ReLU(name="relu_1")(x)

    x = Conv1D(filters=64, kernel_size=3, padding="same", name="conv1d_3")(x)
    x = BatchNormalization(name="bn_2")(x)
    x = ReLU(name="relu_2")(x)

    x = MaxPooling1D(pool_size=4, strides=4, name="maxpool_1")(x)

    x = Flatten(name="flatten")(x)
    x = Dense(256, activation="relu", name="dense_1")(x)
    x = Dropout(0.3, name="dropout_1")(x)
    x = Dense(128, activation="relu", name="dense_2")(x)
    x = Dropout(0.2, name="dropout_2")(x)

    det_output = Dense(1, activation="sigmoid", name="det_output")(x)
    pos_output = Dense(2, activation="sigmoid", name="pos_output")(x)

    pos_min_tf = tf.constant(pos_min.reshape(1, 2), dtype=tf.float32)
    pos_max_tf = tf.constant(pos_max.reshape(1, 2), dtype=tf.float32)
    tx_tf = tf.constant(tx_pos_active, dtype=tf.float32)
    rx_tf = tf.constant(rx_pos_active, dtype=tf.float32)
    rbi_min_tf = tf.constant(rbi_min, dtype=tf.float32)
    rbi_max_tf = tf.constant(rbi_max, dtype=tf.float32)

    def geo_from_pos(pos_norm):
        pos = pos_norm * (pos_max_tf - pos_min_tf) + pos_min_tf

        p = pos[:, tf.newaxis, tf.newaxis, :]
        tx = tx_tf[tf.newaxis, :, tf.newaxis, :]
        rx = rx_tf[tf.newaxis, tf.newaxis, :, :]

        rt = tf.norm(p - tx, axis=-1)
        rr = tf.norm(p - rx, axis=-1)
        rho = rt + rr

        # flatten order:
        # for tx in mf_tx, for rx in active_rx
        rho = tf.reshape(rho, (-1, num_tx * num_rx))
        rho_norm = (rho - rbi_min_tf) / (rbi_max_tf - rbi_min_tf + 1e-8)

        return rho_norm

    geo_output = Lambda(geo_from_pos, name="geo_output")(pos_output)

    return Model(
        inputs=input_signal,
        outputs={
            "det_output": det_output,
            "pos_output": pos_output,
            "geo_output": geo_output,
        },
        name=f"mimo_{topology_name}_det_pos_geo",
    )


def require_keys(data, keys, mat_name):
    missing = [key for key in keys if key not in data]
    if missing:
        raise KeyError(f"{mat_name} missing required fields: {missing}")


def threshold_sweep_report(y_true, det_prob, thresholds=None):
    y_true = np.asarray(y_true).squeeze().astype(int)
    det_prob = np.asarray(det_prob).squeeze()

    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)

    print("\n========== Detection Threshold Sweep ==========")
    print(" th  |  TN   FP   FN   TP  | Precision | Recall/Pd |   F1   |  P_FA")

    for th in thresholds:
        pred = (det_prob >= th).astype(int)

        cm = confusion_matrix(y_true, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        p_fa = fp / max(fp + tn, 1)

        print(
            f"{th:4.2f} | "
            f"{tn:4d} {fp:4d} {fn:4d} {tp:4d} | "
            f"{precision:9.4f} | "
            f"{recall:9.4f} | "
            f"{f1:6.4f} | "
            f"{p_fa:6.4f}"
        )


def scnr_detection_accuracy_report_and_plot(
    scnr_db,
    y_true,
    y_pred,
    out_dir,
    tag="Test",
    bin_width=5.0,
):
    scnr_db = np.asarray(scnr_db).squeeze()
    y_true = np.asarray(y_true).squeeze().astype(int)
    y_pred = np.asarray(y_pred).squeeze().astype(int)

    valid_mask = (y_true == 1) & np.isfinite(scnr_db)

    scnr_t = scnr_db[valid_mask]
    y_pred_t = y_pred[valid_mask]

    if len(scnr_t) == 0:
        print(f"[{tag}] No target samples with valid SCNR. Skip SCNR-accuracy report.")
        return

    scnr_min = np.floor(np.nanmin(scnr_t) / bin_width) * bin_width
    scnr_max = np.ceil(np.nanmax(scnr_t) / bin_width) * bin_width

    if scnr_max <= scnr_min:
        scnr_max = scnr_min + bin_width

    bin_edges = np.arange(scnr_min, scnr_max + bin_width, bin_width)

    print(f"\n[Info] {tag} SCNR auto bins:")
    print(f"SCNR min = {np.nanmin(scnr_t):.2f} dB")
    print(f"SCNR max = {np.nanmax(scnr_t):.2f} dB")
    print(f"bin width = {bin_width:.2f} dB")
    print(f"number of bins = {len(bin_edges) - 1}")

    bin_x = []
    acc_list = []
    count_list = []

    print(f"\n========== {tag} SCNR-binned Detection Accuracy ==========")
    print(" SCNR bin (dB)      | median SCNR | count | correct | accuracy/Pd")

    for i in range(len(bin_edges) - 1):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]

        if i == len(bin_edges) - 2:
            mask = (scnr_t >= lo) & (scnr_t <= hi)
        else:
            mask = (scnr_t >= lo) & (scnr_t < hi)

        count = np.sum(mask)
        label = f"[{lo:7.2f}, {hi:7.2f})"

        if count == 0:
            print(f"{label:18s} | {'NaN':>11s} | {count:5d} | {'-':>7s} | {'NaN':>10s}")
            continue

        correct = np.sum(y_pred_t[mask] == 1)
        acc = correct / count
        x_now = np.nanmedian(scnr_t[mask])

        bin_x.append(x_now)
        acc_list.append(acc)
        count_list.append(count)

        print(
            f"{label:18s} | "
            f"{x_now:11.2f} | "
            f"{count:5d} | "
            f"{correct:7d} | "
            f"{acc:10.4f}"
        )

    if len(acc_list) == 0:
        print(f"[{tag}] No non-empty SCNR bins. Skip plot.")
        return

    bin_x = np.array(bin_x)
    acc_list = np.array(acc_list)
    count_list = np.array(count_list)

    order = np.argsort(bin_x)
    bin_x = bin_x[order]
    acc_list = acc_list[order]
    count_list = count_list[order]

    plt.figure(figsize=(8, 4.5))
    plt.plot(bin_x, acc_list, marker="o")
    plt.xlabel("SCNR (dB)")
    plt.ylabel("Detection Accuracy / Pd")
    plt.title(f"{tag} Detection Accuracy vs SCNR")
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)

    for x, y, n in zip(bin_x, acc_list, count_list):
        plt.text(x, y, f"n={n}", fontsize=8, ha="center", va="bottom")

    save_path = os.path.join(out_dir, f"{tag.lower()}_scnr_detection_accuracy_auto.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"[Info] {tag} SCNR-accuracy figure saved: {save_path}")

PATH_KEY_MAP = {
    (1, 1): "Rbi_11_vec",
    (1, 2): "Rbi_12_vec",
    (2, 1): "Rbi_21_vec",
    (2, 2): "Rbi_22_vec",
}


def safe_nanmean(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.nanmean(x))


def safe_nanrmse(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.sqrt(np.nanmean(x ** 2)))


def safe_nanmedian(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.nanmedian(x))


def safe_nanpercentile(x, q):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.nanpercentile(x, q))

def main():
    print("\n========== GPU SET... ==========")
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    if not os.path.exists(TEST_RX1_PATH):
        raise FileNotFoundError(f"Cannot find Rx1 test MAT file: {TEST_RX1_PATH}")

    if not os.path.exists(TEST_RX2_PATH):
        raise FileNotFoundError(f"Cannot find Rx2 test MAT file: {TEST_RX2_PATH}")

    _terminal_stdout = sys.stdout
    _terminal_stderr = sys.stderr
    _log_file = open(log_path, "w", encoding="utf-8-sig", errors="replace")
    sys.stdout = TeeLogger(_terminal_stdout, _log_file)
    sys.stderr = TeeLogger(_terminal_stderr, _log_file)

    try:
        print("\n========== Load Shared 2Tx Test Data ==========")
        common_required_keys = [
            "feature_4d",
            "label_bin",
            "sample_id",
            "SCNR_dB_vec",
            "target_x_vec",
            "target_y_vec",
            "Rbi_11_vec",
            "Rbi_12_vec",
            "Rbi_21_vec",
            "Rbi_22_vec",
            "tx_pos_all",
            "rx_pos_all",
            "pnSeq1",
            "pnSeq2",
        ]

        common_optional_keys = [
            "SCNR_nominal_dB_vec",
            "snr_vec",
            "rcs_vec",
            "noise_rcs_vec",
            "label_range_cls",
        ]

        print(f"[Info] Rx1 MAT: {TEST_RX1_PATH}", flush=True)
        t0 = time.time()
        data1 = load_mat73_selected(
            TEST_RX1_PATH,
            required_keys=common_required_keys,
            optional_keys=common_optional_keys,
            verbose=False,
        )
        print(f"[Info] Rx1 loaded in {time.time() - t0:.2f} sec", flush=True)

        print(f"[Info] Rx2 MAT: {TEST_RX2_PATH}", flush=True)
        t0 = time.time()
        data2 = load_mat73_selected(
            TEST_RX2_PATH,
            required_keys=common_required_keys,
            optional_keys=common_optional_keys,
            verbose=False,
        )
        print(f"[Info] Rx2 loaded in {time.time() - t0:.2f} sec", flush=True)

        require_keys(data1, common_required_keys, "Rx1 MAT")
        require_keys(data2, common_required_keys, "Rx2 MAT")

        # 檢查 Rx1/Rx2 是否為同一批測試樣本
        check_same_keys = [
            "label_bin",
            "sample_id",
            "target_x_vec",
            "target_y_vec",
            "SCNR_dB_vec",
            "Rbi_11_vec",
            "Rbi_12_vec",
            "Rbi_21_vec",
            "Rbi_22_vec",
        ]

        for key in check_same_keys:
            v1 = np.array(data1[key]).squeeze()
            v2 = np.array(data2[key]).squeeze()

            if not np.allclose(v1, v2, equal_nan=True):
                raise ValueError(f"Rx1 and Rx2 {key} mismatch. Metadata may be misaligned.")

        if data1["feature_4d"].shape != data2["feature_4d"].shape:
            raise ValueError(
                f"Rx1/Rx2 feature_4d shape mismatch: "
                f"{data1['feature_4d'].shape} vs {data2['feature_4d'].shape}"
            )

        shared_test_data = {
            "rx1": data1,
            "rx2": data2,
        }

        def run_one_model_test(mode_name, cfg, shared_test_data):
            print("\n" + "=" * 90)
            print(f"[Test Mode] {mode_name}")
            print("=" * 90)

            active_rx = cfg["active_rx"]
            mf_tx = cfg["mf_tx"]
            train_name = cfg["train_name"]

            active_tx_idx = [t - 1 for t in mf_tx]
            active_rx_idx = [r - 1 for r in active_rx]

            num_tx = len(mf_tx)
            num_rx_expected = len(active_rx)
            num_paths = num_tx * num_rx_expected

            weights_path = os.path.join(cfg["model_dir"], cfg["weights_name"])
            norm_path = os.path.join(cfg["model_dir"], cfg["norm_name"])

            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"[{mode_name}] Cannot find weights: {weights_path}")

            if not os.path.exists(norm_path):
                raise FileNotFoundError(f"[{mode_name}] Cannot find norm params: {norm_path}")

            data_ref = shared_test_data["rx1"]

            # ---------------- 組合 X_test: (N, Rx, P, L, I/Q) ----------------
            X_list = []

            for rx_id in active_rx:
                if rx_id == 1:
                    X_rx = shared_test_data["rx1"]["feature_4d"].astype("float32")
                elif rx_id == 2:
                    X_rx = shared_test_data["rx2"]["feature_4d"].astype("float32")
                else:
                    raise ValueError(f"[{mode_name}] unknown Rx id: {rx_id}")

                if X_rx.ndim != 4:
                    raise ValueError(
                        f"[{mode_name}] Rx{rx_id} feature_4d must be 4D (N,P,L,2), got {X_rx.shape}"
                    )

                X_list.append(X_rx)

            X_test = np.stack(X_list, axis=1).astype("float32")
            del X_list

            if X_test.ndim != 5:
                raise ValueError(f"[{mode_name}] X_test must be 5D (N,Rx,P,L,C), got {X_test.shape}")

            num_samples = X_test.shape[0]
            num_rx = X_test.shape[1]
            num_pulses = X_test.shape[2]
            sequence_length = X_test.shape[3]
            num_features = X_test.shape[4]

            if num_rx != num_rx_expected:
                raise ValueError(
                    f"[{mode_name}] num_rx mismatch: expected {num_rx_expected}, got {num_rx}"
                )

            if num_features != 2:
                raise ValueError(f"[{mode_name}] num_features must be 2 (I/Q), got {num_features}")

            # ---------------- label / metadata ----------------
            y_det = np.array(data_ref["label_bin"]).squeeze().astype("float32")
            true_bin = y_det.squeeze().astype(int)
            target_mask = true_bin == 1

            sample_id = np.array(data_ref["sample_id"]).squeeze().astype("int32")
            scnr_vec = np.array(data_ref["SCNR_dB_vec"]).squeeze().astype("float32")

            target_x = np.array(data_ref["target_x_vec"]).squeeze().astype("float32")
            target_y = np.array(data_ref["target_y_vec"]).squeeze().astype("float32")
            pos_raw = np.stack([target_x, target_y], axis=1).astype("float32")

            if len(true_bin) != num_samples:
                raise ValueError(
                    f"[{mode_name}] label length mismatch: X={num_samples}, y={len(true_bin)}"
                )

            # ---------------- geometry labels: mf_tx × active_rx ----------------
            selected_rbi_keys = [
                PATH_KEY_MAP[(tx_id, rx_id)]
                for tx_id in mf_tx
                for rx_id in active_rx
            ]

            rbi_raw = np.stack(
                [
                    np.array(data_ref[key]).squeeze()
                    for key in selected_rbi_keys
                ],
                axis=1,
            ).astype("float32")

            if rbi_raw.shape[1] != num_paths:
                raise ValueError(
                    f"[{mode_name}] rbi_raw path mismatch: expected {num_paths}, got {rbi_raw.shape[1]}"
                )

            # ---------------- PN / geometry / normalization ----------------
            tx_pos_all_full = np.array(data_ref["tx_pos_all"]).astype("float32")
            rx_pos_all_full = np.array(data_ref["rx_pos_all"]).astype("float32")

            pnSeq_by_tx = {
                1: np.array(data_ref["pnSeq1"]).squeeze().astype("float32"),
                2: np.array(data_ref["pnSeq2"]).squeeze().astype("float32"),
            }

            pnSeq_active = [pnSeq_by_tx[tx_id] for tx_id in mf_tx]

            with open(norm_path, "rb") as f:
                norm_params = pickle.load(f)

            pos_min = np.asarray(norm_params["pos_min"], dtype="float32")
            pos_max = np.asarray(norm_params["pos_max"], dtype="float32")
            rbi_min = float(norm_params["rbi_min"])
            rbi_max = float(norm_params["rbi_max"])

            tx_pos_active = np.asarray(
                norm_params.get("tx_pos_active", tx_pos_all_full[active_tx_idx, :]),
                dtype="float32"
            )
            rx_pos_active = np.asarray(
                norm_params.get("rx_pos_active", rx_pos_all_full[active_rx_idx, :]),
                dtype="float32"
            )

            pn_lengths = [len(pn) for pn in pnSeq_active]
            if len(set(pn_lengths)) != 1:
                raise ValueError(f"[{mode_name}] active PN length mismatch: {pn_lengths}")

            if pn_lengths[0] % 31 != 0:
                raise ValueError(f"[{mode_name}] PN length is not multiple of 31: {pn_lengths[0]}")

            Ovsamp_est = pn_lengths[0] // 31
            cfar_win_len = 31 * Ovsamp_est
            if cfar_win_len % 2 == 0:
                cfar_win_len += 1
            cfar_guard_len = 3 * Ovsamp_est

            print(f"[Info] mode_name={mode_name}, train_name={train_name}")
            print(f"[Info] active_rx={active_rx}, mf_tx={mf_tx}")
            print(f"[Info] X_test shape: {X_test.shape}")
            print(f"[Info] num_paths={num_paths}, selected_rbi_keys={selected_rbi_keys}")
            print(f"[Info] target samples: {np.sum(true_bin == 1)}, no-target samples: {np.sum(true_bin == 0)}")
            print(f"[Info] SCNR range: {np.nanmin(scnr_vec):.2f} ~ {np.nanmax(scnr_vec):.2f} dB")
            print(f"[Info] pos_min={pos_min}, pos_max={pos_max}")
            print(f"[Info] rbi_min={rbi_min:.4f}, rbi_max={rbi_max:.4f}")
            print(f"[Info] Ovsamp_est={Ovsamp_est}, cfar_win_len={cfar_win_len}, cfar_guard_len={cfar_guard_len}")

            # ---------------- build model / load weights ----------------
            model = build_mimo_model(
                num_tx=num_tx,
                num_rx=num_rx,
                num_pulses=num_pulses,
                sequence_length=sequence_length,
                num_features=num_features,
                pnSeq_active=pnSeq_active,
                pos_min=pos_min,
                pos_max=pos_max,
                tx_pos_active=tx_pos_active,
                rx_pos_active=rx_pos_active,
                rbi_min=rbi_min,
                rbi_max=rbi_max,
                cfar_win_len=cfar_win_len,
                cfar_guard_len=cfar_guard_len,
                topology_name=mode_name,
            )

            model.load_weights(weights_path)
            num_parameters = int(model.count_params())

            print(f"[Info] Model weights loaded: {weights_path}")
            print(f"[Info] Norm params loaded: {norm_path}")
            print(f"[Info] model parameters: {num_parameters}")

            # ---------------- predict ----------------
            print(f"\n========== Test Predict: {mode_name} ==========")
            pred_test = model.predict(X_test, verbose=0)

            if isinstance(pred_test, dict):
                det_test_prob = pred_test["det_output"].squeeze()
                pos_test_pred_norm = pred_test["pos_output"]
                geo_test_pred_norm = pred_test["geo_output"]
            else:
                det_test_prob = pred_test[0].squeeze()
                pos_test_pred_norm = pred_test[1]
                geo_test_pred_norm = pred_test[2]

            det_threshold = FIXED_THRESHOLD
            det_test_pred = (det_test_prob >= det_threshold).astype(int)

            pos_test_pred_norm = np.clip(pos_test_pred_norm, 0.0, 1.0)
            pos_test_pred = pos_test_pred_norm * (pos_max - pos_min) + pos_min

            pos_err_m = np.full(len(true_bin), np.nan, dtype="float32")
            if np.sum(target_mask) > 0:
                pos_err_xy_all = pos_test_pred[target_mask] - pos_raw[target_mask]
                pos_err_m[target_mask] = np.linalg.norm(pos_err_xy_all, axis=1)

            # ---------------- error samples ----------------
            wrong_idx = np.where(det_test_pred != true_bin)[0]

            print("\n========== Detection Error Samples ==========")
            print(f"[{mode_name}] error samples: {len(wrong_idx)}")

            for k in wrong_idx[:20]:
                print(
                    f"test_index={k}, "
                    f"sample_id={sample_id[k]}, "
                    f"true_det={int(true_bin[k])}, "
                    f"pred_det={det_test_pred[k]}, "
                    f"prob={det_test_prob[k]:.4f}, "
                    f"SCNR={scnr_vec[k]:.2f} dB"
                )

            # ---------------- detection metrics ----------------
            test_precision_bin = precision_score(true_bin, det_test_pred, zero_division=0)
            test_recall_bin = recall_score(true_bin, det_test_pred, zero_division=0)
            test_f1_bin = f1_score(true_bin, det_test_pred, zero_division=0)
            test_cm_bin = confusion_matrix(true_bin, det_test_pred, labels=[0, 1])
            tn, fp, fn, tp = test_cm_bin.ravel()
            test_pfa = fp / max(fp + tn, 1)

            threshold_sweep_report(
                y_true=true_bin,
                det_prob=det_test_prob,
                thresholds=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            )

            print("\n========== Detection Metrics ==========")
            print(f"[{mode_name}] threshold: {det_threshold:.2f}")
            print(
                f"[{mode_name}] precision: {test_precision_bin:.4f}, "
                f"recall/Pd: {test_recall_bin:.4f}, "
                f"F1: {test_f1_bin:.4f}, "
                f"P_FA: {test_pfa:.4f}"
            )
            print(test_cm_bin)

            # ---------------- confusion matrix figure ----------------
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(test_cm_bin, cmap="Blues")

            ax.set_title(f"{mode_name} Detection Confusion Matrix")
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("True label")

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["No Target", "Target"])
            ax.set_yticklabels(["No Target", "Target"])

            threshold = test_cm_bin.max() / 2

            for ii in range(test_cm_bin.shape[0]):
                for jj in range(test_cm_bin.shape[1]):
                    text_color = "white" if test_cm_bin[ii, jj] > threshold else "black"
                    ax.text(
                        jj,
                        ii,
                        str(test_cm_bin[ii, jj]),
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=16,
                    )

            fig.colorbar(im, ax=ax)
            plt.tight_layout()

            cm_fig_path = os.path.join(fig_dir, f"confusion_matrix_{mode_name}.png")
            plt.savefig(cm_fig_path, dpi=200)
            plt.close()

            print(f"[Info] Confusion matrix figure saved: {cm_fig_path}")

            # ---------------- position metrics ----------------
            if np.sum(target_mask) > 0:
                pos_test_err = pos_test_pred[target_mask] - pos_raw[target_mask]
                pos_mae_xy = np.mean(np.abs(pos_test_err), axis=0)
                pos_rmse_xy = np.sqrt(np.mean(pos_test_err ** 2, axis=0))
                pos_euclidean_mae = np.mean(np.linalg.norm(pos_test_err, axis=1))
                pos_euclidean_rmse = np.sqrt(np.mean(np.linalg.norm(pos_test_err, axis=1) ** 2))

                print(f"[{mode_name}] X MAE: {pos_mae_xy[0]:.4f} m")
                print(f"[{mode_name}] Y MAE: {pos_mae_xy[1]:.4f} m")
                print(f"[{mode_name}] X RMSE: {pos_rmse_xy[0]:.4f} m")
                print(f"[{mode_name}] Y RMSE: {pos_rmse_xy[1]:.4f} m")
                print(f"[{mode_name}] Position Euclidean MAE: {pos_euclidean_mae:.4f} m")
                print(f"[{mode_name}] Position Euclidean RMSE: {pos_euclidean_rmse:.4f} m")
            else:
                pos_mae_xy = np.array([np.nan, np.nan])
                pos_rmse_xy = np.array([np.nan, np.nan])
                pos_euclidean_mae = np.nan
                pos_euclidean_rmse = np.nan
                print(f"[{mode_name}] No target samples. Skip position regression metrics.")

            # ---------------- SCNR-binned plot, 保留原本輸出 ----------------
            scnr_detection_accuracy_report_and_plot(
                scnr_db=scnr_vec,
                y_true=true_bin,
                y_pred=det_test_pred,
                out_dir=fig_dir,
                tag=mode_name,
                bin_width=5.0,
            )

            # ---------------- save per-mode prediction npz ----------------
            pred_path = os.path.join(pred_dir, f"predictions_{mode_name}.npz")
            np.savez(
                pred_path,
                det_prob=det_test_prob,
                det_pred=det_test_pred,
                true_bin=true_bin,
                sample_id=sample_id,
                pos_pred=pos_test_pred,
                pos_true=pos_raw,
                pos_err_m=pos_err_m,
                geo_pred_norm=geo_test_pred_norm,
                rbi_true=rbi_raw,
                scnr_db=scnr_vec,
                confusion_matrix=test_cm_bin,
                active_rx=np.array(active_rx, dtype=np.int32),
                mf_tx=np.array(mf_tx, dtype=np.int32),
                selected_rbi_keys=np.array(selected_rbi_keys),
            )
            print(f"[Info] Test predictions saved: {pred_path}")

            result = {
                "mode_name": mode_name,
                "train_name": train_name,
                "active_rx": active_rx,
                "mf_tx": mf_tx,
                "num_rx": num_rx,
                "num_mf_tx": num_tx,
                "num_virtual_channels": num_paths,
                "num_parameters": num_parameters,
                "num_test_samples": int(len(true_bin)),
                "num_target": int(np.sum(true_bin == 1)),
                "num_no_target": int(np.sum(true_bin == 0)),
                "threshold": float(det_threshold),
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
                "accuracy": float(accuracy_score(true_bin, det_test_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(true_bin, det_test_pred)),
                "precision": float(test_precision_bin),
                "recall_Pd": float(test_recall_bin),
                "P_FA": float(test_pfa),
                "F1": float(test_f1_bin),
                "MCC": float(matthews_corrcoef(true_bin, det_test_pred)),
                "X_MAE_all_target": float(pos_mae_xy[0]),
                "Y_MAE_all_target": float(pos_mae_xy[1]),
                "X_RMSE_all_target": float(pos_rmse_xy[0]),
                "Y_RMSE_all_target": float(pos_rmse_xy[1]),
                "Euclidean_MAE_all_target": float(pos_euclidean_mae),
                "Euclidean_RMSE_all_target": float(pos_euclidean_rmse),
                "Euclidean_Median_all_target": safe_nanmedian(pos_err_m[target_mask]),
                "Euclidean_P90_all_target": safe_nanpercentile(pos_err_m[target_mask], 90),
                "SCNR_mean_target": safe_nanmean(scnr_vec[target_mask]),
                "SCNR_median_target": safe_nanmedian(scnr_vec[target_mask]),
                "SCNR_min_target": float(np.nanmin(scnr_vec[target_mask])) if np.any(target_mask) else np.nan,
                "SCNR_max_target": float(np.nanmax(scnr_vec[target_mask])) if np.any(target_mask) else np.nan,
                "sample_id": sample_id,
                "true_bin": true_bin,
                "det_prob": det_test_prob,
                "det_pred": det_test_pred,
                "target_x": target_x,
                "target_y": target_y,
                "pred_x": pos_test_pred[:, 0],
                "pred_y": pos_test_pred[:, 1],
                "pos_err_m": pos_err_m,
                "scnr_vec": scnr_vec,
            }

            # AUC 類指標需要兩類樣本都存在
            try:
                result["AUC_ROC"] = float(roc_auc_score(true_bin, det_test_prob))
            except Exception:
                result["AUC_ROC"] = np.nan

            try:
                result["AUC_PR"] = float(average_precision_score(true_bin, det_test_prob))
            except Exception:
                result["AUC_PR"] = np.nan

            del model
            tf.keras.backend.clear_session()

            return result

        # ---------------- 一次測試四個 model ----------------
        all_results = {}

        for mode_name, cfg in PROCESSING_CONFIG.items():
            all_results[mode_name] = run_one_model_test(
                mode_name=mode_name,
                cfg=cfg,
                shared_test_data=shared_test_data,
            )

        # ---------------- all_model_test_summary.csv ----------------
        summary_rows = []

        for mode_name, r in all_results.items():
            target_mask = r["true_bin"] == 1
            detected_target_mask = (r["true_bin"] == 1) & (r["det_pred"] == 1)

            pos_err_target = r["pos_err_m"][target_mask]
            pos_err_detected = r["pos_err_m"][detected_target_mask]

            row = {
                "model_name": mode_name,
                "train_name": r["train_name"],
                "active_rx": str(r["active_rx"]),
                "mf_tx": str(r["mf_tx"]),
                "num_rx": r["num_rx"],
                "num_mf_tx": r["num_mf_tx"],
                "num_virtual_channels": r["num_virtual_channels"],
                "num_parameters": r["num_parameters"],
                "num_test_samples": r["num_test_samples"],
                "num_target": r["num_target"],
                "num_no_target": r["num_no_target"],
                "threshold": r["threshold"],
                "TN": r["TN"],
                "FP": r["FP"],
                "FN": r["FN"],
                "TP": r["TP"],
                "accuracy": r["accuracy"],
                "balanced_accuracy": r["balanced_accuracy"],
                "precision": r["precision"],
                "recall_Pd": r["recall_Pd"],
                "P_FA": r["P_FA"],
                "F1": r["F1"],
                "MCC": r["MCC"],
                "AUC_ROC": r["AUC_ROC"],
                "AUC_PR": r["AUC_PR"],
                "X_MAE_all_target": r["X_MAE_all_target"],
                "Y_MAE_all_target": r["Y_MAE_all_target"],
                "X_RMSE_all_target": r["X_RMSE_all_target"],
                "Y_RMSE_all_target": r["Y_RMSE_all_target"],
                "Euclidean_MAE_all_target": r["Euclidean_MAE_all_target"],
                "Euclidean_RMSE_all_target": r["Euclidean_RMSE_all_target"],
                "Euclidean_Median_all_target": r["Euclidean_Median_all_target"],
                "Euclidean_P90_all_target": r["Euclidean_P90_all_target"],
                "Euclidean_MAE_detected_target": safe_nanmean(pos_err_detected),
                "Euclidean_RMSE_detected_target": safe_nanrmse(pos_err_detected),
                "SCNR_mean_target": r["SCNR_mean_target"],
                "SCNR_median_target": r["SCNR_median_target"],
                "SCNR_min_target": r["SCNR_min_target"],
                "SCNR_max_target": r["SCNR_max_target"],
            }

            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(out_dir, "all_model_test_summary.csv")
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

        print("\n========== All Model Test Summary ==========")
        print(summary_df)
        print(f"[Info] Summary saved: {summary_path}")

        # ---------------- all_model_per_sample_predictions.csv ----------------
        ref_mode = list(all_results.keys())[0]
        ref = all_results[ref_mode]

        per_sample_df = pd.DataFrame({
            "sample_id": ref["sample_id"],
            "true_label": ref["true_bin"],
            "target_x": ref["target_x"],
            "target_y": ref["target_y"],
            "SCNR_dB": ref["scnr_vec"],
        })

        for mode_name, r in all_results.items():
            per_sample_df[f"{mode_name}_det_prob"] = r["det_prob"]
            per_sample_df[f"{mode_name}_det_pred"] = r["det_pred"]
            per_sample_df[f"{mode_name}_pred_x"] = r["pred_x"]
            per_sample_df[f"{mode_name}_pred_y"] = r["pred_y"]
            per_sample_df[f"{mode_name}_pos_err_m"] = r["pos_err_m"]
            per_sample_df[f"{mode_name}_correct_det"] = (r["det_pred"] == r["true_bin"]).astype(int)

        per_sample_path = os.path.join(out_dir, "all_model_per_sample_predictions.csv")
        per_sample_df.to_csv(per_sample_path, index=False, encoding="utf-8-sig")
        print(f"[Info] Per-sample predictions saved: {per_sample_path}")

        # ---------------- fixed_pfa_summary.csv ----------------
        fixed_pfa_rows = []

        for mode_name, r in all_results.items():
            y_true = r["true_bin"]
            det_prob = r["det_prob"]
            pos_err = r["pos_err_m"]

            no_target_prob = det_prob[y_true == 0]

            for target_pfa in FIXED_PFA_LIST:
                if len(no_target_prob) == 0:
                    selected_th = np.nan
                    y_pred_pfa = np.zeros_like(y_true)
                else:
                    selected_th = np.quantile(no_target_prob, 1.0 - target_pfa)
                    y_pred_pfa = (det_prob >= selected_th).astype(int)

                tn, fp, fn, tp = confusion_matrix(y_true, y_pred_pfa, labels=[0, 1]).ravel()

                detected_target_mask = (y_true == 1) & (y_pred_pfa == 1)
                pos_err_detected = pos_err[detected_target_mask]

                fixed_pfa_rows.append({
                    "model_name": mode_name,
                    "target_PFA": target_pfa,
                    "selected_threshold": selected_th,
                    "actual_PFA": fp / max(fp + tn, 1),
                    "Pd": tp / max(tp + fn, 1),
                    "precision": precision_score(y_true, y_pred_pfa, zero_division=0),
                    "F1": f1_score(y_true, y_pred_pfa, zero_division=0),
                    "TN": int(tn),
                    "FP": int(fp),
                    "FN": int(fn),
                    "TP": int(tp),
                    "Euclidean_MAE_detected_target": safe_nanmean(pos_err_detected),
                    "Euclidean_RMSE_detected_target": safe_nanrmse(pos_err_detected),
                })

        fixed_pfa_df = pd.DataFrame(fixed_pfa_rows)
        fixed_pfa_path = os.path.join(out_dir, "fixed_pfa_summary.csv")
        fixed_pfa_df.to_csv(fixed_pfa_path, index=False, encoding="utf-8-sig")
        print(f"[Info] Fixed-PFA summary saved: {fixed_pfa_path}")

        # ---------------- scnr_binned_model_comparison.csv ----------------
        # 注意：目前 SCNR_dB_vec 對 no-target 通常是 NaN，因此這裡是 target-only Pd/position 分箱。
        scnr_rows = []
        bin_width = 5.0

        for mode_name, r in all_results.items():
            y_true = r["true_bin"]
            y_pred = r["det_pred"]
            scnr = r["scnr_vec"]
            pos_err = r["pos_err_m"]

            valid_target_mask = (y_true == 1) & np.isfinite(scnr)

            if np.sum(valid_target_mask) == 0:
                continue

            scnr_t = scnr[valid_target_mask]
            scnr_min = np.floor(np.nanmin(scnr_t) / bin_width) * bin_width
            scnr_max = np.ceil(np.nanmax(scnr_t) / bin_width) * bin_width

            if scnr_max <= scnr_min:
                scnr_max = scnr_min + bin_width

            bin_edges = np.arange(scnr_min, scnr_max + bin_width, bin_width)

            for bi in range(len(bin_edges) - 1):
                lo = bin_edges[bi]
                hi = bin_edges[bi + 1]

                if bi == len(bin_edges) - 2:
                    bin_mask = valid_target_mask & (scnr >= lo) & (scnr <= hi)
                else:
                    bin_mask = valid_target_mask & (scnr >= lo) & (scnr < hi)

                count_target = int(np.sum(bin_mask))
                if count_target == 0:
                    continue

                y_b = y_true[bin_mask]
                pred_b = y_pred[bin_mask]
                pos_err_b = pos_err[bin_mask]

                pd_bin = np.sum(pred_b == 1) / max(count_target, 1)

                scnr_rows.append({
                    "model_name": mode_name,
                    "scnr_bin_left": lo,
                    "scnr_bin_right": hi,
                    "scnr_median": safe_nanmedian(scnr[bin_mask]),
                    "count_target": count_target,
                    "Pd": pd_bin,
                    "Euclidean_MAE_target": safe_nanmean(pos_err_b),
                    "Euclidean_RMSE_target": safe_nanrmse(pos_err_b),
                    "Euclidean_Median_target": safe_nanmedian(pos_err_b),
                    "Euclidean_P90_target": safe_nanpercentile(pos_err_b, 90),
                })

        scnr_df = pd.DataFrame(scnr_rows)
        scnr_path = os.path.join(out_dir, "scnr_binned_model_comparison.csv")
        scnr_df.to_csv(scnr_path, index=False, encoding="utf-8-sig")
        print(f"[Info] SCNR-binned comparison saved: {scnr_path}")

        # ---------------- model_disagreement_matrix.csv ----------------
        mode_names = list(all_results.keys())
        disagree_mat = np.zeros((len(mode_names), len(mode_names)), dtype=int)

        for i, mi in enumerate(mode_names):
            for j, mj in enumerate(mode_names):
                pred_i = all_results[mi]["det_pred"]
                pred_j = all_results[mj]["det_pred"]
                disagree_mat[i, j] = int(np.sum(pred_i != pred_j))

        disagree_df = pd.DataFrame(disagree_mat, index=mode_names, columns=mode_names)
        disagree_path = os.path.join(out_dir, "model_disagreement_matrix.csv")
        disagree_df.to_csv(disagree_path, encoding="utf-8-sig")
        print(f"[Info] Disagreement matrix saved: {disagree_path}")

        # ---------------- diagnostics: failure samples ----------------
        diag_df = per_sample_df.copy()

        all_wrong_mask = np.ones(len(diag_df), dtype=bool)
        for mode_name in mode_names:
            all_wrong_mask &= (diag_df[f"{mode_name}_correct_det"].values == 0)

        all_wrong_df = diag_df[all_wrong_mask]
        all_wrong_path = os.path.join(diag_dir, "all_models_wrong_samples.csv")
        all_wrong_df.to_csv(all_wrong_path, index=False, encoding="utf-8-sig")

        for mode_name in mode_names:
            only_correct_mask = (diag_df[f"{mode_name}_correct_det"].values == 1)
            for other_name in mode_names:
                if other_name == mode_name:
                    continue
                only_correct_mask &= (diag_df[f"{other_name}_correct_det"].values == 0)

            only_correct_df = diag_df[only_correct_mask]
            only_correct_path = os.path.join(diag_dir, f"only_{mode_name}_correct_samples.csv")
            only_correct_df.to_csv(only_correct_path, index=False, encoding="utf-8-sig")

        print(f"[Info] Diagnostics saved in: {diag_dir}")

        # ---------------- experiment config ----------------
        config_out = {
            "test_rx1_path": TEST_RX1_PATH,
            "test_rx2_path": TEST_RX2_PATH,
            "model_root": MODEL_ROOT,
            "out_dir": out_dir,
            "fixed_threshold": FIXED_THRESHOLD,
            "fixed_pfa_list": FIXED_PFA_LIST,
            "processing_config": PROCESSING_CONFIG,
        }

        config_path = os.path.join(out_dir, "test_experiment_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_out, f, indent=4, ensure_ascii=False)

        print(f"[Info] Experiment config saved: {config_path}")
        print(f"[Info] Test terminal output saved TXT: {log_path}")

    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = _terminal_stdout
        sys.stderr = _terminal_stderr
        _log_file.close()


if __name__ == "__main__":
    main()
