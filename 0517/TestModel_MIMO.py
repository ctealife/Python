# -*- coding: utf-8 -*-
import os
import sys
import time
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.io import loadmat
from keras import Input, Model
from keras.layers import Conv1D, MaxPooling1D, Flatten, BatchNormalization, Dense, ReLU, Dropout
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Lambda
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------- 路徑設定 ----------------
dir_path = r"C:\鈞論文\paper_2026\CNN\training_result\0512\30000 SAMPLE CASE1 成功"
weights_path  = os.path.join(dir_path, "mimo_det_pos_geo_weights.h5")
norm_path  = os.path.join(dir_path, "mimo_norm_params.pkl")

mat_file_rx1_path = os.path.join(dir_path, "testing_data_rx1.mat")
mat_file_rx2_path = os.path.join(dir_path, "testing_data_rx2.mat")

out_dir = os.path.join(dir_path, "TestResult_MIMO")
os.makedirs(out_dir, exist_ok=True)

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
    num_rx,
    num_pulses,
    sequence_length,
    num_features,
    pnSeq1,
    pnSeq2,
    pos_min,
    pos_max,
    tx_pos_all,
    rx_pos_all,
    rbi_min,
    rbi_max,
):
    input_signal = Input(
        shape=(num_rx, num_pulses, sequence_length, num_features),
        name="mimo_feature_input",
    )

    pn1_mf = pnSeq1[::-1].copy().astype("float32")
    pn2_mf = pnSeq2[::-1].copy().astype("float32")

    if len(pn1_mf) != len(pn2_mf):
        raise ValueError("pnSeq1 and pnSeq2 length mismatch.")

    kernel_size = len(pn1_mf)
    init_weight = np.zeros((kernel_size, 2, 4), dtype=np.float32)
    init_weight[:, 0, 0] = pn1_mf
    init_weight[:, 1, 1] = pn1_mf
    init_weight[:, 0, 2] = pn2_mf
    init_weight[:, 1, 3] = pn2_mf

    x = Lambda(
        lambda t: tf.reshape(t, (-1, sequence_length, num_features)),
        name="merge_batch_rx_pulse",
    )(input_signal)

    x = Conv1D(
        filters=4,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer=Constant(init_weight),
        use_bias=False,
        trainable=True,
        name="pn_matched_filter_bank",
    )(x)

    x = Lambda(
        lambda t: tf.reshape(t, (-1, sequence_length, 2, 2)),
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
            (batch_size_dyn, num_rx, num_pulses, sequence_length, 2),
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
        win_len=31,
        guard_len=3,
        rank_ratio=0.85,
        eps=1e-8,
        name="os_cfar_ratio",
    )

    x = Lambda(log_layer, name="log_power")(x)

    def restore_mimo_map(t):
        batch_size_dyn = tf.shape(t)[0] // (2 * num_rx * num_pulses)

        t = tf.reshape(
            t,
            (batch_size_dyn, 2, num_rx, num_pulses, sequence_length, 1),
        )
        t = tf.squeeze(t, axis=-1)
        t = tf.transpose(t, perm=[0, 4, 1, 2, 3])
        t = tf.reshape(t, (batch_size_dyn, sequence_length, 2 * num_rx * num_pulses))
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
    tx_tf = tf.constant(tx_pos_all, dtype=tf.float32)
    rx_tf = tf.constant(rx_pos_all, dtype=tf.float32)
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
        rho = tf.reshape(rho, (-1, 4))
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
        name="mimo_2tx2rx_det_pos_geo",
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

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Cannot find model weights: {weights_path}")

    if not os.path.exists(norm_path):
        raise FileNotFoundError(f"Cannot find MIMO normalization params: {norm_path}")

    if not os.path.exists(mat_file_rx1_path):
        raise FileNotFoundError(f"Cannot find Rx1 MAT file: {mat_file_rx1_path}")

    if not os.path.exists(mat_file_rx2_path):
        raise FileNotFoundError(f"Cannot find Rx2 MAT file: {mat_file_rx2_path}")

    _terminal_stdout = sys.stdout
    _terminal_stderr = sys.stderr
    _log_file = open(log_path, "w", encoding="utf-8-sig", errors="replace")
    sys.stdout = TeeLogger(_terminal_stdout, _log_file)
    sys.stderr = TeeLogger(_terminal_stderr, _log_file)

    try:
        print(f"[Info] Rx1 MAT: {mat_file_rx1_path}", flush=True)
        t0 = time.time()
        data1 = loadmat(mat_file_rx1_path)
        print(f"[Info] Rx1 loaded in {time.time() - t0:.2f} sec", flush=True)

        print(f"[Info] Rx2 MAT: {mat_file_rx2_path}", flush=True)
        t0 = time.time()
        data2 = loadmat(mat_file_rx2_path)
        print(f"[Info] Rx2 loaded in {time.time() - t0:.2f} sec", flush=True)

        require_keys(
            data1,
            [
                "feature_4d",
                "label_bin",
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
            ],
            "Rx1 MAT",
        )
        require_keys(data2, ["feature_4d", "label_bin"], "Rx2 MAT")

        X_rx1 = data1["feature_4d"].astype("float32")
        X_rx2 = data2["feature_4d"].astype("float32")

        label_bin_1 = np.array(data1["label_bin"]).squeeze().astype("float32")
        label_bin_2 = np.array(data2["label_bin"]).squeeze().astype("float32")

        if not np.array_equal(label_bin_1, label_bin_2):
            raise ValueError("Rx1 and Rx2 label_bin mismatch. Sample order may be inconsistent.")
        
        if "sample_id" not in data1 or "sample_id" not in data2:
            raise KeyError("MAT file does not contain sample_id. Please regenerate testing_data_rx1/rx2 with sample_id.")

        sample_id_1 = np.array(data1["sample_id"]).squeeze().astype("int32")
        sample_id_2 = np.array(data2["sample_id"]).squeeze().astype("int32")

        if not np.array_equal(sample_id_1, sample_id_2):
            raise ValueError("Rx1 and Rx2 sample_id mismatch. MIMO samples are not aligned.")

        check_same_keys = [
            "target_x_vec",
            "target_y_vec",
            "SCNR_dB_vec",
            "Rbi_11_vec",
            "Rbi_12_vec",
            "Rbi_21_vec",
            "Rbi_22_vec",
        ]

        for key in check_same_keys:
            if key not in data1 or key not in data2:
                raise KeyError(f"Rx1 or Rx2 MAT file does not contain {key}.")

            v1 = np.array(data1[key]).squeeze()
            v2 = np.array(data2[key]).squeeze()

            if not np.allclose(v1, v2, equal_nan=True):
                raise ValueError(f"Rx1 and Rx2 {key} mismatch. Metadata may be misaligned.")

        if X_rx1.shape != X_rx2.shape:
            raise ValueError(f"Rx1/Rx2 feature_4d shape mismatch: {X_rx1.shape} vs {X_rx2.shape}")

        X_test = np.stack([X_rx1, X_rx2], axis=1).astype("float32")
        y_det = label_bin_1.astype("float32")
        true_bin = y_det.squeeze().astype(int)
        target_mask = true_bin == 1

        if X_test.ndim != 5:
            raise ValueError(f"X_test must be 5D (N, Rx, P, L, C), got {X_test.shape}")

        num_rx = X_test.shape[1]
        num_pulses = X_test.shape[2]
        sequence_length = X_test.shape[3]
        num_features = X_test.shape[4]

        if num_rx != 2:
            raise ValueError(f"num_rx must be 2, got {num_rx}")

        if num_features != 2:
            raise ValueError(f"num_features must be 2 (I/Q), got {num_features}")

        scnr_vec = np.array(data1["SCNR_dB_vec"]).squeeze().astype("float32")
        target_x = np.array(data1["target_x_vec"]).squeeze().astype("float32")
        target_y = np.array(data1["target_y_vec"]).squeeze().astype("float32")
        pos_raw = np.stack([target_x, target_y], axis=1).astype("float32")

        rbi_raw = np.stack(
            [
                np.array(data1["Rbi_11_vec"]).squeeze(),
                np.array(data1["Rbi_12_vec"]).squeeze(),
                np.array(data1["Rbi_21_vec"]).squeeze(),
                np.array(data1["Rbi_22_vec"]).squeeze(),
            ],
            axis=1,
        ).astype("float32")

        tx_pos_all = np.array(data1["tx_pos_all"]).astype("float32")
        rx_pos_all = np.array(data1["rx_pos_all"]).astype("float32")
        pnSeq1 = np.array(data1["pnSeq1"]).squeeze().astype("float32")
        pnSeq2 = np.array(data1["pnSeq2"]).squeeze().astype("float32")

        with open(norm_path, "rb") as f:
            norm_params = pickle.load(f)

        pos_min = np.asarray(norm_params["pos_min"], dtype="float32")
        pos_max = np.asarray(norm_params["pos_max"], dtype="float32")
        rbi_min = float(norm_params["rbi_min"])
        rbi_max = float(norm_params["rbi_max"])

        # Reuse geometry from the MAT file so the model build matches the tested data.
        # Fall back to saved params if a future MAT file omits geometry fields.
        tx_pos_all = np.asarray(norm_params.get("tx_pos_all", tx_pos_all), dtype="float32")
        rx_pos_all = np.asarray(norm_params.get("rx_pos_all", rx_pos_all), dtype="float32")

        print(f"[Info] X MIMO shape: {X_test.shape}")
        print(f"[Info] y_det shape: {y_det.shape}")
        print(f"[Info] target samples: {np.sum(true_bin == 1)}, no-target samples: {np.sum(true_bin == 0)}")
        print(f"[Info] SCNR range: {np.nanmin(scnr_vec):.2f} ~ {np.nanmax(scnr_vec):.2f} dB")
        print(f"[Info] pos_min={pos_min}, pos_max={pos_max}")
        print(f"[Info] rbi_min={rbi_min:.4f}, rbi_max={rbi_max:.4f}")

        model = build_mimo_model(
            num_rx=num_rx,
            num_pulses=num_pulses,
            sequence_length=sequence_length,
            num_features=num_features,
            pnSeq1=pnSeq1,
            pnSeq2=pnSeq2,
            pos_min=pos_min,
            pos_max=pos_max,
            tx_pos_all=tx_pos_all,
            rx_pos_all=rx_pos_all,
            rbi_min=rbi_min,
            rbi_max=rbi_max,
        )

        model.load_weights(weights_path)
        print(f"[Info] Model weights loaded: {weights_path}")
        print(f"[Info] MIMO normalization params loaded: {norm_path}")

        print("\n========== Test Predict ==========")
        pred_test = model.predict(X_test, verbose=0)

        if isinstance(pred_test, dict):
            det_test_prob = pred_test["det_output"].squeeze()
            pos_test_pred_norm = pred_test["pos_output"]
            geo_test_pred_norm = pred_test["geo_output"]
        else:
            det_test_prob = pred_test[0].squeeze()
            pos_test_pred_norm = pred_test[1]
            geo_test_pred_norm = pred_test[2]

        det_threshold = 0.5
        det_test_pred = (det_test_prob >= det_threshold).astype(int)

        pos_test_pred_norm = np.clip(pos_test_pred_norm, 0.0, 1.0)
        pos_test_pred = pos_test_pred_norm * (pos_max - pos_min) + pos_min

        wrong_idx = np.where(det_test_pred != true_bin)[0]

        print("\n========== Detection Error Samples ==========")
        print(f"[Test] error samples: {len(wrong_idx)}")

        for k in wrong_idx[:20]:
            print(
                f"test_index={k}, "
                f"true_det={int(true_bin[k])}, "
                f"pred_det={det_test_pred[k]}, "
                f"prob={det_test_prob[k]:.4f}, "
                f"SCNR={scnr_vec[k]:.2f} dB"
            )

        test_precision_bin = precision_score(true_bin, det_test_pred, zero_division=0)
        test_recall_bin = recall_score(true_bin, det_test_pred, zero_division=0)
        test_f1_bin = f1_score(true_bin, det_test_pred, zero_division=0)
        test_cm_bin = confusion_matrix(true_bin, det_test_pred, labels=[0, 1])

        threshold_sweep_report(
            y_true=true_bin,
            det_prob=det_test_prob,
            thresholds=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        )

        print("\n========== Detection Metrics ==========")
        print(f"[Test] threshold: {det_threshold:.2f}")
        print(
            f"[Test] precision: {test_precision_bin:.4f}, "
            f"recall/Pd: {test_recall_bin:.4f}, "
            f"F1: {test_f1_bin:.4f}"
        )
        print(test_cm_bin)
        
        # ---------------- 儲存 Confusion Matrix 圖 ----------------
        fig, ax = plt.subplots(figsize=(5, 4))

        im = ax.imshow(test_cm_bin, cmap="Blues")

        ax.set_title("MIMO Detection Confusion Matrix")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["No Target", "Target"])
        ax.set_yticklabels(["No Target", "Target"])

        threshold = test_cm_bin.max() / 2

        for i in range(test_cm_bin.shape[0]):
            for j in range(test_cm_bin.shape[1]):
                text_color = "white" if test_cm_bin[i, j] > threshold else "black"

                ax.text(
                    j,
                    i,
                    str(test_cm_bin[i, j]),
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=16
                )

        fig.colorbar(im, ax=ax)
        plt.tight_layout()

        cm_fig_path = os.path.join(out_dir, "test_confusion_matrix.png")
        plt.savefig(cm_fig_path, dpi=200)
        plt.close()

        print(f"[Info] Confusion matrix figure saved: {cm_fig_path}")

        if np.sum(target_mask) > 0:
            pos_test_err = pos_test_pred[target_mask] - pos_raw[target_mask]
            pos_mae_xy = np.mean(np.abs(pos_test_err), axis=0)
            pos_rmse_xy = np.sqrt(np.mean(pos_test_err ** 2, axis=0))
            pos_euclidean_mae = np.mean(np.linalg.norm(pos_test_err, axis=1))

            print(f"[Test] X MAE: {pos_mae_xy[0]:.4f} m")
            print(f"[Test] Y MAE: {pos_mae_xy[1]:.4f} m")
            print(f"[Test] X RMSE: {pos_rmse_xy[0]:.4f} m")
            print(f"[Test] Y RMSE: {pos_rmse_xy[1]:.4f} m")
            print(f"[Test] Position Euclidean MAE: {pos_euclidean_mae:.4f} m")
        else:
            print("[Test] No target samples. Skip position regression metrics.")

        scnr_detection_accuracy_report_and_plot(
            scnr_db=scnr_vec,
            y_true=true_bin,
            y_pred=det_test_pred,
            out_dir=out_dir,
            tag="Test",
            bin_width=5.0,
        )

        pred_path = os.path.join(out_dir, "test_mimo_predictions.npz")
        np.savez(
            pred_path,
            det_prob=det_test_prob,
            det_pred=det_test_pred,
            true_bin=true_bin,
            pos_pred=pos_test_pred,
            pos_true=pos_raw,
            geo_pred_norm=geo_test_pred_norm,
            rbi_true=rbi_raw,
            scnr_db=scnr_vec,
            confusion_matrix=test_cm_bin,
        )
        print(f"[Info] Test predictions saved: {pred_path}")
        print(f"[Info] Test terminal output saved TXT: {log_path}")

    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = _terminal_stdout
        sys.stderr = _terminal_stderr
        _log_file.close()


if __name__ == "__main__":
    main()
