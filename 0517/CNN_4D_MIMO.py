# -*- coding: utf-8 -*-
import os
import sys
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from collections import Counter
from keras import Input, Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.initializers import Constant
from keras.layers import Conv1D, MaxPooling1D, Flatten, BatchNormalization, Dense, ReLU, Dropout
from tensorflow.keras.layers import Lambda
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
# import hdf5storage
import h5py
import gc
import shutil


sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------- 路徑與終端 LOG ----------------

dir_path = r"C:\鈞論文\paper_2026\CNN\training_result\0516"

mat_file_rx1_path = os.path.join(dir_path, "training_data_rx1.mat")
mat_file_rx2_path = os.path.join(dir_path, "training_data_rx2.mat")

model_dir = os.path.join(dir_path, "Training_Result_MIMO")
os.makedirs(model_dir, exist_ok=True)

# ---------------- 參數 ----------------
batch_size    = 32
num_epochs    = 50
learning_rate = 1e-4

def read_mat73_dataset(h5file, key, dtype=None):
    """
    讀取 MATLAB v7.3 MAT 檔中的單一 numeric dataset。

    MATLAB v7.3 使用 HDF5 儲存，直接用 h5py 讀取時維度順序通常會反過來。
    因此對 ndim >= 2 的陣列做 np.transpose，使其回到 MATLAB 原本的維度順序。
    例如 MATLAB: (N, P, L, 2)
    h5py 讀到:   (2, L, P, N)
    transpose 後: (N, P, L, 2)
    """
    if key not in h5file:
        raise KeyError(f"MAT v7.3 檔中找不到變數: {key}")

    obj = h5file[key]

    if not isinstance(obj, h5py.Dataset):
        raise TypeError(f"{key} 不是 numeric dataset；目前 loader 只處理 numeric array。")

    arr = obj[()]

    if arr.ndim >= 2:
        arr = np.transpose(arr)

    arr = np.squeeze(arr)

    if dtype is not None:
        arr = arr.astype(dtype, copy=False)

    return arr


def load_mat73_selected(mat_path, required_keys, optional_keys=None, verbose=False):
    """
    只讀取指定變數，避免 hdf5storage.loadmat 一次載入整個 MAT 檔。
    """
    if optional_keys is None:
        optional_keys = []

    data = {}

    file_size_gb = os.path.getsize(mat_path) / (1024 ** 3)
    if verbose:
        print(f"[Info] 開啟 MAT v7.3: {mat_path}", flush=True)
        print(f"[Info] 檔案大小: {file_size_gb:.2f} GB", flush=True)

    with h5py.File(mat_path, "r") as f:
        for key in required_keys:
            t0 = time.time()
            data[key] = read_mat73_dataset(f, key)

            arr = data[key]
            size_mb = arr.nbytes / (1024 ** 2) if hasattr(arr, "nbytes") else 0.0

            if verbose:
                print(
                    f"[Load] {key:24s} shape={arr.shape!s:18s} "
                    f"dtype={arr.dtype} size={size_mb:8.2f} MB "
                    f"time={time.time() - t0:6.2f}s",
                    flush=True
                )

        for key in optional_keys:
            if key not in f:
                if verbose:
                    print(f"[Load] optional key not found: {key}", flush=True)
                continue

            t0 = time.time()
            data[key] = read_mat73_dataset(f, key)

            arr = data[key]
            size_mb = arr.nbytes / (1024 ** 2) if hasattr(arr, "nbytes") else 0.0

            if verbose:
                print(
                    f"[Load] {key:24s} shape={arr.shape!s:18s} "
                    f"dtype={arr.dtype} size={size_mb:8.2f} MB "
                    f"time={time.time() - t0:6.2f}s",
                    flush=True
                )

    return data

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


log_path = os.path.join(
    model_dir,
    f"{os.path.splitext(os.path.basename(__file__))[0]}_terminal_log.txt"
)
_log_file = open(log_path, "w", encoding="utf-8-sig", errors="replace")
_original_stdout = sys.stdout
_original_stderr = sys.stderr
sys.stdout = TeeLogger(_original_stdout, _log_file)
sys.stderr = TeeLogger(_original_stderr, _log_file)

# ---------------- GPU 設定 ----------------
print("\n========== GPU SET... ==========")
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

if not os.path.exists(mat_file_rx1_path):
    raise FileNotFoundError(f"找不到 Rx1 MAT 檔案: {mat_file_rx1_path}")

if not os.path.exists(mat_file_rx2_path):
    raise FileNotFoundError(f"找不到 Rx2 MAT 檔案: {mat_file_rx2_path}")

same_metadata_keys = [
    "target_x_vec",
    "target_y_vec",
    "SCNR_dB_vec",
    "snr_vec",
    "rcs_vec",
    "noise_rcs_vec",
    "Rbi_11_vec",
    "Rbi_12_vec",
    "Rbi_21_vec",
    "Rbi_22_vec",
    "label_range_cls"
]

rx1_required_keys = [
    "feature_4d",
    "label_bin",
    "sample_id",
    "target_x_vec",
    "target_y_vec",
    "SCNR_dB_vec",
    "snr_vec",
    "rcs_vec",
    "noise_rcs_vec",
    "Rbi_11_vec",
    "Rbi_12_vec",
    "Rbi_21_vec",
    "Rbi_22_vec",
    "label_range_cls",
    "tx_pos_all",
    "rx_pos_all",
    "pnSeq1",
    "pnSeq2"
]

rx1_optional_keys = [
    "SCNR_nominal_dB_vec"
]

rx2_required_keys = [
    "feature_4d",
    "label_bin",
    "sample_id"
] + same_metadata_keys

rx2_optional_keys = [
    "SCNR_nominal_dB_vec"
]

print(f"[Info] 載入 Rx1 MAT 檔案: {mat_file_rx1_path}", flush=True)
t0 = time.time()
data1 = load_mat73_selected(
    mat_file_rx1_path,
    required_keys=rx1_required_keys,
    optional_keys=rx1_optional_keys,
    verbose=False
)
print(f"[Info] Rx1 載入完成，耗時 {time.time() - t0:.2f} 秒", flush=True)

print(f"[Info] 載入 Rx2 MAT 檔案: {mat_file_rx2_path}", flush=True)
t0 = time.time()
data2 = load_mat73_selected(
    mat_file_rx2_path,
    required_keys=rx2_required_keys,
    optional_keys=rx2_optional_keys,
    verbose=False
)
print(f"[Info] Rx2 載入完成，耗時 {time.time() - t0:.2f} 秒", flush=True)

X_rx1 = data1.pop("feature_4d").astype("float32", copy=False)   # (N, P, L, 2)
X_rx2 = data2.pop("feature_4d").astype("float32", copy=False)   # (N, P, L, 2)

label_bin_1 = np.array(data1["label_bin"]).squeeze().astype("float32")
label_bin_2 = np.array(data2["label_bin"]).squeeze().astype("float32")

if not np.array_equal(label_bin_1, label_bin_2):
    raise ValueError("Rx1 與 Rx2 的 label_bin 不一致，代表兩個 MAT 檔樣本順序可能錯位。")

if "sample_id" not in data1 or "sample_id" not in data2:
    raise KeyError("MAT 檔中找不到 sample_id。請在 Simulation_Clutter_v15.m 中對 Rx1/Rx2 同時儲存 sample_id。")

sample_id_1 = np.array(data1["sample_id"]).squeeze().astype("int32")
sample_id_2 = np.array(data2["sample_id"]).squeeze().astype("int32")

if not np.array_equal(sample_id_1, sample_id_2):
    raise ValueError("Rx1 與 Rx2 的 sample_id 不一致，代表兩個 MAT 檔樣本順序錯位。")

for key in same_metadata_keys:
    if key not in data1 or key not in data2:
        raise KeyError(f"Rx1 或 Rx2 MAT 檔中找不到 {key}。")

    v1 = np.array(data1[key]).squeeze()
    v2 = np.array(data2[key]).squeeze()

    same = np.allclose(v1, v2, equal_nan=True)

    if not same:
        raise ValueError(f"Rx1 與 Rx2 的 {key} 不一致，代表樣本 metadata 錯位或生成流程有問題。")
    if key not in data1 or key not in data2:
        raise KeyError(f"Rx1 或 Rx2 MAT 檔中找不到 {key}。")

    v1 = np.array(data1[key]).squeeze()
    v2 = np.array(data2[key]).squeeze()

    same = np.allclose(v1, v2, equal_nan=True)

    if not same:
        raise ValueError(f"Rx1 與 Rx2 的 {key} 不一致，代表樣本 metadata 錯位或生成流程有問題。")

if X_rx1.shape != X_rx2.shape:
    raise ValueError(f"Rx1/Rx2 feature_4d shape 不一致: {X_rx1.shape} vs {X_rx2.shape}")

# MIMO input: (N, Rx, P, L, I/Q)
X = np.empty(
    (X_rx1.shape[0], 2, X_rx1.shape[1], X_rx1.shape[2], X_rx1.shape[3]),
    dtype=np.float32
)

X[:, 0, :, :, :] = X_rx1
X[:, 1, :, :, :] = X_rx2

del X_rx1, X_rx2
gc.collect()

# detection label
y_det = label_bin_1.astype("float32")
target_mask_all = (y_det == 1)

if np.sum(target_mask_all) == 0:
    raise ValueError("資料中沒有任何有目標樣本，無法建立 position regression label。")

# ---------------- Metadata 與 raw labels ----------------
# 注意：這裡只讀 raw label，不在此處做 normalization。
# normalization 必須等 train / validation split 之後，只用 training set 計算。

# Global SCNR metadata
# 注意：
# 這裡的 SCNR_dB_vec 必須來自 MATLAB 端的 global clutter-based SCNR，
# 也就是「控制樣本整體雜波背景下的目標強度」，
# 不再代表 target 所在 range bin 附近的 local SCNR。
if "SCNR_dB_vec" not in data1:
    raise KeyError("MAT 檔中找不到 SCNR_dB_vec。請確認 MATLAB 端已儲存 global clutter-based SCNR。")

scnr_global_vec = np.array(data1["SCNR_dB_vec"]).squeeze().astype("float32")

# Nominal SCNR metadata：MATLAB 端指定的 SCNR 排程值，例如 -80:10:10
if "SCNR_nominal_dB_vec" in data1:
    scnr_nominal_vec = np.array(data1["SCNR_nominal_dB_vec"]).squeeze().astype("float32")
else:
    scnr_nominal_vec = np.full_like(scnr_global_vec, np.nan, dtype="float32")

# SNR metadata
if "snr_vec" not in data1:
    raise KeyError("MAT 檔中找不到 snr_vec。")

snr_vec = np.array(data1["snr_vec"]).squeeze().astype("float32")

# RCS metadata
if "rcs_vec" not in data1:
    raise KeyError("MAT 檔中找不到 rcs_vec。")

rcs_vec = np.array(data1["rcs_vec"]).squeeze().astype("float32")

# no-target noise RCS metadata
if "noise_rcs_vec" in data1:
    noise_rcs_vec = np.array(data1["noise_rcs_vec"]).squeeze().astype("float32")
else:
    noise_rcs_vec = np.full_like(rcs_vec, np.nan, dtype="float32")

# range class metadata
if "label_range_cls" in data1:
    range_cls = np.array(data1["label_range_cls"]).squeeze().astype("int32")
else:
    range_cls = np.zeros_like(y_det, dtype="int32")

# position raw label: true target position in common 2D coordinate
target_x = np.array(data1["target_x_vec"]).squeeze().astype("float32")
target_y = np.array(data1["target_y_vec"]).squeeze().astype("float32")
pos_raw = np.stack([target_x, target_y], axis=1).astype("float32")  # (N, 2)

# position loss mask：只有有目標樣本計算 position loss
pos_mask = target_mask_all.astype("float32")

# 四條 Tx-Rx bistatic path length
# 順序固定為 Tx1Rx1, Tx1Rx2, Tx2Rx1, Tx2Rx2
rbi_raw = np.stack([
    np.array(data1["Rbi_11_vec"]).squeeze(),
    np.array(data1["Rbi_12_vec"]).squeeze(),
    np.array(data1["Rbi_21_vec"]).squeeze(),
    np.array(data1["Rbi_22_vec"]).squeeze()
], axis=1).astype("float32")

# radar geometry
tx_pos_all = np.array(data1["tx_pos_all"]).astype("float32")  # (2, 2)
rx_pos_all = np.array(data1["rx_pos_all"]).astype("float32")  # (2, 2)

# PN codes
pnSeq1 = np.array(data1["pnSeq1"]).squeeze().astype("float32")
pnSeq2 = np.array(data1["pnSeq2"]).squeeze().astype("float32")

# ---------------- PN / Ovsamp / CFAR window check ----------------
print("\n========== PN / Matched Filter / CFAR Check ==========")
print(f"len(pnSeq1) = {len(pnSeq1)}")
print(f"len(pnSeq2) = {len(pnSeq2)}")

if len(pnSeq1) != len(pnSeq2):
    raise ValueError("pnSeq1 與 pnSeq2 長度不一致。")

if len(pnSeq1) % 31 != 0:
    raise ValueError("pnSeq 長度不是 31 的整數倍，請檢查 MATLAB 端 oversampling。")

Ovsamp_est = len(pnSeq1) // 31

# OS-CFAR-like window 依 oversampling 放大
cfar_win_len = 31 * Ovsamp_est

# CFAR window 建議使用奇數長度，確保中心 CUT 存在
if cfar_win_len % 2 == 0:
    cfar_win_len += 1

cfar_guard_len = 3 * Ovsamp_est

print(f"Estimated Ovsamp from PN length = {Ovsamp_est}")
print(f"CFAR win_len = {cfar_win_len}")
print(f"CFAR guard_len = {cfar_guard_len}")


if X.ndim != 5:
    raise ValueError(f"X 維度不符合預期，應為 5D (N, Rx, P, L, C)，但得到 {X.shape}")

num_rx          = X.shape[1]
num_pulses      = X.shape[2]
sequence_length = X.shape[3]
num_features    = X.shape[4]

if num_rx != 2:
    raise ValueError(f"num_rx 應為 2，但得到 {num_rx}")

if num_features != 2:
    raise ValueError(f"num_features 應為 2 (I/Q)，但得到 {num_features}")

print(f"[Info] X MIMO shape: {X.shape}")  # (N, 2, P, L, 2)
print(f"[Info] y_det shape: {y_det.shape}")
print(f"[Info] pos_raw shape: {pos_raw.shape}")
print(f"[Info] rbi_raw shape: {rbi_raw.shape}")
print(f"[Info] target samples: {np.sum(y_det == 1)}, no-target samples: {np.sum(y_det == 0)}")
print(
    f"[Info] Global clutter-based SCNR range: "
    f"{np.nanmin(scnr_global_vec):.2f} ~ {np.nanmax(scnr_global_vec):.2f} dB"
)
print("[Info] SCNR definition: target power controlled by whole-sample Rx1 clutter average, not local target-bin clutter.")


def os_cfar_ratio_layer(x, win_len=31, guard_len=3, rank_ratio=0.75, eps=1e-8, name="os_cfar_ratio"):
    """
    近似 OS-CFAR ratio layer
    x shape: (batch, sample, 1)

    輸出:
        y = x / (Z_os + eps)

    參數:
        win_len     : 參考窗總長度 (不含中心 CUT 是否排除由 guard_len 決定)
        guard_len   : CUT 左右各排除幾個 guard cells
        rank_ratio  : 排序後取的位置比例，0.75 表示取偏大的 ordered statistic
        eps         : 避免除零
    """
    def _layer(t):
        # t: (B, L, 1)
        t = tf.squeeze(t, axis=-1)   # (B, L)

        pad = win_len // 2
        t_pad = tf.pad(t, [[0, 0], [pad, pad]], mode="REFLECT")   # (B, L+2pad)

        # sliding window: (B, L, win_len)
        frames = tf.signal.frame(t_pad, frame_length=win_len, frame_step=1, axis=1)

        # 建立 mask，排除中心 CUT 與 guard cells
        center = win_len // 2
        idx = tf.range(win_len)
        valid_mask = tf.logical_or(idx < (center - guard_len), idx > (center + guard_len))
        valid_mask = tf.cast(valid_mask, t.dtype)   # (win_len,)

        # 把 guard/CUT 位置設成很大的值，排序時丟到最後
        large_val = tf.constant(1e9, dtype=t.dtype)
        masked_frames = frames + (1.0 - valid_mask)[tf.newaxis, tf.newaxis, :] * large_val

        # 升冪排序
        sorted_vals = tf.sort(masked_frames, axis=-1, direction='ASCENDING')

        # 真正有效的參考 cell 數
        ref_len = win_len - (2 * guard_len + 1)
        k = tf.cast(tf.round(rank_ratio * tf.cast(ref_len - 1, tf.float32)), tf.int32)
        k = tf.clip_by_value(k, 0, ref_len - 1)

        # 取 ordered statistic
        z_os = sorted_vals[:, :, k]   # (B, L)

        # ratio output
        y = t / (z_os + eps)

        return tf.expand_dims(y, axis=-1)   # (B, L, 1)

    return Lambda(_layer, name=name)(x)


def power_layer(x):
    return tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)

def log_layer(x):
    return tf.math.log(x + 1e-6)

class OneLineBatchProgress(tf.keras.callbacks.Callback):
    def __init__(self, bar_width=28, stream=None):
        super().__init__()
        self.bar_width = bar_width
        self.stream = stream if stream is not None else sys.__stdout__

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch + 1
        self.epochs = self.params.get("epochs", 0)
        self.steps = self.params.get("steps", 0)
        self.t0 = time.time()

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}

        step = batch + 1
        total = self.steps if self.steps else step

        frac = step / total
        filled = int(self.bar_width * frac)
        bar = "=" * filled + ">" + "." * max(self.bar_width - filled - 1, 0)

        elapsed = time.time() - self.t0
        eta = elapsed / step * (total - step) if step > 0 else 0.0

        loss = logs.get("loss", np.nan)
        det_acc = logs.get("det_output_accuracy", np.nan)
        pos_mae = logs.get("pos_output_mae", np.nan)
        geo_mae = logs.get("geo_output_mae", np.nan)

        msg = (
            f"Epoch {self.epoch}/{self.epochs} "
            f"[{bar}] {step}/{total} "
            f"- ETA: {eta:5.1f}s "
            f"- loss: {loss:.4f} "
            f"- det_acc: {det_acc:.4f} "
            f"- pos_mae: {pos_mae:.4f} "
            f"- geo_mae: {geo_mae:.4f}"
        )

        # 避免 terminal 寬度不足導致換行
        cols = shutil.get_terminal_size((140, 20)).columns
        msg = msg[:max(cols - 2, 20)]

        self.stream.write("\r" + msg.ljust(cols - 1))
        self.stream.flush()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        loss = logs.get("loss", np.nan)
        val_loss = logs.get("val_loss", np.nan)

        det_acc = logs.get("det_output_accuracy", np.nan)
        val_det_acc = logs.get("val_det_output_accuracy", np.nan)

        pos_mae = logs.get("pos_output_mae", np.nan)
        val_pos_mae = logs.get("val_pos_output_mae", np.nan)

        geo_mae = logs.get("geo_output_mae", np.nan)
        val_geo_mae = logs.get("val_geo_output_mae", np.nan)

        msg = (
            f"Epoch {self.epoch}/{self.epochs} done "
            f"- loss: {loss:.4f} "
            f"- val_loss: {val_loss:.4f} "
            f"- det_acc: {det_acc:.4f} "
            f"- val_det_acc: {val_det_acc:.4f} "
            f"- pos_mae: {pos_mae:.4f} "
            f"- val_pos_mae: {val_pos_mae:.4f} "
            f"- geo_mae: {geo_mae:.4f} "
            f"- val_geo_mae: {val_geo_mae:.4f}"
        )

        cols = shutil.get_terminal_size((180, 20)).columns
        msg = msg[:max(cols - 2, 20)]

        self.stream.write("\r" + msg.ljust(cols - 1) + "\n")
        self.stream.flush()


# ---------------- 建立 composite stratification key ----------------
# 目的：validation set 不只依照 label_bin 分層，
# 而是同時考慮 target/no-target、SCNR、SNR、RCS、range class。

def build_stratify_key(y_det, scnr_global_vec, snr_vec, rcs_vec, noise_rcs_vec, range_cls):
    keys = []

    for i in range(len(y_det)):
        yi = int(y_det[i])

        if yi == 1:
            # target samples：依 global clutter-based SCNR、SNR、RCS、range class 粗分層
            if np.isfinite(scnr_global_vec[i]):
                scnr_bin = int(np.floor(scnr_global_vec[i] / 5.0))   # 每 5 dB 一格
            else:
                scnr_bin = -999

            if np.isfinite(snr_vec[i]):
                snr_bin = int(np.round(snr_vec[i]))
            else:
                snr_bin = -999

            if np.isfinite(rcs_vec[i]):
                rcs_db = 10.0 * np.log10(rcs_vec[i] + 1e-12)
                rcs_bin = int(np.round(rcs_db))
            else:
                rcs_bin = -999

            r_bin = int(range_cls[i])

            keys.append(f"T_scnr{scnr_bin}_snr{snr_bin}_rcs{rcs_bin}_r{r_bin}")

        else:
            # no-target samples：沒有真實 SCNR 與 target range，
            # 只依 SNR 與 noise reference RCS 粗分層。
            if np.isfinite(snr_vec[i]):
                snr_bin = int(np.round(snr_vec[i]))
            else:
                snr_bin = -999

            if np.isfinite(noise_rcs_vec[i]):
                noise_rcs_db = 10.0 * np.log10(noise_rcs_vec[i] + 1e-12)
                noise_rcs_bin = int(np.round(noise_rcs_db))
            else:
                noise_rcs_bin = -999

            keys.append(f"N_snr{snr_bin}_nrcs{noise_rcs_bin}")

    return np.array(keys)


def collapse_rare_stratify_groups(raw_key, y_det, min_count_per_group=10):
    """
    將樣本數太少的分層類別合併，避免 train_test_split(stratify=...) 失敗。
    """
    counts = Counter(raw_key)

    safe_key = np.array([
        k if counts[k] >= min_count_per_group else f"{int(y_det[i])}_RARE"
        for i, k in enumerate(raw_key)
    ])

    return safe_key


raw_stratify_key = build_stratify_key(
    y_det=y_det,
    scnr_global_vec=scnr_global_vec,
    snr_vec=snr_vec,
    rcs_vec=rcs_vec,
    noise_rcs_vec=noise_rcs_vec,
    range_cls=range_cls
)

stratify_key = collapse_rare_stratify_groups(
    raw_key=raw_stratify_key,
    y_det=y_det,
    min_count_per_group=10
)

# 檢查 stratify_key 是否過細。
# 若類別數太多或仍有類別樣本數小於 2，退回較粗的分層。
n_val = int(np.ceil(len(y_det) * 0.1))
key_counts = Counter(stratify_key)

if (len(np.unique(stratify_key)) > n_val) or (min(key_counts.values()) < 2):
    print("[Warning] composite stratification 類別過細，改用 label + coarse SCNR stratification。")

    coarse_key = []
    for i in range(len(y_det)):
        if int(y_det[i]) == 1 and np.isfinite(scnr_global_vec[i]):
            scnr_bin = int(np.floor(scnr_global_vec[i] / 10.0))  # 每 10 dB 一格
            coarse_key.append(f"T_globalSCNR{scnr_bin}")
        else:
            coarse_key.append("N")

    stratify_key = np.array(coarse_key)

    key_counts = Counter(stratify_key)
    if min(key_counts.values()) < 2:
        print("[Warning] coarse stratification 仍有稀有類別，退回只用 label_bin stratification。")
        stratify_key = y_det.astype(int)


# ---------------- 切 train / val index ----------------
idx = np.arange(len(y_det))

idx_train, idx_val = train_test_split(
    idx,
    test_size=0.1,
    stratify=stratify_key,
    random_state=13
)


# ---------------- train-only normalization ----------------
# 僅使用 training set 中的 target samples 計算 normalization parameters。
# 這是為了避免 validation set 資訊洩漏到訓練流程。

train_target_mask_for_norm = (y_det[idx_train] == 1)

if np.sum(train_target_mask_for_norm) == 0:
    raise ValueError("Training set 中沒有 target samples，無法計算 position / geometry normalization。")

pos_min = np.nanmin(pos_raw[idx_train][train_target_mask_for_norm], axis=0).astype("float32")
pos_max = np.nanmax(pos_raw[idx_train][train_target_mask_for_norm], axis=0).astype("float32")

rbi_min = np.nanmin(rbi_raw[idx_train][train_target_mask_for_norm]).astype("float32")
rbi_max = np.nanmax(rbi_raw[idx_train][train_target_mask_for_norm]).astype("float32")


# ---------------- 建立 normalized labels ----------------
y_pos = np.zeros_like(pos_raw, dtype="float32")
y_geo = np.zeros_like(rbi_raw, dtype="float32")

y_pos[target_mask_all] = (
    (pos_raw[target_mask_all] - pos_min) /
    (pos_max - pos_min + 1e-8)
)

y_geo[target_mask_all] = (
    (rbi_raw[target_mask_all] - rbi_min) /
    (rbi_max - rbi_min + 1e-8)
)


# ---------------- validation normalization range check ----------------
val_target_mask_for_check = (y_det[idx_val] == 1)

if np.sum(val_target_mask_for_check) > 0:
    val_pos_norm_check = (
        (pos_raw[idx_val][val_target_mask_for_check] - pos_min) /
        (pos_max - pos_min + 1e-8)
    )

    print("[Check] val normalized position min:", np.nanmin(val_pos_norm_check, axis=0))
    print("[Check] val normalized position max:", np.nanmax(val_pos_norm_check, axis=0))

    if (np.nanmin(val_pos_norm_check) < -0.05) or (np.nanmax(val_pos_norm_check) > 1.05):
        print("[Warning] validation position label 明顯超出 training normalization range。")
        print("[Warning] 若此情況頻繁發生，建議改用固定物理座標邊界，而非 train-only min/max。")


# ---------------- 依 index 取出 train / val arrays ----------------
X_train = X[idx_train]
X_val   = X[idx_val]

det_train = y_det[idx_train].reshape(-1, 1)
det_val   = y_det[idx_val].reshape(-1, 1)

pos_train = y_pos[idx_train]
pos_val   = y_pos[idx_val]

geo_train = y_geo[idx_train]
geo_val   = y_geo[idx_val]

mask_train = pos_mask[idx_train].astype("float32")
mask_val   = pos_mask[idx_val].astype("float32")

pos_raw_train = pos_raw[idx_train]
pos_raw_val   = pos_raw[idx_val]

rbi_raw_train = rbi_raw[idx_train]
rbi_raw_val   = rbi_raw[idx_val]

scnr_global_train = scnr_global_vec[idx_train]
scnr_global_val   = scnr_global_vec[idx_val]

scnr_nominal_train = scnr_nominal_vec[idx_train]
scnr_nominal_val   = scnr_nominal_vec[idx_val]


# ---------------- split quality report ----------------
def report_split_distribution(tag, idx_set):
    y = y_det[idx_set].astype(int)

    print(f"\n========== {tag} Split Distribution ==========")
    print(f"total samples     : {len(idx_set)}")
    print(f"target samples    : {np.sum(y == 1)}")
    print(f"no-target samples : {np.sum(y == 0)}")

    target_idx = idx_set[y == 1]

    if len(target_idx) > 0:
        print("\n[Target Global Clutter-based SCNR]")
        print("definition: target power controlled by whole-sample Rx1 clutter average")
        print(f"min  = {np.nanmin(scnr_global_vec[target_idx]):.2f} dB")
        print(f"mean = {np.nanmean(scnr_global_vec[target_idx]):.2f} dB")
        print(f"max  = {np.nanmax(scnr_global_vec[target_idx]):.2f} dB")
        
        if np.any(np.isfinite(scnr_nominal_vec[target_idx])):
            print("\n[Target Nominal SCNR distribution]")
            vals, cnts = np.unique(scnr_nominal_vec[target_idx][np.isfinite(scnr_nominal_vec[target_idx])], return_counts=True)
            for v, c in zip(vals, cnts):
                print(f"SCNR_nom={v:7.1f} dB | count={c}")

        print("\n[Target SNR distribution]")
        vals, cnts = np.unique(snr_vec[target_idx], return_counts=True)
        for v, c in zip(vals, cnts):
            print(f"SNR={v:6.1f} dB | count={c}")

        finite_rcs = rcs_vec[target_idx][np.isfinite(rcs_vec[target_idx])]
        if len(finite_rcs) > 0:
            print("\n[Target RCS summary]")
            print(f"min  = {np.nanmin(finite_rcs):.2f}")
            print(f"mean = {np.nanmean(finite_rcs):.2f}")
            print(f"max  = {np.nanmax(finite_rcs):.2f}")

        print("\n[Target range class distribution]")
        vals, cnts = np.unique(range_cls[target_idx], return_counts=True)
        for v, c in zip(vals, cnts):
            print(f"range_cls={v} | count={c}")


# report_split_distribution("Train", idx_train)
report_split_distribution("Validation", idx_val)

print(f"\n[Info] 驗證集筆數: {len(X_val)}")
print(f"[Info] val target samples: {np.sum(det_val == 1)}, val no-target samples: {np.sum(det_val == 0)}")
print(f"[Info] pos_min={pos_min}, pos_max={pos_max}")
print(f"[Info] rbi_min={rbi_min:.4f}, rbi_max={rbi_max:.4f}")
print(f"[Info] y_pos shape: {y_pos.shape}")
print(f"[Info] y_geo shape: {y_geo.shape}")

# ---------------- Model ----------------
def build_mimo_model(num_rx, num_pulses, sequence_length, num_features,
    pnSeq1, pnSeq2, pos_min, pos_max,
    tx_pos_all, rx_pos_all, rbi_min, rbi_max,
    cfar_win_len, cfar_guard_len):
    
    input_signal = Input(
        shape=(num_rx, num_pulses, sequence_length, num_features),
        name="mimo_feature_input"
    )

    pn1_mf = pnSeq1[::-1].copy().astype("float32")
    pn2_mf = pnSeq2[::-1].copy().astype("float32")

    if len(pn1_mf) != len(pn2_mf):
        raise ValueError("pnSeq1 與 pnSeq2 長度不一致。")

    kernel_size = len(pn1_mf)

    # filters = 4:
    # filter 0: PN1 for I
    # filter 1: PN1 for Q
    # filter 2: PN2 for I
    # filter 3: PN2 for Q
    init_weight = np.zeros((kernel_size, 2, 4), dtype=np.float32)
    init_weight[:, 0, 0] = pn1_mf
    init_weight[:, 1, 1] = pn1_mf
    init_weight[:, 0, 2] = pn2_mf
    init_weight[:, 1, 3] = pn2_mf

    # (B, Rx, P, L, 2) -> (B*Rx*P, L, 2)
    x = Lambda(
        lambda t: tf.reshape(t, (-1, sequence_length, num_features)),
        name="merge_batch_rx_pulse"
    )(input_signal)

    # PN matched filter bank
    x = Conv1D(
        filters=4,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer=Constant(init_weight),
        use_bias=False,
        trainable=True,
        name="pn_matched_filter_bank"
    )(x)

    # (B*Rx*P, L, 4) -> (B*Rx*P, L, 2Tx, 2IQ)
    x = Lambda(
        lambda t: tf.reshape(t, (-1, sequence_length, 2, 2)),
        name="split_tx_iq"
    )(x)

    # 每個 Tx code 分別算 power: I^2 + Q^2
    # output: (B*Rx*P, L, 2Tx)
    x = Lambda(
        lambda t: tf.reduce_sum(tf.square(t), axis=-1),
        name="per_tx_power"
    )(x)

    # 整理成 virtual channel 格式給 OS-CFAR-like layer
    # (B*Rx*P, L, 2Tx) -> (B*Tx*Rx*P, L, 1)
    def to_virtual_channels_for_cfar(t):
        batch_size_dyn = tf.shape(t)[0] // (num_rx * num_pulses)

        t = tf.reshape(
            t,
            (batch_size_dyn, num_rx, num_pulses, sequence_length, 2)
        )  # (B, Rx, P, L, Tx)

        t = tf.transpose(t, perm=[0, 4, 1, 2, 3])  # (B, Tx, Rx, P, L)

        t = tf.reshape(
            t,
            (-1, sequence_length, 1)
        )  # (B*Tx*Rx*P, L, 1)

        return t

    x = Lambda(
        to_virtual_channels_for_cfar,
        name="to_virtual_channels_for_cfar"
    )(x)

    x = os_cfar_ratio_layer(
        x,
        win_len=cfar_win_len,
        guard_len=cfar_guard_len,
        rank_ratio=0.85,
        eps=1e-8,
        name="os_cfar_ratio"
    )

    x = Lambda(log_layer, name="log_power")(x)

    # 還原成 CNN 可用的 map
    # (B*Tx*Rx*P, L, 1) -> (B, L, Tx*Rx*P)
    def restore_mimo_map(t):
        batch_size_dyn = tf.shape(t)[0] // (2 * num_rx * num_pulses)

        t = tf.reshape(
            t,
            (batch_size_dyn, 2, num_rx, num_pulses, sequence_length, 1)
        )  # (B, Tx, Rx, P, L, 1)

        t = tf.squeeze(t, axis=-1)                # (B, Tx, Rx, P, L)
        t = tf.transpose(t, perm=[0, 4, 1, 2, 3]) # (B, L, Tx, Rx, P)

        t = tf.reshape(
            t,
            (batch_size_dyn, sequence_length, 2 * num_rx * num_pulses)
        )  # (B, L, Tx*Rx*P)

        return t

    x = Lambda(
        restore_mimo_map,
        name="mimo_virtual_pulse_as_channel"
    )(x)

    # 1D-CNN backend
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

    # detection output
    det_output = Dense(1, activation="sigmoid", name="det_output")(x)

    # normalized position output: (x, y), both in 0~1
    pos_output = Dense(2, activation="sigmoid", name="pos_output")(x)

    # geometry auxiliary output
    pos_min_tf = tf.constant(pos_min.reshape(1, 2), dtype=tf.float32)
    pos_max_tf = tf.constant(pos_max.reshape(1, 2), dtype=tf.float32)

    tx_tf = tf.constant(tx_pos_all, dtype=tf.float32)  # (2, 2)
    rx_tf = tf.constant(rx_pos_all, dtype=tf.float32)  # (2, 2)

    rbi_min_tf = tf.constant(rbi_min, dtype=tf.float32)
    rbi_max_tf = tf.constant(rbi_max, dtype=tf.float32)

    def geo_from_pos(pos_norm):
        # pos_norm: (B, 2), normalized position
        pos = pos_norm * (pos_max_tf - pos_min_tf) + pos_min_tf  # (B, 2)

        p  = pos[:, tf.newaxis, tf.newaxis, :]       # (B, 1, 1, 2)
        tx = tx_tf[tf.newaxis, :, tf.newaxis, :]     # (1, Tx, 1, 2)
        rx = rx_tf[tf.newaxis, tf.newaxis, :, :]     # (1, 1, Rx, 2)

        Rt = tf.norm(p - tx, axis=-1)  # (B, Tx, 1)
        Rr = tf.norm(p - rx, axis=-1)  # (B, 1, Rx)

        rho = Rt + Rr                  # (B, Tx, Rx)

        # flatten order:
        # Tx1Rx1, Tx1Rx2, Tx2Rx1, Tx2Rx2
        rho = tf.reshape(rho, (-1, 4))

        rho_norm = (rho - rbi_min_tf) / (rbi_max_tf - rbi_min_tf + 1e-8)

        return rho_norm

    geo_output = Lambda(geo_from_pos, name="geo_output")(pos_output)

    model = Model(
        inputs=input_signal,
        outputs={
            "det_output": det_output,
            "pos_output": pos_output,
            "geo_output": geo_output
        },
        name="mimo_2tx2rx_det_pos_geo"
    )

    return model


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
    cfar_win_len=cfar_win_len,
    cfar_guard_len=cfar_guard_len
)

pos_huber = tf.keras.losses.Huber(delta=0.03)
geo_huber = tf.keras.losses.Huber(delta=0.03)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate,clipnorm=1.0),
    loss={
        "det_output": "binary_crossentropy",
        "pos_output": "mae",
        "geo_output": "mae"
    },
    loss_weights={
        "det_output": 0.8,
        "pos_output": 1.0,
        "geo_output": 0.5
    },
    metrics={
        "det_output": ["accuracy"]
    },
    weighted_metrics={
        "pos_output": ["mae"],
        "geo_output": ["mae"]
    }
)

model.summary()

# callbacks
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.4,
    patience=3,
    min_lr=1e-7,
    verbose=1
)
early_stp = EarlyStopping(
    monitor='val_loss',
    patience=6,
    restore_best_weights=True,
    mode='min'
)

# ---------------- 訓練 ----------------
train_start_time = time.time()

history = model.fit(
    X_train,
    {
        "det_output": det_train,
        "pos_output": pos_train,
        "geo_output": geo_train
    },
    sample_weight={
        "det_output": np.ones_like(det_train.squeeze(), dtype="float32"),
        "pos_output": mask_train,
        "geo_output": mask_train
    },
    epochs=num_epochs,
    batch_size=batch_size,
    validation_data=(
        X_val,
        {
            "det_output": det_val,
            "pos_output": pos_val,
            "geo_output": geo_val
        },
        {
            "det_output": np.ones_like(det_val.squeeze(), dtype="float32"),
            "pos_output": mask_val,
            "geo_output": mask_val
        }
    ),
    shuffle=True,
    verbose=0,
    callbacks=[early_stp, reduce_lr, OneLineBatchProgress()]
)

train_end_time = time.time()
train_total_sec = train_end_time - train_start_time

hours = int(train_total_sec // 3600)
minutes = int((train_total_sec % 3600) // 60)
seconds = train_total_sec % 60


print("\n========== 訓練時間 ==========")
print(f"本次訓練用了: {hours}h {minutes}m {seconds:.1f}s")

# ---------------- 預測（用驗證集） ----------------
pred_val = model.predict(X_val, verbose=0)

if isinstance(pred_val, dict):
    det_val_prob = pred_val["det_output"].squeeze()
    pos_val_pred_norm = pred_val["pos_output"]
    geo_val_pred_norm = pred_val["geo_output"]
else:
    det_val_prob = pred_val[0].squeeze()
    pos_val_pred_norm = pred_val[1]
    geo_val_pred_norm = pred_val[2]

det_threshold = 0.5
det_val_pred = (det_val_prob >= det_threshold).astype(int)

# normalized position -> physical position
pos_val_pred_norm = np.clip(pos_val_pred_norm, 0.0, 1.0)
pos_val_pred = pos_val_pred_norm * (pos_max - pos_min) + pos_min


# # ---------------- 預測（用訓練集） ----------------
# pred_train = model.predict(X_train, verbose=0)

# if isinstance(pred_train, dict):
#     det_train_prob = pred_train["det_output"].squeeze()
#     pos_train_pred_norm = pred_train["pos_output"]
#     geo_train_pred_norm = pred_train["geo_output"]
# else:
#     det_train_prob = pred_train[0].squeeze()
#     pos_train_pred_norm = pred_train[1]
#     geo_train_pred_norm = pred_train[2]

# det_train_pred = (det_train_prob >= det_threshold).astype(int)

# pos_train_pred_norm = np.clip(pos_train_pred_norm, 0.0, 1.0)
# pos_train_pred = pos_train_pred_norm * (pos_max - pos_min) + pos_min

# ---------------- 找錯誤樣本（針對驗證集 detection） ----------------
wrong_idx = np.where(det_val_pred != det_val.squeeze().astype(int))[0]

print("\n========== Detection 錯誤樣本 ==========")
print(f"[Val] 錯誤樣本數: {len(wrong_idx)}")

header = (
    f"{'val_index':>9} | "
    f"{'global_index':>12} | "
    f"{'true_det':>8} | "
    f"{'pred_det':>8} | "
    f"{'prob':>8} | "
    f"{'Global_SCNR(dB)':>15}"
)

print(header)
print("-" * len(header))

for k in wrong_idx[:20]:
    scnr_str = "NaN" if not np.isfinite(scnr_global_val[k]) else f"{float(scnr_global_val[k]):.2f}"

    print(
        f"{int(k):9d} | "
        f"{int(idx_val[k]):12d} | "
        f"{int(det_val[k, 0]):8d} | "
        f"{int(det_val_pred[k]):8d} | "
        f"{float(det_val_prob[k]):8.4f} | "
        f"{scnr_str:>15}"
    )

# ---------------- 評估 ----------------
# train_eval = model.evaluate(
#     X_train,
#     {
#         "det_output": det_train,
#         "pos_output": pos_train,
#         "geo_output": geo_train
#     },
#     sample_weight={
#         "det_output": np.ones_like(det_train.squeeze(), dtype="float32"),
#         "pos_output": mask_train,
#         "geo_output": mask_train
#     },
#     verbose=0
# )

val_eval = model.evaluate(
    X_val,
    {
        "det_output": det_val,
        "pos_output": pos_val,
        "geo_output": geo_val
    },
    sample_weight={
        "det_output": np.ones_like(det_val.squeeze(), dtype="float32"),
        "pos_output": mask_val,
        "geo_output": mask_val
    },
    verbose=0
)

# ---------- binary detection metrics ----------
val_precision_bin = precision_score(det_val.squeeze().astype(int), det_val_pred, zero_division=0)
val_recall_bin    = recall_score(det_val.squeeze().astype(int), det_val_pred, zero_division=0)
val_f1_bin        = f1_score(det_val.squeeze().astype(int), det_val_pred, zero_division=0)
val_cm_bin        = confusion_matrix(det_val.squeeze().astype(int), det_val_pred)

# ---------- detection threshold sweep ----------
def threshold_sweep_report(y_true, det_prob, thresholds=None):
    """
    掃描不同 detection threshold，觀察 P_FA、P_D、precision、F1 的變化。
    y_true   : 真實 detection label，0/1
    det_prob : det_output 輸出的目標機率
    """
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
        recall    = recall_score(y_true, pred, zero_division=0)
        f1        = f1_score(y_true, pred, zero_division=0)

        p_fa = fp / max(fp + tn, 1)

        print(
            f"{th:4.2f} | "
            f"{tn:4d} {fp:4d} {fn:4d} {tp:4d} | "
            f"{precision:9.4f} | "
            f"{recall:9.4f} | "
            f"{f1:6.4f} | "
            f"{p_fa:6.4f}"
        )
        
def global_scnr_detection_accuracy_report_and_plot(
    scnr_global_db,
    y_true,
    y_pred,
    out_dir,
    tag="Val",
    bin_width=1.0,
    min_bin_count=30
):
    """
    依 global clutter-based SCNR 自動分組統計 target samples only 的 detection accuracy。

    這裡的 SCNR 定義為：
        目標功率 /（整段 Rx1 clutter 平均功率 + 整段 Rx1 noise 平均功率）

    也就是控制樣本整體雜波背景下的目標強度，
    不是 target 所在 range bin 附近的 local SCNR。

    對 target samples 而言，detection accuracy 等價於 Pd。
    bin_width = 1.0 表示每 1 dB 一格。
    """

    scnr_global_db = np.asarray(scnr_global_db).squeeze()
    y_true = np.asarray(y_true).squeeze().astype(int)
    y_pred = np.asarray(y_pred).squeeze().astype(int)

    # SCNR 只對 target samples 有嚴格物理意義
    valid_mask = (y_true == 1) & np.isfinite(scnr_global_db)

    scnr_t = scnr_global_db[valid_mask]
    y_pred_t = y_pred[valid_mask]

    if len(scnr_t) == 0:
        print(f"[{tag}] 沒有可用的 target global SCNR 樣本，無法繪製 global SCNR-accuracy 圖。")
        return

    # ---------- 自動決定 SCNR bin ----------
    scnr_min = np.floor(np.nanmin(scnr_t) / bin_width) * bin_width
    scnr_max = np.ceil(np.nanmax(scnr_t) / bin_width) * bin_width

    if scnr_max <= scnr_min:
        scnr_max = scnr_min + bin_width

    bin_edges = np.arange(scnr_min, scnr_max + bin_width, bin_width)

    print(f"\n[Info] {tag} global clutter-based SCNR auto bins:")
    print(f"Global SCNR min = {np.nanmin(scnr_t):.2f} dB")
    print(f"Global SCNR max = {np.nanmax(scnr_t):.2f} dB")
    print(f"bin width = {bin_width:.2f} dB")
    print(f"number of bins = {len(bin_edges) - 1}")

    bin_x = []
    acc_list = []
    count_list = []

    print(f"\n========== {tag} Global SCNR-binned Detection Accuracy ==========")
    print(" Global SCNR bin (dB) | median global SCNR | count | correct | accuracy/Pd | reliability")

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
            print(f"{label:18s} | {'NaN':>11s} | {count:5d} | {'-':>7s} | {'NaN':>10s} | {'LOW':>10s}")
            continue

        correct = np.sum(y_pred_t[mask] == 1)
        acc = correct / count

        reliability = "OK" if count >= min_bin_count else "LOW"
        # 用該 bin 內實際 SCNR 的中位數當 x 軸位置，比 bin center 更準確
        x_now = np.nanmedian(scnr_t[mask])

        bin_x.append(x_now)
        acc_list.append(acc)
        count_list.append(count)

        print(f"{label:18s} | "
            f"{x_now:11.2f} | "
            f"{count:5d} | "
            f"{correct:7d} | "
            f"{acc:10.4f} | "
            f"{reliability:>10s}")

    if len(acc_list) == 0:
        print(f"[{tag}] 所有 SCNR bin 皆無樣本，略過繪圖。")
        return

    bin_x = np.array(bin_x)
    acc_list = np.array(acc_list)
    count_list = np.array(count_list)

    # 依 x 軸排序，避免畫線順序錯亂
    order = np.argsort(bin_x)
    bin_x = bin_x[order]
    acc_list = acc_list[order]
    count_list = count_list[order]

    plt.figure(figsize=(8, 4.5))
    plt.plot(bin_x, acc_list, marker='o')
    plt.xlabel("Global SCNR (dB)")
    plt.ylabel("Detection Accuracy / Pd")
    plt.title(f"{tag} Detection Accuracy vs Global SCNR")
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)

    for x, y, n in zip(bin_x, acc_list, count_list):
        plt.text(x, y, f"n={n}", fontsize=8, ha='center', va='bottom')

    save_path = os.path.join(out_dir, f"{tag.lower()}_global_scnr_detection_accuracy_auto.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"[Info] {tag} global SCNR-accuracy 圖已儲存: {save_path}")

# ---------- position regression metrics ----------
# train_target_mask = (det_train.squeeze() == 1)
val_target_mask   = (det_val.squeeze() == 1)

# train_pos_err = pos_train_pred[train_target_mask] - pos_raw_train[train_target_mask]
val_pos_err   = pos_val_pred[val_target_mask] - pos_raw_val[val_target_mask]

# train_pos_mae_xy = np.mean(np.abs(train_pos_err), axis=0)
val_pos_mae_xy   = np.mean(np.abs(val_pos_err), axis=0)

# train_pos_rmse_xy = np.sqrt(np.mean(train_pos_err ** 2, axis=0))
val_pos_rmse_xy   = np.sqrt(np.mean(val_pos_err ** 2, axis=0))

# train_pos_euclidean_mae = np.mean(np.linalg.norm(train_pos_err, axis=1))
val_pos_euclidean_mae   = np.mean(np.linalg.norm(val_pos_err, axis=1))

threshold_sweep_report(
    y_true=det_val.squeeze(),
    det_prob=det_val_prob,
    thresholds=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
)

print("\n========== Detection Metrics ==========")
print(f"[Val]   precision: {val_precision_bin:.4f}, recall/Pd: {val_recall_bin:.4f}, F1: {val_f1_bin:.4f}")
print(val_cm_bin)

# print("\n========== Position Regression Metrics ==========")
# print(f"[Train] X MAE: {train_pos_mae_xy[0]:.4f} m")
# print(f"[Train] Y MAE: {train_pos_mae_xy[1]:.4f} m")
# print(f"[Train] X RMSE: {train_pos_rmse_xy[0]:.4f} m")
# print(f"[Train] Y RMSE: {train_pos_rmse_xy[1]:.4f} m")
# print(f"[Train] Position Euclidean MAE: {train_pos_euclidean_mae:.4f} m")

print(f"[Val]   X MAE: {val_pos_mae_xy[0]:.4f} m")
print(f"[Val]   Y MAE: {val_pos_mae_xy[1]:.4f} m")
print(f"[Val]   X RMSE: {val_pos_rmse_xy[0]:.4f} m")
print(f"[Val]   Y RMSE: {val_pos_rmse_xy[1]:.4f} m")
print(f"[Val]   Position Euclidean MAE: {val_pos_euclidean_mae:.4f} m")

global_scnr_detection_accuracy_report_and_plot(
    scnr_global_db=scnr_global_val,
    y_true=det_val.squeeze(),
    y_pred=det_val_pred,
    out_dir=model_dir,
    tag="Val",
    bin_width=5.0,
    min_bin_count=30
)

# ---------------- 儲存 ----------------
os.makedirs(model_dir, exist_ok=True)

plt.figure(figsize=(6, 4))
plt.plot(history.history["det_output_accuracy"], label="train_det")
plt.plot(history.history["val_det_output_accuracy"], label="val_det")
plt.xlabel("Epoch")
plt.ylabel("Detection Accuracy")
plt.legend()
plt.title("Detection Accuracy")
plt.savefig(os.path.join(model_dir, "det_acc.png"))
plt.close()

plt.figure(figsize=(6, 4))
plt.plot(history.history["loss"], label="train_total")
plt.plot(history.history["val_loss"], label="val_total")
plt.xlabel("Epoch")
plt.ylabel("Total Loss")
plt.legend()
plt.title("Total Loss")
plt.savefig(os.path.join(model_dir, "total_loss.png"))
plt.close()

plt.figure(figsize=(6, 4))
plt.plot(history.history["pos_output_mae"], label="train_pos_mae")
plt.plot(history.history["val_pos_output_mae"], label="val_pos_mae")
plt.xlabel("Epoch")
plt.ylabel("Normalized Position MAE")
plt.legend()
plt.title("Position Regression MAE")
plt.savefig(os.path.join(model_dir, "pos_mae.png"))
plt.close()

plt.figure(figsize=(6, 4))
plt.plot(history.history["geo_output_mae"], label="train_geo_mae")
plt.plot(history.history["val_geo_output_mae"], label="val_geo_mae")
plt.xlabel("Epoch")
plt.ylabel("Normalized Geometry MAE")
plt.legend()
plt.title("Bistatic Geometry Auxiliary MAE")
plt.savefig(os.path.join(model_dir, "geo_mae.png"))
plt.close()

weights_path = os.path.join(model_dir, "mimo_det_pos_geo_weights.h5")
norm_path = os.path.join(model_dir, "mimo_norm_params.pkl")

with open(norm_path, "wb") as f:
    pickle.dump({
        "pos_min": pos_min,
        "pos_max": pos_max,
        "rbi_min": float(rbi_min),
        "rbi_max": float(rbi_max),
        "tx_pos_all": tx_pos_all,
        "rx_pos_all": rx_pos_all
    }, f)

print(f"[Info] MIMO normalization params 已存到: {norm_path}")

model.save_weights(weights_path)
print(f"[Info] 模型權重已存到: {weights_path}")

history_path = os.path.join(model_dir, "history.pkl")
with open(history_path, "wb") as f:
    pickle.dump(history.history, f)
print(f"[Info] history 已存到: {history_path}")

# 不執行 Train Predict/Evaluate，避免 X_train 過大造成記憶體不足。
# 訓練集的 loss / accuracy 已經存在 history.history 中；
# 最終模型效能請以 validation set 或獨立 test set 評估。

print(f"[Info] 訓練時間後的終端輸出已存成 TXT: {log_path}")

sys.stdout.flush()
sys.stderr.flush()

sys.stdout = _original_stdout
sys.stderr = _original_stderr

_log_file.close()