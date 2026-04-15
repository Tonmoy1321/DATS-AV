import os
import time
import json
import glob
import random
import warnings
import subprocess
import tempfile
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys

# --------------------------
# Silence OpenCV/FFmpeg logging (e.g., "moov atom not found")
# --------------------------
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)  # OpenCV >= 4.5
except Exception:
    pass
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

# --------------------------
# GPU preference (optional)
# --------------------------
if torch.cuda.is_available():
    try:
        torch.cuda.set_device(0)
    except Exception:
        pass

# =========================
# VISUAL-75 presets & default VIS-20 (indices)
# =========================
RECOMMENDED_SETS = {
    "one": [39],
    "small": [39, 40, 53, 36, 65, 37, 33, 34, 0, 4],
    "bins9_11_pack": [33, 34, 36, 37, 39, 40, 51, 52, 53, 63, 64, 65, 0, 4],
}
DEFAULT_VISUAL20_INDICES = [70, 72, 71, 73, 66, 68, 67, 69, 74, 65, 3, 64, 20, 41, 53, 14, 17, 52, 23, 8]
DEFAULT_SELECTION_JSON = "visual75_crossdomain_selection.json"

# =========================
# Split alias helpers
# =========================
def _desired_splits(mode: str, subset_req: str) -> List[str]:
    s = subset_req.lower().strip()
    if mode == "av_deepfake1m":
        if s in {"eval", "val"}:
            return ["val", "eval"]
        return [s]
    if mode == "lavdf":
        if s == "eval":
            return ["dev"]
        return [s]
    return [s]

def _folder_aliases_for_files(mode: str, subset_req: str) -> List[str]:
    s = subset_req.lower().strip()
    if mode == "av_deepfake1m":
        if s in {"eval", "val"}:
            return ["val", "eval", "validation"]
        return [s]
    if mode == "lavdf":
        if s == "eval":
            return ["dev", "eval", "validation"]
        return [s]
    return [s]

# =========================
# Selection JSON helpers (VIS-20)
# =========================
def _normalize_selection_indices(obj: Any) -> Optional[List[int]]:
    if obj is None:
        return None
    if isinstance(obj, list) and all(isinstance(x, (int, np.integer)) for x in obj):
        return sorted(set(int(x) for x in obj))
    if isinstance(obj, dict):
        for k in ["indices", "selected_indices", "vis20", "visual20", "top20", "selection"]:
            if k in obj:
                vals = obj[k]
                if isinstance(vals, list) and all(isinstance(x, (int, np.integer)) for x in vals):
                    return sorted(set(int(x) for x in vals))
    return None

def _load_vis20_indices(json_path: Optional[str], key: Optional[str] = None, mode_hint: Optional[str] = None) -> Optional[List[int]]:
    path = json_path or DEFAULT_SELECTION_JSON
    if not (isinstance(path, str) and os.path.isfile(path)):
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception:
        return None

    sel = _normalize_selection_indices(data)
    if sel is not None:
        return sel

    if isinstance(data, dict):
        if key and key in data:
            sel = _normalize_selection_indices(data[key])
            if sel is not None:
                return sel
        if mode_hint:
            if mode_hint in data:
                sel = _normalize_selection_indices(data[mode_hint])
                if sel is not None:
                    return sel
            aliases = {
                "av_deepfake1m": ["avdf1m", "av-deepfake1m", "av_df1m", "avdf_1m", "avdf"],
                "fakeavceleb": ["favc", "fakeav", "fake-av-celeb"],
                "lavdf": ["lav-df", "lav_df", "lav"],
            }.get(mode_hint, [])
            for a in aliases:
                if a in data:
                    sel = _normalize_selection_indices(data[a])
                    if sel is not None:
                        return sel
        for v in data.values():
            sel = _normalize_selection_indices(v)
            if sel is not None:
                return sel
    return None

# =========================
# Path helpers
# =========================
VIDEO_EXTS = [".mp4", ".mov", ".avi", ".mkv", ".webm",
              ".MP4", ".MOV", ".AVI", ".MKV", ".WEBM"]
AUDIO_EXTS = [".wav", ".mp3", ".m4a", ".flac", ".aac",
              ".WAV", ".MP3", ".M4A", ".FLAC", ".AAC"]

SUBFOLDERS_GUESS = [
    "eval", "val", "validation", "dev", "test", "train",
    "Eval", "Val", "Validation", "Dev", "Test", "Train",
    "evaluation", "Evaluation", "VALIDATION", "EVAL"
]

def _maybe_swap_ext_to_video(path: str) -> List[str]:
    cand = []
    bn, ext = os.path.splitext(path)
    if ext in AUDIO_EXTS or ext == "":
        for vext in VIDEO_EXTS:
            cand.append(bn + vext)
    return cand

def _glob_one(pattern: str) -> Optional[str]:
    for p in glob.iglob(pattern, recursive=True):
        if os.path.isfile(p):
            return p
    return None

def _find_case_insensitive(root_dir: str, base_noext: str) -> Optional[str]:
    base_lower = base_noext.lower()
    for ext in VIDEO_EXTS:
        pattern = os.path.join(root_dir, "**", "*" + ext)
        for p in glob.iglob(pattern, recursive=True):
            try:
                if os.path.isfile(p):
                    if os.path.splitext(os.path.basename(p))[0].lower() == base_lower:
                        return p
            except Exception:
                continue
    return None

def _find_video_by_basename(root_dir: str, base_noext: str) -> Optional[str]:
    for sub in SUBFOLDERS_GUESS:
        for ext in VIDEO_EXTS:
            p = os.path.join(root_dir, sub, base_noext + ext)
            if os.path.isfile(p):
                return p
    p = _find_case_insensitive(root_dir, base_noext)
    if p:
        return p
    for ext in VIDEO_EXTS:
        pattern = os.path.join(root_dir, "**", base_noext + ext)
        p = _glob_one(pattern)
        if p:
            return p
    return None

def _resolve_visual_path(root_dir: str, csv_fp: str, subset_hints: List[str]) -> Optional[str]:
    if os.path.isabs(csv_fp) and os.path.isfile(csv_fp) and os.path.splitext(csv_fp)[1] in VIDEO_EXTS:
        return csv_fp
    cand_abs = os.path.join(root_dir, csv_fp) if not os.path.isabs(csv_fp) else csv_fp
    if os.path.isfile(cand_abs) and os.path.splitext(cand_abs)[1] in VIDEO_EXTS:
        return cand_abs
    for p2 in _maybe_swap_ext_to_video(cand_abs):
        if os.path.isfile(p2):
            return p2
    base = os.path.splitext(os.path.basename(csv_fp))[0]
    for hint in subset_hints:
        for ext in VIDEO_EXTS:
            p = os.path.join(root_dir, hint, base + ext)
            if os.path.isfile(p):
                return p
    return _find_video_by_basename(root_dir, base)

# =========================
# Dataset readers
# =========================
def _read_favc_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    need = {"split", "file_path", "label"}
    if not need.issubset({c.lower() for c in df.columns}):
        raise ValueError("CSV must contain columns: split, file_path, label")
    cols = {c.lower(): c for c in df.columns}
    out = df[[cols["split"], cols["file_path"], cols["label"]]].copy()
    out.columns = ["split", "file_path", "label"]
    out["split"] = out["split"].astype(str).str.strip().str.lower()

    def _lab2int(x):
        if isinstance(x, str):
            xs = x.strip().lower()
            if xs in ("real", "0"):
                return 0
            if xs in ("fake", "1"):
                return 1
        try:
            xi = int(x)
            return 1 if xi == 1 else 0
        except Exception:
            pass
        raise ValueError(f"Unrecognized label value: {x}")

    out["label"] = out["label"].apply(_lab2int).astype(int)
    out["file_path"] = out["file_path"].astype(str).str.strip()
    return out

def _lavdf_iter_entries(json_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)
    items = list(data.values()) if isinstance(data, dict) else list(data)
    for it in items:
        file_field = (it.get("file") or it.get("path") or "").strip()
        if not file_field:
            continue
        split = str(it.get("split", "")).strip().lower()
        if split not in {"train", "test", "dev", "eval", "validation", "val"}:
            split = "train"
        modify_audio = bool(it.get("modify_audio", False))
        modify_video = bool(it.get("modify_video", False))
        periods = it.get("fake_periods", []) or []
        periods_clean = []
        for p in periods:
            if isinstance(p, (list, tuple)) and len(p) == 2:
                try:
                    s = float(p[0]); e = float(p[1])
                    if e > s: periods_clean.append([s, e])
                except Exception:
                    pass
        yield {
            "file": file_field,
            "split": split,
            "modify_audio": modify_audio,
            "modify_video": modify_video,
            "fake_periods": periods_clean,
        }

def _avdf1m_iter_entries(json_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)
    items = list(data.values()) if isinstance(data, dict) else list(data)
    for it in items:
        file_field = (it.get("file") or it.get("path") or "").strip()
        if not file_field:
            continue
        split = (it.get("split") or "val").strip().lower()
        modify_type = (it.get("modify_type") or "").strip().lower()
        segs = it.get("fake_segments", []) or []
        segs_clean = []
        for p in segs:
            if isinstance(p, (list, tuple)) and len(p) == 2:
                try:
                    s = float(p[0]); e = float(p[1])
                    if e > s: segs_clean.append([s, e])
                except Exception:
                    pass
        yield {
            "file": file_field,
            "split": split,
            "modify_type": modify_type,
            "fake_segments": segs_clean,
        }

# =========================
# OpenFace helpers
# =========================
EYE_L = list(range(36, 42))
EYE_R = list(range(42, 48))
MOUTH_OUT = list(range(48, 60))

def _ensure_openface_csv(video_path: str, cache_dir: str, openface_binary: str) -> Optional[str]:
    os.makedirs(cache_dir or ".", exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = os.path.join(cache_dir, f"{base}.csv")
    if os.path.isfile(csv_path):
        return csv_path
    if not openface_binary:
        return None
    try:
        cmd = [openface_binary, "-f", video_path, "-aus", "-2Dfp", "-out_dir", cache_dir]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return csv_path if os.path.isfile(csv_path) else None
    except Exception:
        return None

def _read_openface_csv(csv_path: str) -> Optional[pd.DataFrame]:
    for enc in (None, "utf-8-sig", "latin1"):
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            df.columns = df.columns.str.strip()
            return df
        except Exception:
            continue
    return None

def _collect_frame_landmarks(df: pd.DataFrame):
    try:
        lm_cols = [f"{ax}_{i}" for i in range(68) for ax in ("x", "y")]
        if not set(lm_cols).issubset(df.columns):
            return None, None
        lms = df[lm_cols].values.reshape((-1, 68, 2)).astype(np.float32)
        success = df["success"].astype(np.float32).values if "success" in df.columns else np.ones((len(df),), dtype=np.float32)
        return lms, success
    except Exception:
        return None, None

def _map_frames_to_of_rows(df: Optional[pd.DataFrame], frame_indices: List[int], fps_video: float) -> List[int]:
    if df is None:
        return [0 for _ in frame_indices]
    n = len(df)
    if n == 0:
        return [0 for _ in frame_indices]
    if "timestamp" in df.columns:
        ts = df["timestamp"].to_numpy(dtype=np.float64)
        rows = []
        for fidx in frame_indices:
            t = float(fidx) / max(fps_video, 1e-6)
            j = int(np.searchsorted(ts, t, side="left"))
            if j == n:
                j = n - 1
            elif j > 0 and (t - ts[j - 1]) < (ts[j] - t):
                j -= 1
            rows.append(max(min(j, n - 1), 0))
        return rows
    if "frame" in df.columns:
        of_frames = df["frame"].to_numpy(dtype=np.int64)
        return [int(np.argmin(np.abs(of_frames - fidx))) for fidx in frame_indices]
    return [max(min(int(f), n - 1), 0) for f in frame_indices]

# =========================
# Visual-75 utilities
# =========================
def _summary_stats(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float32)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": 0.0, "std": 0.0, "skew": 0.0, "kurt": 0.0, "p50": 0.0, "iqr": 0.0}
    from scipy.stats import skew, kurtosis
    p25, p50, p75 = np.percentile(x, [25, 50, 75])
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "skew": float(skew(x, bias=False)) if x.size > 2 else 0.0,
        "kurt": float(kurtosis(x, bias=False)) if x.size > 3 else 0.0,
        "p50": float(p50),
        "iqr": float(p75 - p25),
    }

def _ar1(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    if x.size < 3:
        return 0.0
    x = x - np.mean(x)
    den = float(np.dot(x[:-1], x[:-1])) + 1e-8
    num = float(np.dot(x[:-1], x[1:]))
    return float(num / den)

def _psd_three_band_stats(x: np.ndarray, fs: float) -> List[float]:
    x = np.asarray(x, dtype=np.float32)
    if x.size < 8 or not np.any(np.isfinite(x)):
        return [0.0] * 9
    from scipy.signal import welch
    f, Pxx = welch(x, fs=fs, nperseg=min(256, len(x)))
    if Pxx.size == 0 or np.all(~np.isfinite(Pxx)):
        return [0.0] * 9
    fmax = f[-1] if f.size > 0 else fs / 2.0
    b1 = f <= (fmax / 3.0)
    b2 = (f > (fmax / 3.0)) & (f <= (2.0 * fmax / 3.0))
    b3 = f > (2.0 * fmax / 3.0)
    def band_stats(mask):
        vals = Pxx[mask]
        if vals.size == 0:
            return 0.0, 0.0
        return float(np.mean(vals)), float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
    low_m, low_s = band_stats(b1)
    mid_m, mid_s = band_stats(b2)
    high_m, high_s = band_stats(b3)
    def r(a, b): return float(a / (b + 1e-6))
    return [low_m, mid_m, high_m, low_s, mid_s, high_s, r(low_m, mid_m), r(mid_m, high_m), r(low_m, high_m)]

def _video_len(path: str) -> int:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return max(total, 0)

def _fps(path: str) -> float:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 25.0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    cap.release()
    return fps if fps > 1e-6 else 25.0

def _safe_bbox_from_pts(pts, W, H, expand):
    good = np.isfinite(pts).all(axis=1)
    if good.sum() < 3:
        return None
    p = pts[good]
    x_min = int(np.floor(p[:, 0].min()))
    y_min = int(np.floor(p[:, 1].min()))
    x_max = int(np.ceil(p[:, 0].max()))
    y_max = int(np.ceil(p[:, 1].max()))
    w = x_max - x_min
    h = y_max - y_min
    if w <= 0 or h <= 0:
        return None
    x_exp = int(round(w * expand))
    y_exp = int(round(h * expand))
    x1 = max(0, x_min - x_exp)
    y1 = max(0, y_min - y_exp)
    x2 = min(x_max + x_exp, W)
    y2 = min(y_max + y_exp, H)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2

def _visual75_from_frames_and_openface(
    video_path: str,
    frame_indices: List[int],
    fps_video: float,
    openface_cache_csv: str,
    mouth_expand_ratio: float = 0.4,
    flow_bins: int = 12
) -> Optional[np.ndarray]:
    df = _read_openface_csv(openface_cache_csv)
    if df is None:
        return None
    lms_all, success = _collect_frame_landmarks(df)
    if lms_all is None:
        return None

    rows = _map_frames_to_of_rows(df, frame_indices, fps_video)
    # Ensure at least two steps for flow (helps with very short/tiled spans)
    if len(rows) < 2:
        rows = rows + rows
    if len(rows) < 2:
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    flow_energy, flow_hists = [], []
    for ti in range(len(rows) - 1):
        r0 = int(rows[ti]); r1 = int(rows[ti + 1])
        if success is not None:
            if r0 >= len(success) or r1 >= len(success) or (success[r0] < 0.5 or success[r1] < 0.5):
                flow_energy.append(0.0); flow_hists.append(np.zeros((flow_bins,), dtype=np.float32)); continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_indices[ti])); ok0, f0 = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_indices[ti + 1])); ok1, f1 = cap.read()
        if not ok0 or not ok1 or f0 is None or f1 is None:
            flow_energy.append(0.0); flow_hists.append(np.zeros((flow_bins,), dtype=np.float32)); continue
        Hf, Wf = f0.shape[:2]
        lm = lms_all[r0]
        if not np.all(np.isfinite(lm)):
            flow_energy.append(0.0); flow_hists.append(np.zeros((flow_bins,), dtype=np.float32)); continue
        pts = lm[MOUTH_OUT]
        bb = _safe_bbox_from_pts(pts, Wf, Hf, expand=mouth_expand_ratio)
        if bb is None:
            flow_energy.append(0.0); flow_hists.append(np.zeros((flow_bins,), dtype=np.float32)); continue
        x1, y1, x2, y2 = bb
        g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        try:
            flow = cv2.calcOpticalFlowFarneback(
                g0, g1, None, pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                poly_n=5, poly_sigma=1.2, flags=0
            )
            u = flow[y1:y2, x1:x2, 0]; v = flow[y1:y2, x1:x2, 1]
            mag = np.sqrt(u * u + v * v).astype(np.float32)
            if mag.size == 0:
                flow_energy.append(0.0); flow_hists.append(np.zeros((flow_bins,), dtype=np.float32)); continue
            q95 = float(np.percentile(mag, 95)) if np.isfinite(mag).all() else 0.0
            scale = max(q95, 1e-6)
            mag_n = np.clip(mag / scale, 0.0, 1.0)
            flow_energy.append(float(mag_n.mean()))
            hist, _ = np.histogram(mag_n, bins=flow_bins, range=(0.0, 1.0), density=False)
            hist = hist.astype(np.float32); s = float(hist.sum())
            if s > 0: hist /= s
            flow_hists.append(hist)
        except Exception:
            flow_energy.append(0.0); flow_hists.append(np.zeros((flow_bins,), dtype=np.float32))
    cap.release()

    flow_energy = np.asarray(flow_energy, dtype=np.float32)
    flow_hists = np.asarray(flow_hists, dtype=np.float32)
    if flow_hists.ndim != 2 or flow_hists.shape[1] != 12:
        flow_hists = np.zeros((max(len(rows) - 1, 1), 12), dtype=np.float32)

    sse = _summary_stats(flow_energy)
    energy_feats = [sse["mean"], sse["std"], sse["skew"], sse["kurt"], sse["p50"], sse["iqr"]]
    dyn_feats = []
    for k in range(12):
        series = flow_hists[:, k]
        ss = _summary_stats(series)
        dyn_feats.extend([ss["mean"], ss["std"], _ar1(series)])
    if flow_hists.shape[0] >= 2:
        d1 = np.diff(flow_hists, axis=0)
        d1_std = [float(np.std(d1[:, k], ddof=1)) if d1.shape[0] > 1 else float(np.std(d1[:, k])) for k in range(12)]
    else:
        d1_std = [0.0] * 12
    if flow_hists.shape[0] >= 3:
        d2 = np.diff(flow_hists, n=2, axis=0)
        d2_std = [float(np.std(d2[:, k], ddof=1)) if d2.shape[0] > 1 else float(np.std(d2[:, k])) for k in range(12)]
    else:
        d2_std = [0.0] * 12
    band9 = _psd_three_band_stats(flow_energy, fs=float(fps_video))
    visual75 = np.array(energy_feats + dyn_feats + d1_std + d2_std + band9, dtype=np.float32)
    if visual75.shape[0] != 75:
        return None
    if not np.all(np.isfinite(visual75)):
        return None
    return visual75

# =========================
# Audio helpers
# =========================
def _have_torchaudio():
    try:
        import torchaudio  # noqa
        return True
    except Exception:
        return False

def _have_librosa():
    try:
        import librosa  # noqa
        return True
    except Exception:
        return False

def _ffmpeg_read_mono_16k(path: str, sr: int = 16000) -> Optional[np.ndarray]:
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr), "-f", "wav", tmp.name]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                import soundfile as sf
                audio, file_sr = sf.read(tmp.name, dtype="float32", always_2d=False)
                if file_sr != sr and _have_librosa():
                    import librosa
                    audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
                return audio.astype(np.float32, copy=False)
            except Exception:
                import wave
                with wave.open(tmp.name, "rb") as wf:
                    n = wf.getnframes()
                    frames = wf.readframes(n)
                    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                return audio
    except Exception:
        return None

def _load_audio_mono_16k(path: str, target_sr: int = 16000) -> Optional[np.ndarray]:
    if _have_torchaudio():
        try:
            import torchaudio
            s, sr = torchaudio.load(path)
            if s.ndim == 2 and s.size(0) > 1:
                s = s.mean(dim=0, keepdim=True)
            elif s.ndim == 1:
                s = s.unsqueeze(0)
            if sr != target_sr:
                s = torchaudio.functional.resample(s, sr, target_sr)
            return s.squeeze(0).numpy().astype(np.float32, copy=False)
        except Exception:
            pass
    if _have_librosa():
        try:
            import librosa
            s, sr = librosa.load(path, sr=target_sr, mono=True)
            return s.astype(np.float32, copy=False)
        except Exception:
            pass
    return _ffmpeg_read_mono_16k(path, sr=target_sr)

def _stft_mag(audio: np.ndarray, sr: int, n_fft: int, hop: int, win: int) -> np.ndarray:
    if _have_librosa():
        import librosa
        S = librosa.stft(audio, n_fft=n_fft, hop_length=hop, win_length=win, window="hann", center=True)
        return np.abs(S).T.astype(np.float32)
    if _have_torchaudio():
        import torch as _t
        wav = _t.from_numpy(audio).unsqueeze(0)
        spec = _t.stft(
            wav, n_fft=n_fft, hop_length=hop, win_length=win,
            window=_t.hann_window(win), return_complex=True
        )
        return spec.abs().squeeze(0).transpose(0, 1).cpu().numpy().astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(audio, win)[::hop]
    window = np.hanning(win)
    S = np.fft.rfft(frames * window, n=n_fft)
    return np.abs(S).astype(np.float32)

def _logmel(audio: np.ndarray, sr: int, n_fft: int, hop: int, win: int, n_mels: int) -> np.ndarray:
    if _have_librosa():
        import librosa
        M = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=n_fft, hop_length=hop, win_length=win,
            n_mels=n_mels, power=2.0
        )
        return librosa.power_to_db(M, ref=np.max).T.astype(np.float32)
    if _have_torchaudio():
        import torch as _t
        import torchaudio
        wav = _t.from_numpy(audio).unsqueeze(0)
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop, win_length=win, n_mels=n_mels
        )(wav).squeeze(0)
        return _t.log(mel + 1e-6).transpose(0, 1).cpu().numpy().astype(np.float32)
    S = _stft_mag(audio, sr, n_fft, hop, win) ** 2
    return np.log(S + 1e-6)

def _mfcc_block(audio: np.ndarray, sr: int, n_mfcc: int, n_fft: int, hop: int, win: int) -> Dict[str, np.ndarray]:
    if _have_librosa():
        import librosa
        MF = librosa.feature.mfcc(
            y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft,
            hop_length=hop, win_length=win
        ).T.astype(np.float32)
        D1 = librosa.feature.delta(MF.T).T.astype(np.float32)
        D2 = librosa.feature.delta(MF.T, order=2).T.astype(np.float32)
        return {"mfcc": MF, "d_mfcc": D1, "dd_mfcc": D2}
    else:
        from scipy.fftpack import dct
        mel = _logmel(audio, sr, n_fft, hop, win, n_mels=max(40, n_mfcc))
        MF = dct(mel, type=2, norm="ortho", axis=1)[:, :n_mfcc]
        def delta(x):
            x = np.asarray(x, dtype=np.float32)
            pad = np.pad(x, ((1, 1), (0, 0)), mode="edge")
            return (pad[2:] - pad[:-2]) / 2.0
        D1 = delta(MF); D2 = delta(D1)
        return {"mfcc": MF.astype(np.float32), "d_mfcc": D1.astype(np.float32), "dd_mfcc": D2.astype(np.float32)}

def _audio_feature_pack(
    audio_seg: np.ndarray, sr: int,
    n_fft: int, hop: int, win: int,
    n_mels: int, n_mfcc: int
) -> Tuple[np.ndarray, np.ndarray, int]:
    S_mag = _stft_mag(audio_seg, sr, n_fft, hop, win)  # [F,S]
    F = int(S_mag.shape[0])

    # RMS 6
    rms_series = np.sqrt(np.sum(S_mag ** 2, axis=1))
    ss = _summary_stats(rms_series)
    rms_feats = [ss["mean"], ss["std"], ss["skew"], ss["kurt"], ss["p50"], ss["iqr"]]

    # MFCC 12*(mean,std,ar1)=36 + delta std 12 + dd std 12 = 60
    mfcc_blk = _mfcc_block(audio_seg, sr, n_mfcc=max(13, n_mfcc), n_fft=n_fft, hop=hop, win=win)
    MF = mfcc_blk["mfcc"]
    if MF.shape[1] < 13:
        pad = np.zeros((MF.shape[0], 13 - MF.shape[1]), dtype=np.float32)
        MF = np.concatenate([MF, pad], axis=1)
    MF_12 = MF[:, 1:13].astype(np.float32)
    D1 = mfcc_blk["d_mfcc"]; D2 = mfcc_blk["dd_mfcc"]
    if D1.shape[1] < 13:
        pad = np.zeros((D1.shape[0], 13 - D1.shape[1]), dtype=np.float32)
        D1 = np.concatenate([D1, pad], axis=1)
    if D2.shape[1] < 13:
        pad = np.zeros((D2.shape[0], 13 - D2.shape[1]), dtype=np.float32)
        D2 = np.concatenate([D2, pad], axis=1)
    D1_12 = D1[:, 1:13].astype(np.float32)
    D2_12 = D2[:, 1:13].astype(np.float32)

    mfcc_feats = []
    for k in range(12):
        xk = MF_12[:, k]
        ssk = _summary_stats(xk)
        mfcc_feats.extend([ssk["mean"], ssk["std"], _ar1(xk)])

    d_feats  = [float(np.std(D1_12[:, k], ddof=1)) if D1_12.shape[0] > 1 else 0.0 for k in range(12)]
    dd_feats = [float(np.std(D2_12[:, k], ddof=1)) if D2_12.shape[0] > 1 else 0.0 for k in range(12)]

    # Mel tri-band (9)
    M = _logmel(audio_seg, sr, n_fft, hop, win, n_mels=n_mels)
    M = M if M.ndim == 2 else np.atleast_2d(M)
    Mmean = M.mean(axis=0) if M.shape[0] > 0 else np.zeros((n_mels,), dtype=np.float32)
    Mstd = M.std(axis=0, ddof=1) if M.shape[0] > 1 else np.zeros_like(Mmean)

    def band(idx0, idx1):
        i0 = max(0, idx0); i1 = min(n_mels - 1, idx1)
        if i1 < i0: return np.array([0.0]), np.array([0.0])
        sl = slice(i0, i1 + 1)
        return Mmean[sl], Mstd[sl]

    low_m, low_s = band(0, 10)
    mid_m, mid_s = band(11, 30)
    hig_m, hig_s = band(31, n_mels - 1)
    low_mean, mid_mean, hig_mean = float(low_m.mean()), float(mid_m.mean()), float(hig_m.mean())
    low_std, mid_std, hig_std = float(low_s.mean()), float(mid_s.mean()), float(hig_s.mean())

    def safe_ratio(a, b): return float(a / (b + 1e-6))
    mel_feats = [
        low_mean, mid_mean, hig_mean,
        low_std,  mid_std,  hig_std,
        safe_ratio(low_mean, mid_mean),
        safe_ratio(mid_mean, hig_mean),
        safe_ratio(low_mean, hig_mean),
    ]

    audio_feats_75 = np.array(rms_feats + mfcc_feats + d_feats + dd_feats + mel_feats, dtype=np.float32)
    assert audio_feats_75.shape[0] == 75
    return audio_feats_75, S_mag.astype(np.float32), F

# =========================
# Frame selection
# =========================
def _pick_indices_window(start_f: int, end_f: int, frames_per_clip: int, stride: int) -> List[int]:
    eff = max(1, int(stride))
    need = (frames_per_clip - 1) * eff + 1
    if end_f - start_f + 1 < need:
        end = min(end_f, start_f + need - 1)
        return list(range(start_f, end + 1, eff))
    s = start_f; e = s + need - 1
    return list(range(s, e + 1, eff))

def _tile_span_indices(a: int, b: int, need: int, eff: int) -> List[int]:
    """
    Return exactly `need` frame indices by repeating the span [a..b] with step=eff.
    Example: a=100, b=105, eff=1, need=10 -> 100..105,100..103
    """
    if b < a:
        return []
    base = list(range(a, b + 1, eff))
    if len(base) == 0:
        base = [a]
    reps = (need + len(base) - 1) // len(base)
    tiled = (base * reps)[:need]
    return tiled

# =========================
# Face cropper (YOLOv8n-face with OF alignment fallback)
# =========================
class _FaceCropper:
    def __init__(self, weights_path: str = "yolov8n-face.pt", img_size: int = 224, conf: float = 0.25, iou: float = 0.5):
        self.weights_path = weights_path
        self.img_size = int(img_size)
        self.conf = float(conf); self.iou = float(iou)
        self._yolo = None; self._ready = False
        try:
            from ultralytics import YOLO
            self._yolo = YOLO(self.weights_path)
            self._ready = True
        except Exception:
            self._yolo = None; self._ready = False

    def _crop_from_box(self, frame_bgr: np.ndarray, box_xyxy: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = [int(v) for v in box_xyxy]
        H, W = frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)
        if x2 <= x1 or y2 <= y1: return None
        face = frame_bgr[y1:y2, x1:x2, :]
        face = cv2.resize(face, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        return face

    def detect_and_crop(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        if not self._ready:
            return None
        try:
            res = self._yolo.predict(
                source=frame_bgr[:, :, ::-1], imgsz=self.img_size,
                conf=self.conf, iou=self.iou, verbose=False
            )
            if (not res) or (res[0].boxes is None) or (res[0].boxes.xyxy is None) or (len(res[0].boxes.xyxy) == 0):
                return None
            boxes = res[0].boxes.xyxy.cpu().numpy()
            confs = res[0].boxes.conf.cpu().numpy()
            idx = int(np.argmax(confs))
            x1, y1, x2, y2 = boxes[idx].astype(int).tolist()
            return self._crop_from_box(frame_bgr, (x1, y1, x2, y2))
        except Exception:
            return None

def _align_crop_from_landmarks(frame_bgr: np.ndarray, lm_68: np.ndarray, out_size: int = 224) -> Optional[np.ndarray]:
    try:
        left_eye = lm_68[EYE_L].mean(axis=0); right_eye = lm_68[EYE_R].mean(axis=0)
        dy, dx = right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        eyes_center = ((left_eye[0] + right_eye[0]) * 0.5, (left_eye[1] + right_eye[1]) * 0.5)
        M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
        rotated = cv2.warpAffine(frame_bgr, M, (frame_bgr.shape[1], frame_bgr.shape[0]), flags=cv2.INTER_LINEAR)
        dist = np.hypot(dx, dy)
        box_w = int(dist * 2.0); box_h = int(dist * 2.5)
        x1 = int(eyes_center[0] - box_w / 2); y1 = int(eyes_center[1] - box_h * 0.4)
        x2 = x1 + box_w; y2 = y1 + box_h
        H, W = rotated.shape[:2]
        x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(W - 1, x2), min(H - 1, y2)
        if x2 <= x1 or y2 <= y1: return None
        face = rotated[y1:y2, x1:x2, :]
        face = cv2.resize(face, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
        return face
    except Exception:
        return None

def _fallback_center_crop(frame_bgr: np.ndarray, out_size: int = 224) -> np.ndarray:
    H, W = frame_bgr.shape[:2]
    side = min(H, W)
    cx, cy = W // 2, H // 2
    x1 = max(0, cx - side // 2); y1 = max(0, cy - side // 2)
    x2 = min(W, x1 + side); y2 = min(H, y1 + side)
    crop = frame_bgr[y1:y2, x1:x2, :]
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)

# =========================
# Quick ffprobe gate
# =========================
def _ffprobe_quick_check(path: str) -> bool:
    """
    True if ffprobe confirms a readable video stream with nonzero frames/duration.
    Falls back to simple existence if ffprobe is unavailable.
    """
    if not os.path.isfile(path):
        return False
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames", "-of", "csv=p=0", path
        ]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if r.returncode == 0:
            out = (r.stdout or b"").decode("utf-8", "ignore").strip()
            if out.isdigit() and int(out) > 0:
                return True
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", path
        ]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if r.returncode == 0:
            dur = (r.stdout or b"").decode("utf-8", "ignore").strip()
            try:
                return float(dur) > 0.1
            except Exception:
                return False
        return False
    except Exception:
        return os.path.isfile(path)

# =========================
# Multimodal Dataset (returns vis20, vis75, face, stft, aud75, labels)
# Now includes per-modality labels for LAVDF & AV-DF1M; FAVC unchanged (no per-modality).
# =========================
class UnifiedAVDataset(Dataset):
    """
    Each sample returns:
      (
        x_vis20[ B, Dv20 ],     # VIS-20 (subset for AV-Dissonance)
        x_vis75[ B, 75   ],     # VIS-75 (full for Visual-Only)
        face[ ... ],            # [B,3,H,W] or [B,T,3,H,W]
        stft[ B, F, S ],
        x_aud[ B, 75 ],
        label_mm[ B ],          # multimodal label (0/1)
        label_a[ B ]|None,      # audio label (0/1) for LAVDF/AVDF1M, else None (FAVC)
        label_v[ B ]|None,      # visual label (0/1) for LAVDF/AVDF1M, else None (FAVC)
        a_len[ B ],
        video_path
      )

    If visual (VIS-75) or audio feature extraction fails → sample is skipped with a warning.
    """

    def __init__(
        self,
        *,
        root_dir: str,
        mode: str,                 # 'fakeavceleb'|'lavdf'|'av_deepfake1m'
        subset: str = "train",
        csv_path: Optional[str] = None,
        json_path: Optional[str] = None,
        # visual controls
        frames_per_clip: int = 25,
        stride: int = 1,
        use_fake_periods: bool = True,
        openface_binary: str = "../OpenFace/build/bin/FeatureExtraction",
        favc_au_cache_dir: str = "../OpenFace_Cache",
        lavdf_au_cache_dir: str = "../OpenFace_Cache_LAVDF",
        avdf1m_au_cache_dir: str = "../OpenFace_Cache_AVDF1M",
        mouth_expand_ratio: float = 0.4,
        flow_bins: int = 12,
        precomputed_dir: Optional[str] = None,
        compute_if_missing: bool = True,
        # visual selection
        feature_set: str = "bins9_11_pack",
        feature_indices: Optional[List[int]] = None,
        selection_json_path: Optional[str] = None,
        selection_key: Optional[str] = None,
        enforce_vis20: bool = True,
        # face crop
        face_detector_weights: str = "yolov8n-face.pt",
        face_img_size: int = 224,
        # temporal face clip
        return_face_seq: bool = False,
        face_seq_len: int = 8,
        face_seq_stride: int = 3,
        # audio params
        audio_sr: int = 16000,
        stft_n_fft: int = 400,
        stft_hop: int = 160,
        stft_win: int = 400,
        n_mels: int = 64,
        n_mfcc: int = 13,
        min_stft_frames: int = 16,
        # misc
        balance_minority: bool = True,
        seed: int = 42,
        silent_missing: bool = True,
        # BACK-COMPAT (no longer controls output): always returns vis20 & vis75
        visual_feat_mode: str = "vis20",
        # failure logging
        fail_log_dir: Optional[str] = None,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode.lower().strip()
        assert self.mode in {"fakeavceleb", "lavdf", "av_deepfake1m"}
        self.subset_req = str(subset).strip().lower()
        self.frames_per_clip = int(frames_per_clip)
        self.stride = max(1, int(stride))
        self.use_fake_periods = bool(use_fake_periods)
        self.openface_binary = openface_binary
        self.favc_au_cache_dir = favc_au_cache_dir
        self.lavdf_au_cache_dir = lavdf_au_cache_dir
        self.avdf1m_au_cache_dir = avdf1m_au_cache_dir
        self.mouth_expand_ratio = float(mouth_expand_ratio)
        self.flow_bins = int(flow_bins)
        self.precomputed_dir = precomputed_dir
        self.compute_if_missing = bool(compute_if_missing)
        self.face_img_size = int(face_img_size)
        self.face_cropper = _FaceCropper(weights_path=face_detector_weights, img_size=self.face_img_size)

        self.return_face_seq = bool(return_face_seq)
        self.face_seq_len = max(1, int(face_seq_len))
        self.face_seq_stride = max(1, int(face_seq_stride))

        self.audio_sr = int(audio_sr)
        self.stft_n_fft = int(stft_n_fft)
        self.stft_hop = int(stft_hop)
        self.stft_win = int(stft_win)
        self.n_mels = int(n_mels)
        self.n_mfcc = int(n_mfcc)
        self.min_stft_frames = int(min_stft_frames)

        self.rng = random.Random(seed)
        self.silent_missing = bool(silent_missing)
        self._warn_count = 0
        self._warn_cap = 100
        self.balance_minority = bool(balance_minority)

        self.visual_feat_mode = visual_feat_mode.lower().strip()  # informational only

        # Failure logging
        self.fail_log_dir = fail_log_dir
        self._fail_log_file = None
        if self.fail_log_dir:
            os.makedirs(self.fail_log_dir, exist_ok=True)
            self._fail_log_file = os.path.join(
                self.fail_log_dir,
                f"fails_{self.subset_req}_{self.mode}_{os.getpid()}.jsonl"
            )

        # Visual feature selection priority: explicit -> JSON -> default
        choice = None
        if feature_indices is not None and len(feature_indices) > 0:
            choice = list(feature_indices)
        if choice is None:
            sel_from_json = _load_vis20_indices(selection_json_path, key=selection_key, mode_hint=self.mode)
            if sel_from_json:
                choice = list(sel_from_json)
        if choice is None:
            choice = list(DEFAULT_VISUAL20_INDICES) if enforce_vis20 else list(RECOMMENDED_SETS.get(feature_set, []))
        if enforce_vis20 and len(choice) > 20:
            choice = choice[:20]
        self.sel_idx = sorted(set(int(i) for i in choice))
        self.Dv20 = len(self.sel_idx)     # expose VIS-20 dim
        self.Dv = self.Dv20               # back-compat alias
        self.Dv75 = 75

        if self.precomputed_dir:
            os.makedirs(self.precomputed_dir, exist_ok=True)

        desired = _desired_splits(self.mode, self.subset_req)
        folder_hints = _folder_aliases_for_files(self.mode, self.subset_req)

        # samples: List[(path, y_mm, segs, y_a|None, y_v|None)]
        base_real: List[Tuple[str, int, Optional[List[List[float]]], Optional[int], Optional[int]]] = []
        base_fake: List[Tuple[str, int, Optional[List[List[float]]], Optional[int], Optional[int]]] = []
        missing_debug = []

        if self.mode == "fakeavceleb":
            if not csv_path:
                raise ValueError("csv_path required for FakeAVCeleb")
            df = _read_favc_csv(csv_path)
            df = df[df["split"].isin(desired)]
            for _, row in df.iterrows():
                p = str(row["file_path"]).strip()
                lab = int(row["label"])
                abs_path = _resolve_visual_path(self.root_dir, p, subset_hints=folder_hints)
                if not abs_path or not os.path.isfile(abs_path):
                    if len(missing_debug) < 25:
                        missing_debug.append(p)
                    if not self.silent_missing:
                        warnings.warn(f"[{self.subset_req}] missing media: {p}")
                    continue
                (base_real if lab == 0 else base_fake).append((abs_path, lab, None, None, None))

        elif self.mode == "lavdf":
            if not json_path:
                raise ValueError("json_path required for LAV-DF")
            for entry in _lavdf_iter_entries(json_path):
                if entry["split"] not in desired:
                    continue
                rel = entry["file"]
                abs_path = rel if os.path.isabs(rel) else os.path.join(self.root_dir, rel)
                if not os.path.isfile(abs_path):
                    if len(missing_debug) < 25:
                        missing_debug.append(rel)
                    if not self.silent_missing:
                        warnings.warn(f"[{self.subset_req}] missing: {abs_path}")
                    continue
                y_a = 1 if entry.get("modify_audio", False) else 0
                y_v = 1 if entry.get("modify_video", False) else 0
                lab = 1 if (y_a or y_v) else 0
                segs = entry.get("fake_periods", []) or []
                (base_real if lab == 0 else base_fake).append((abs_path, lab, segs if segs else None, y_a, y_v))

        else:  # av_deepfake1m
            if not json_path:
                raise ValueError("json_path required for AV-Deepfake1M")
            for entry in _avdf1m_iter_entries(json_path):
                if entry["split"] not in desired:
                    continue
                rel = entry["file"]
                abs_path = rel if os.path.isabs(rel) else os.path.join(self.root_dir, rel)
                if not os.path.isfile(abs_path):
                    if len(missing_debug) < 25:
                        missing_debug.append(rel)
                    if not self.silent_missing:
                        warnings.warn(f"[{self.subset_req}] missing: {abs_path}")
                    continue
                mt = (entry.get("modify_type", "") or "").strip().lower()
                y_a = 1 if mt in {"audio_modified", "both_modified"} else 0
                y_v = 1 if mt in {"visual_modified", "both_modified"} else 0
                lab = 1 if (y_a or y_v) else 0
                segs = entry.get("fake_segments", []) or []
                (base_real if lab == 0 else base_fake).append((abs_path, lab, segs if segs else None, y_a, y_v))

        if (len(base_real) + len(base_fake)) == 0:
            msg = (f"No samples for mode={self.mode}, subset={self.subset_req} (accepted={desired}) under root={self.root_dir}.")
            if missing_debug:
                msg += "\nExamples (first 25):\n  - " + "\n  - ".join(missing_debug)
            raise RuntimeError(msg)

        samples: List[Tuple[str, int, Optional[List[List[float]]], Optional[int], Optional[int]]] = base_real + base_fake
        if self.balance_minority:
            n0, n1 = len(base_real), len(base_fake)
            if n0 != n1:
                minority = base_fake if n0 > n1 else base_real
                gap = abs(n0 - n1)
                for _ in range(gap):
                    src = self.rng.choice(minority)
                    segs = [s[:] for s in (src[2] or [])] if src[2] else None
                    # keep per-modality labels aligned
                    samples.append((src[0], src[1], segs, src[3], src[4]))

        self.samples = samples
        n_real = sum(1 for _, lab, _, _, _ in self.samples if lab == 0)
        n_fake = len(self.samples) - n_real
        print(f"[{self.subset_req.upper()}][{self.mode}] kept={len(self.samples)} "
              f"(Real={n_real}, Fake={n_fake}) | balance_minority={self.balance_minority} | use_fake_periods={self.use_fake_periods}")
        print(f"Selected VIS-20 dims: Dv20={self.Dv20} -> idx={self.sel_idx} | visual_feat_mode={self.visual_feat_mode} (output=vis20+vis75)")

    # --------- helpers ---------
    def _log_fail(self, video_path: str, code: str, extra: Optional[Dict[str, Any]] = None):
        if not self._fail_log_file:
            return
        rec = {
            "ts": time.time(),
            "subset": self.subset_req,
            "mode": self.mode,
            "video": video_path,
            "reason": code,
        }
        if extra:
            rec.update(extra)
        try:
            with open(self._fail_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _of_cache_dir(self) -> str:
        if self.mode == "lavdf":
            return self.lavdf_au_cache_dir
        if self.mode == "av_deepfake1m":
            return self.avdf1m_au_cache_dir
        return self.favc_au_cache_dir

    def _v75_npy_path(self, video_path: str) -> Optional[str]:
        if not self.precomputed_dir:
            return None
        base = os.path.splitext(os.path.basename(video_path))[0]
        return os.path.join(self.precomputed_dir, base + ".v75.npy")

    def _select_vis20(self, v75: np.ndarray) -> Optional[np.ndarray]:
        if v75 is None or v75.size != 75 or not np.all(np.isfinite(v75)):
            return None
        return v75[self.sel_idx].astype(np.float32)

    def _prepare_frame_window(self, video_path: str, label: int, segments_sec: Optional[List[List[float]]]) -> Tuple[List[int], float]:
        total = _video_len(video_path)
        if total <= 0:
            return [], 25.0
        fps = _fps(video_path)
        eff = max(1, self.stride)
        need = (self.frames_per_clip - 1) * eff + 1

        if self.use_fake_periods and label == 1 and segments_sec:
            viable = []
            longest = None; longest_len = -1
            for s_sec, e_sec in segments_sec:
                a = max(int(round(s_sec * fps)), 0)
                b = min(int(round(e_sec * fps)), total - 1)
                if b > a:
                    L = b - a + 1
                    if L > longest_len:
                        longest, longest_len = (a, b), L
                    if L >= need:
                        viable.append((a, b))
            # Use a viable (long enough) span
            if viable:
                a, b = self.rng.choice(viable)
                start = self.rng.randint(a, b - need + 1)
                end = start + (need - 1)
                return list(range(start, end + 1, eff)), fps
            # If longest exists but is short, tile it to reach 'need'
            if longest is not None:
                a, b = longest
                idxs = _tile_span_indices(a, b, need, eff)
                return idxs, fps

        # Fallback: take from the start of the video
        start = 0; end = min(total - 1, start + need - 1)
        return list(range(start, end + 1, eff)), fps

    def _extract_face_at(self, video_path: str, frame_index: int,
                         of_row: Optional[int], of_df: Optional[pd.DataFrame]) -> Optional[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return None
        crop = self.face_cropper.detect_and_crop(frame)
        if crop is not None:
            return crop
        if of_df is not None and of_row is not None and 0 <= of_row < len(of_df):
            lms, _ = _collect_frame_landmarks(of_df)
            if lms is not None and of_row < len(lms):
                lm = lms[of_row]
                if np.all(np.isfinite(lm)):
                    acrop = _align_crop_from_landmarks(frame, lm, out_size=self.face_img_size)
                    if acrop is not None:
                        return acrop
        return _fallback_center_crop(frame, out_size=self.face_img_size)

    def _warn(self, msg: str):
        # Rate-limit warnings unless silent_missing=False
        if not self.silent_missing or self._warn_count < self._warn_cap:
            warnings.warn(msg)
            self._warn_count += 1

    # --------- dataset API ---------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            # Unpack: (path, y_mm, segs, y_a|None, y_v|None)
            video_path, label, segments_sec, y_a, y_v = self.samples[idx]
            label = int(label)

            # Quick container check to prevent noisy FFmpeg errors & skip broken files
            if not _ffprobe_quick_check(video_path):
                self._log_fail(video_path, "ffprobe_failed_bad_container")
                return None

            # Reject unreadable
            total = _video_len(video_path)
            if total <= 0:
                self._log_fail(video_path, "unreadable_video_total_frames_0")
                self._warn(f"[skip:unreadable_video] {video_path}")
                return None

            # 1) Visual window indices
            frame_idx, fps = self._prepare_frame_window(video_path, label, segments_sec)
            if not frame_idx:
                self._log_fail(video_path, "no_frames_after_selection",
                               {"label": int(label), "segments_sec": segments_sec})
                self._warn(f"[skip:no_frames] {video_path}")
                return None
            start_f = frame_idx[0]; end_f = frame_idx[-1]

            # 2) OpenFace cache + rows mapping
            csv_path = _ensure_openface_csv(video_path, self._of_cache_dir(), self.openface_binary)
            of_df = _read_openface_csv(csv_path) if csv_path else None
            of_rows = _map_frames_to_of_rows(of_df, frame_idx, fps)

            # 3) Visual-75 (precomputed or compute) — must be valid; otherwise SKIP
            v75 = None
            npy_path = self._v75_npy_path(video_path)
            if npy_path and os.path.isfile(npy_path):
                try:
                    v75 = np.load(npy_path).astype(np.float32)
                except Exception:
                    v75 = None
            else:
                if self.compute_if_missing and csv_path is not None:
                    v75 = _visual75_from_frames_and_openface(
                        video_path=video_path,
                        frame_indices=frame_idx,
                        fps_video=fps,
                        openface_cache_csv=csv_path,
                        mouth_expand_ratio=self.mouth_expand_ratio,
                        flow_bins=self.flow_bins,
                    )
                    if v75 is not None and npy_path is not None:
                        try:
                            np.save(npy_path, v75)
                        except Exception:
                            pass

            # Validate VIS-75
            if v75 is None or v75.shape != (75,) or not np.all(np.isfinite(v75)):
                self._log_fail(video_path, "visual75_failure",
                               {"has_csv": bool(csv_path), "csv_path": csv_path,
                                "precomputed_exists": bool(npy_path and os.path.isfile(npy_path))})
                self._warn(f"[skip:visual75_failure] {video_path} | reason="
                           f"{'no_precomputed' if (npy_path and not os.path.isfile(npy_path)) else 'bad_or_missing_openface_or_invalid_v75'}")
                return None

            x_vis75_np = v75
            x_vis20_np = self._select_vis20(x_vis75_np)
            if x_vis20_np is None or x_vis20_np.shape[0] != self.Dv20 or not np.all(np.isfinite(x_vis20_np)):
                self._log_fail(video_path, "vis20_selection_failure", {"idx": self.sel_idx})
                self._warn(f"[skip:vis20_selection_failure] {video_path}")
                return None

            x_vis75_t = torch.from_numpy(x_vis75_np).to(torch.float32)          # [75]
            x_vis20_t = torch.from_numpy(x_vis20_np).to(torch.float32)          # [Dv20]

            # 4) Face (single mid frame or temporal clip)
            if not self.return_face_seq:
                mid_i = frame_idx[len(frame_idx) // 2]
                mid_row = of_rows[len(of_rows) // 2] if len(of_rows) > 0 else None
                face_bgr = self._extract_face_at(video_path, mid_i, mid_row, of_df)
                if face_bgr is None:
                    self._log_fail(video_path, "face_crop_failure_single")
                    self._warn(f"[skip:face_failure] {video_path}")
                    return None
                face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                face_out = torch.from_numpy(face_rgb.transpose(2, 0, 1)).to(torch.float32) / 255.0  # [3,H,W]
            else:
                seq_idx: List[int] = []
                step = int(self.face_seq_stride) if self.face_seq_stride > 0 else 1
                for i in range(0, len(frame_idx), step):
                    seq_idx.append(frame_idx[i])
                if len(seq_idx) < self.face_seq_len:
                    while len(seq_idx) < self.face_seq_len:
                        seq_idx.append(frame_idx[-1])
                if len(seq_idx) > self.face_seq_len:
                    extra = len(seq_idx) - self.face_seq_len
                    drop_l = extra // 2
                    drop_r = extra - drop_l
                    seq_idx = seq_idx[drop_l: len(seq_idx) - drop_r]
                of_rows_seq = _map_frames_to_of_rows(of_df, seq_idx, fps)
                faces_list: List[torch.Tensor] = []
                for fi, ofr in zip(seq_idx, of_rows_seq):
                    crop_bgr = self._extract_face_at(video_path, fi, ofr, of_df)
                    if crop_bgr is None:
                        self._log_fail(video_path, "face_crop_failure_seq", {"fi": int(fi)})
                        self._warn(f"[skip:face_seq_failure] {video_path}")
                        return None
                    face_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                    t = torch.from_numpy(face_rgb.transpose(2, 0, 1)).to(torch.float32) / 255.0
                    faces_list.append(t)
                face_out = torch.stack(faces_list, dim=0)  # [T,3,H,W]

            # 5) Audio strictly aligned to visual window
            a_start_sec = float(start_f) / max(fps, 1e-6)
            a_end_sec   = float(end_f + 1) / max(fps, 1e-6)

            audio = _load_audio_mono_16k(video_path, target_sr=self.audio_sr)
            if audio is None or audio.size == 0:
                # sidecar audio try
                base = os.path.splitext(video_path)[0]
                found = None
                for ext in AUDIO_EXTS:
                    cand = base + ext
                    if os.path.isfile(cand):
                        found = cand; break
                audio = _load_audio_mono_16k(found, target_sr=self.audio_sr) if found else None
            if audio is None or audio.size == 0:
                self._log_fail(video_path, "audio_missing")
                self._warn(f"[skip:audio_missing] {video_path}")
                return None

            a_start = max(0, int(round(a_start_sec * self.audio_sr)))
            a_end   = min(audio.shape[0], int(round(a_end_sec * self.audio_sr)))
            if a_end <= a_start:
                approx_samples = int(round((self.frames_per_clip / max(fps, 1e-6)) * self.audio_sr))
                a_end = min(audio.shape[0], a_start + approx_samples)
            if a_end <= a_start:
                self._log_fail(video_path, "audio_window_invalid", {"a_start": a_start, "a_end": a_end, "fps": float(fps)})
                self._warn(f"[skip:audio_window_invalid] {video_path}")
                return None
            seg = audio[a_start:a_end]
            if seg.size <= 16:
                self._log_fail(video_path, "audio_too_short", {"seg_len": int(seg.size)})
                self._warn(f"[skip:audio_too_short] {video_path}")
                return None

            audio75, stft, a_len = _audio_feature_pack(
                seg, self.audio_sr,
                self.stft_n_fft, self.stft_hop, self.stft_win,
                self.n_mels, self.n_mfcc
            )

            if audio75.shape != (75,) or not np.all(np.isfinite(audio75)):
                self._log_fail(video_path, "audio75_invalid")
                self._warn(f"[skip:audio75_invalid] {video_path}")
                return None
            if stft.ndim != 2 or stft.shape[0] <= 0 or stft.shape[1] <= 0 or not np.all(np.isfinite(stft)):
                self._log_fail(video_path, "stft_invalid", {"shape": list(stft.shape)})
                self._warn(f"[skip:stft_invalid] {video_path}")
                return None

            stft_t    = torch.from_numpy(stft).to(torch.float32)    # [F,S]
            audio75_t = torch.from_numpy(audio75).to(torch.float32) # [75]
            label_mm_t   = torch.tensor(label, dtype=torch.long)
            label_a_t = torch.tensor(int(y_a), dtype=torch.long) if y_a is not None else None
            label_v_t = torch.tensor(int(y_v), dtype=torch.long) if y_v is not None else None

            # RETURN
            return x_vis20_t, x_vis75_t, face_out, stft_t, audio75_t, label_mm_t, label_a_t, label_v_t, int(a_len), video_path

        except Exception as e:
            self._log_fail(self.samples[idx][0] if idx < len(self.samples) else "unknown",
                           "exception", {"etype": type(e).__name__, "msg": str(e)})
            self._warn(f"[skip:exception] {self.samples[idx][0] if idx < len(self.samples) else 'unknown'} | {type(e).__name__}: {e}")
            return None

# =========================
# Collate + Loader
# =========================
def _pad_time(batch_tensors: List[torch.Tensor]):
    lengths = [t.size(0) for t in batch_tensors]
    Fmax = max(lengths); D = batch_tensors[0].size(1)
    out = torch.zeros((len(batch_tensors), Fmax, D), dtype=torch.float32)
    for i, t in enumerate(batch_tensors):
        out[i, :t.size(0)] = t
    return out, torch.tensor(lengths, dtype=torch.long)

def _stack_optional(labels_list: List[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
    # If any is None, return None; else stack long tensor
    if any(l is None for l in labels_list):
        return None
    return torch.stack([l if torch.is_tensor(l) else torch.tensor(l) for l in labels_list]).long()

def collate_unified_av(batch):
    """
    Drops None samples. Returns:
      x_vis20[B,Dv20], x_vis75[B,75],
      face[B,3,H,W] or face[B,T,3,H,W],
      stft[B,Fmax,S], x_aud[B,75],
      labels_mm[B], labels_a[B]|None, labels_v[B]|None,
      a_len[B], paths[list]
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    # Unpack the tuples
    x_vis20, x_vis75, faces, stfts, x_aud, labels_mm, labels_a, labels_v, a_lens, paths = zip(*batch)

    x_vis20 = torch.stack(list(x_vis20), dim=0)  # [B,Dv20]
    x_vis75 = torch.stack(list(x_vis75), dim=0)  # [B,75]

    faces0 = faces[0]
    if torch.is_tensor(faces0) and faces0.ndim == 3:
        faces = torch.stack(list(faces), dim=0)  # [B,3,H,W]
    elif torch.is_tensor(faces0) and faces0.ndim == 4:
        faces = torch.stack(list(faces), dim=0)  # [B,T,3,H,W]
    else:
        raise ValueError(f"Unexpected face tensor shape: {None if faces0 is None else tuple(faces0.shape)}")

    stft_pad, a_lengths = _pad_time(list(stfts))
    x_aud   = torch.stack(list(x_aud), dim=0)
    labels_mm = torch.stack([torch.tensor(l) if not torch.is_tensor(l) else l for l in labels_mm]).long()

    labels_a = _stack_optional(list(labels_a))
    labels_v = _stack_optional(list(labels_v))

    return x_vis20, x_vis75, faces, stft_pad, x_aud, labels_mm, labels_a, labels_v, a_lengths, list(paths)

class ProgressDataLoader(DataLoader):
    def __init__(self, *args, show_tqdm: bool = False, desc: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self._show_tqdm = bool(show_tqdm)
        self._tqdm_desc = str(desc) if desc else "dataloader"

    def __iter__(self):
        it = super().__iter__()
        if self._show_tqdm and (sys.stdout.isatty() or sys.stderr.isatty()):
            for batch in tqdm(it, total=len(self), desc=self._tqdm_desc, ncols=100, leave=False):
                if batch is None:
                    continue
                yield batch
            return
        for batch in it:
            if batch is None:
                continue
            yield batch

def get_unified_av_dataloader(
    *,
    root_dir: str,
    mode: str,                         # 'fakeavceleb'|'lavdf'|'av_deepfake1m'
    subset: str,
    csv_path: Optional[str] = None,
    json_path: Optional[str] = None,
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: int = 42,
    # visual
    frames_per_clip: int = 25,
    stride: int = 1,
    use_fake_periods: bool = True,
    openface_binary: str = "",
    favc_au_cache_dir: str = "../OpenFace_Cache",
    lavdf_au_cache_dir: str = "../OpenFace_Cache_LAVDF",
    avdf1m_au_cache_dir: str = "../OpenFace_Cache_AVDF1M",
    mouth_expand_ratio: float = 0.4,
    flow_bins: int = 12,
    precomputed_dir: Optional[str] = None,
    compute_if_missing: bool = False,
    feature_set: str = "bins9_11_pack",
    feature_indices: Optional[List[int]] = None,
    # selection JSON (to match training VIS-20)
    selection_json_path: Optional[str] = None,
    selection_key: Optional[str] = None,
    enforce_vis20: bool = True,
    # face
    face_detector_weights: str = "yolov8n-face.pt",
    face_img_size: int = 224,
    # temporal faces
    return_face_seq: bool = False,
    face_seq_len: int = 8,
    face_seq_stride: int = 3,
    # audio
    audio_sr: int = 16000,
    stft_n_fft: int = 400,
    stft_hop: int = 160,
    stft_win: int = 400,
    n_mels: int = 64,
    n_mfcc: int = 13,
    min_stft_frames: int = 16,
    # misc
    balance_minority: bool = True,
    pin_memory: bool = True,
    show_tqdm: bool = True,
    # BACK-COMPAT: this is now informational; loader always returns both vis20 & vis75
    visual_feat_mode: str = "vis20",
    # failure logging
    fail_log_dir: Optional[str] = None,
):
    ds = UnifiedAVDataset(
        root_dir=root_dir, mode=mode, subset=subset,
        csv_path=csv_path, json_path=json_path,
        frames_per_clip=frames_per_clip, stride=stride, use_fake_periods=use_fake_periods,
        openface_binary=openface_binary,
        favc_au_cache_dir=favc_au_cache_dir, lavdf_au_cache_dir=lavdf_au_cache_dir, avdf1m_au_cache_dir=avdf1m_au_cache_dir,
        mouth_expand_ratio=mouth_expand_ratio, flow_bins=flow_bins,
        precomputed_dir=precomputed_dir, compute_if_missing=compute_if_missing,
        feature_set=feature_set, feature_indices=feature_indices,
        selection_json_path=selection_json_path, selection_key=selection_key, enforce_vis20=enforce_vis20,
        face_detector_weights=face_detector_weights, face_img_size=face_img_size,
        return_face_seq=return_face_seq, face_seq_len=face_seq_len, face_seq_stride=face_seq_stride,
        audio_sr=audio_sr, stft_n_fft=stft_n_fft, stft_hop=stft_hop, stft_win=stft_win,
        n_mels=n_mels, n_mfcc=n_mfcc, min_stft_frames=min_stft_frames,
        balance_minority=balance_minority, seed=seed,
        silent_missing=True,
        visual_feat_mode=visual_feat_mode,
        fail_log_dir=fail_log_dir,
    )
    desc = (f"{subset}/{mode} A/V-aligned (VIS-20+VIS-75"
            f", face{face_img_size}"
            + (f", T={face_seq_len}@{face_seq_stride}" if return_face_seq else "")
            + f", sr={audio_sr}, clipT={frames_per_clip})")
    return ProgressDataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_unified_av,
        drop_last=False,
        show_tqdm=show_tqdm,
        desc=desc,
    )


#-------------------------------------------------------------------------------------------------------------------------------#
# import os
# import time
# import json
# import glob
# import random
# import warnings
# import subprocess
# import tempfile
# from typing import List, Tuple, Optional, Dict, Any

# import cv2
# import numpy as np
# import pandas as pd
# import torch
# from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm
# import sys

# # === NEW
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from pathlib import Path
# import hashlib
# import shutil

# # --------------------------
# # Silence OpenCV/FFmpeg logging (e.g., "moov atom not found")
# # --------------------------
# try:
#     cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)  # OpenCV >= 4.5
# except Exception:
#     pass
# os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

# # --------------------------
# # GPU preference (optional)
# # --------------------------
# if torch.cuda.is_available():
#     try:
#         torch.cuda.set_device(0)
#     except Exception:
#         pass

# # =========================
# # VISUAL-75 presets & default VIS-20 (indices)
# # =========================
# RECOMMENDED_SETS = {
#     "one": [39],
#     "small": [39, 40, 53, 36, 65, 37, 33, 34, 0, 4],
#     "bins9_11_pack": [33, 34, 36, 37, 39, 40, 51, 52, 53, 63, 64, 65, 0, 4],
# }
# DEFAULT_VISUAL20_INDICES = [70, 72, 71, 73, 66, 68, 67, 69, 74, 65, 3, 64, 20, 41, 53, 14, 17, 52, 23, 8]
# DEFAULT_SELECTION_JSON = "visual75_crossdomain_selection.json"

# # =========================
# # Split alias helpers
# # =========================
# def _desired_splits(mode: str, subset_req: str) -> List[str]:
#     s = subset_req.lower().strip()
#     if mode == "av_deepfake1m":
#         if s in {"eval", "val"}:
#             return ["val", "eval"]
#         return [s]
#     if mode == "lavdf":
#         if s == "eval":
#             return ["dev"]
#         return [s]
#     return [s]

# def _folder_aliases_for_files(mode: str, subset_req: str) -> List[str]:
#     s = subset_req.lower().strip()
#     if mode == "av_deepfake1m":
#         if s in {"eval", "val"}:
#             return ["val", "eval", "validation"]
#         return [s]
#     if mode == "lavdf":
#         if s == "eval":
#             return ["dev", "eval", "validation"]
#         return [s]
#     return [s]

# # =========================
# # Selection JSON helpers (VIS-20)
# # =========================
# def _normalize_selection_indices(obj: Any) -> Optional[List[int]]:
#     if obj is None:
#         return None
#     if isinstance(obj, list) and all(isinstance(x, (int, np.integer)) for x in obj):
#         return sorted(set(int(x) for x in obj))
#     if isinstance(obj, dict):
#         for k in ["indices", "selected_indices", "vis20", "visual20", "top20", "selection"]:
#             if k in obj:
#                 vals = obj[k]
#                 if isinstance(vals, list) and all(isinstance(x, (int, np.integer)) for x in vals):
#                     return sorted(set(int(x) for x in vals))
#     return None

# def _load_vis20_indices(json_path: Optional[str], key: Optional[str] = None, mode_hint: Optional[str] = None) -> Optional[List[int]]:
#     path = json_path or DEFAULT_SELECTION_JSON
#     if not (isinstance(path, str) and os.path.isfile(path)):
#         return None
#     try:
#         with open(path, "r") as f:
#             data = json.load(f)
#     except Exception:
#         return None

#     sel = _normalize_selection_indices(data)
#     if sel is not None:
#         return sel

#     if isinstance(data, dict):
#         if key and key in data:
#             sel = _normalize_selection_indices(data[key])
#             if sel is not None:
#                 return sel
#         if mode_hint:
#             if mode_hint in data:
#                 sel = _normalize_selection_indices(data[mode_hint])
#                 if sel is not None:
#                     return sel
#             aliases = {
#                 "av_deepfake1m": ["avdf1m", "av-deepfake1m", "av_df1m", "avdf_1m", "avdf"],
#                 "fakeavceleb": ["favc", "fakeav", "fake-av-celeb"],
#                 "lavdf": ["lav-df", "lav_df", "lav"],
#             }.get(mode_hint, [])
#             for a in aliases:
#                 if a in data:
#                     sel = _normalize_selection_indices(data[a])
#                     if sel is not None:
#                         return sel
#         for v in data.values():
#             sel = _normalize_selection_indices(v)
#             if sel is not None:
#                 return sel
#     return None

# # =========================
# # Path helpers
# # =========================
# VIDEO_EXTS = [".mp4", ".mov", ".avi", ".mkv", ".webm",
#               ".MP4", ".MOV", ".AVI", ".MKV", ".WEBM"]
# AUDIO_EXTS = [".wav", ".mp3", ".m4a", ".flac", ".aac",
#               ".WAV", ".MP3", ".M4A", ".FLAC", ".AAC"]

# SUBFOLDERS_GUESS = [
#     "eval", "val", "validation", "dev", "test", "train",
#     "Eval", "Val", "Validation", "Dev", "Test", "Train",
#     "evaluation", "Evaluation", "VALIDATION", "EVAL"
# ]

# def _maybe_swap_ext_to_video(path: str) -> List[str]:
#     cand = []
#     bn, ext = os.path.splitext(path)
#     if ext in AUDIO_EXTS or ext == "":
#         for vext in VIDEO_EXTS:
#             cand.append(bn + vext)
#     return cand

# def _glob_one(pattern: str) -> Optional[str]:
#     for p in glob.iglob(pattern, recursive=True):
#         if os.path.isfile(p):
#             return p
#     return None

# def _find_case_insensitive(root_dir: str, base_noext: str) -> Optional[str]:
#     base_lower = base_noext.lower()
#     for ext in VIDEO_EXTS:
#         pattern = os.path.join(root_dir, "**", "*" + ext)
#         for p in glob.iglob(pattern, recursive=True):
#             try:
#                 if os.path.isfile(p):
#                     if os.path.splitext(os.path.basename(p))[0].lower() == base_lower:
#                         return p
#             except Exception:
#                 continue
#     return None

# def _find_video_by_basename(root_dir: str, base_noext: str) -> Optional[str]:
#     for sub in SUBFOLDERS_GUESS:
#         for ext in VIDEO_EXTS:
#             p = os.path.join(root_dir, sub, base_noext + ext)
#             if os.path.isfile(p):
#                 return p
#     p = _find_case_insensitive(root_dir, base_noext)
#     if p:
#         return p
#     for ext in VIDEO_EXTS:
#         pattern = os.path.join(root_dir, "**", base_noext + ext)
#         p = _glob_one(pattern)
#         if p:
#             return p
#     return None

# def _resolve_visual_path(root_dir: str, csv_fp: str, subset_hints: List[str]) -> Optional[str]:
#     if os.path.isabs(csv_fp) and os.path.isfile(csv_fp) and os.path.splitext(csv_fp)[1] in VIDEO_EXTS:
#         return csv_fp
#     cand_abs = os.path.join(root_dir, csv_fp) if not os.path.isabs(csv_fp) else csv_fp
#     if os.path.isfile(cand_abs) and os.path.splitext(cand_abs)[1] in VIDEO_EXTS:
#         return cand_abs
#     for p2 in _maybe_swap_ext_to_video(cand_abs):
#         if os.path.isfile(p2):
#             return p2
#     base = os.path.splitext(os.path.basename(csv_fp))[0]
#     for hint in subset_hints:
#         for ext in VIDEO_EXTS:
#             p = os.path.join(root_dir, hint, base + ext)
#             if os.path.isfile(p):
#                 return p
#     return _find_video_by_basename(root_dir, base)

# # =========================
# # Dataset readers
# # =========================
# def _read_favc_csv(csv_path: str) -> pd.DataFrame:
#     df = pd.read_csv(csv_path)
#     need = {"split", "file_path", "label"}
#     if not need.issubset({c.lower() for c in df.columns}):
#         raise ValueError("CSV must contain columns: split, file_path, label")
#     cols = {c.lower(): c for c in df.columns}
#     out = df[[cols["split"], cols["file_path"], cols["label"]]].copy()
#     out.columns = ["split", "file_path", "label"]
#     out["split"] = out["split"].astype(str).str.strip().str.lower()

#     def _lab2int(x):
#         if isinstance(x, str):
#             xs = x.strip().lower()
#             if xs in ("real", "0"):
#                 return 0
#             if xs in ("fake", "1"):
#                 return 1
#         try:
#             xi = int(x)
#             return 1 if xi == 1 else 0
#         except Exception:
#             pass
#         raise ValueError(f"Unrecognized label value: {x}")

#     out["label"] = out["label"].apply(_lab2int).astype(int)
#     out["file_path"] = out["file_path"].astype(str).str.strip()
#     return out

# def _lavdf_iter_entries(json_path: str):
#     with open(json_path, "r") as f:
#         data = json.load(f)
#     items = list(data.values()) if isinstance(data, dict) else list(data)
#     for it in items:
#         file_field = (it.get("file") or it.get("path") or "").strip()
#         if not file_field:
#             continue
#         split = str(it.get("split", "")).strip().lower()
#         if split not in {"train", "test", "dev", "eval", "validation", "val"}:
#             split = "train"
#         modify_audio = bool(it.get("modify_audio", False))
#         modify_video = bool(it.get("modify_video", False))
#         periods = it.get("fake_periods", []) or []
#         periods_clean = []
#         for p in periods:
#             if isinstance(p, (list, tuple)) and len(p) == 2:
#                 try:
#                     s = float(p[0]); e = float(p[1])
#                     if e > s: periods_clean.append([s, e])
#                 except Exception:
#                     pass
#         yield {
#             "file": file_field,
#             "split": split,
#             "modify_audio": modify_audio,
#             "modify_video": modify_video,
#             "fake_periods": periods_clean,
#         }

# def _avdf1m_iter_entries(json_path: str):
#     with open(json_path, "r") as f:
#         data = json.load(f)
#     items = list(data.values()) if isinstance(data, dict) else list(data)
#     for it in items:
#         file_field = (it.get("file") or it.get("path") or "").strip()
#         if not file_field:
#             continue
#         split = (it.get("split") or "val").strip().lower()
#         modify_type = (it.get("modify_type") or "").strip().lower()
#         segs = it.get("fake_segments", []) or []
#         segs_clean = []
#         for p in segs:
#             if isinstance(p, (list, tuple)) and len(p) == 2:
#                 try:
#                     s = float(p[0]); e = float(p[1])
#                     if e > s: segs_clean.append([s, e])
#                 except Exception:
#                     pass
#         yield {
#             "file": file_field,
#             "split": split,
#             "modify_type": modify_type,
#             "fake_segments": segs_clean,
#         }

# # =========================
# # OpenFace helpers
# # =========================
# EYE_L = list(range(36, 42))
# EYE_R = list(range(42, 48))
# MOUTH_OUT = list(range(48, 60))

# # === NEW: stable ID for state tracking
# def _stable_id_from_path(p: str) -> str:
#     return hashlib.md5(p.encode("utf-8", "ignore")).hexdigest()

# # === NEW: cleanup any OpenFace artifacts for a video (csv, hog, txt, etc.)
# def _cleanup_openface_artifacts(cache_dir: str, video_path: str):
#     try:
#         base = os.path.splitext(os.path.basename(video_path))[0]
#         stem = os.path.join(cache_dir, base)
#         for ext in [".csv", ".hog", ".txt"]:
#             fp = stem + ext
#             if os.path.isfile(fp):
#                 try:
#                     os.remove(fp)
#                 except Exception:
#                     pass
#         # Also clean any per-frame dumps OpenFace may write into a subdir
#         subdir = os.path.join(cache_dir, base)
#         if os.path.isdir(subdir):
#             try:
#                 shutil.rmtree(subdir, ignore_errors=True)
#             except Exception:
#                 pass
#     except Exception:
#         pass

# def _ensure_openface_csv(video_path: str, cache_dir: str, openface_binary: str) -> Tuple[Optional[str], bool]:
#     """
#     Returns (csv_path, created_now).
#     created_now=True if we ran OpenFace in this call.
#     """
#     os.makedirs(cache_dir or ".", exist_ok=True)
#     base = os.path.splitext(os.path.basename(video_path))[0]
#     csv_path = os.path.join(cache_dir, f"{base}.csv")
#     if os.path.isfile(csv_path):
#         return csv_path, False
#     if not openface_binary:
#         return None, False
#     try:
#         cmd = [openface_binary, "-f", video_path, "-aus", "-2Dfp", "-out_dir", cache_dir]
#         subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         return (csv_path if os.path.isfile(csv_path) else None), True
#     except Exception:
#         return None, False

# def _read_openface_csv(csv_path: str) -> Optional[pd.DataFrame]:
#     for enc in (None, "utf-8-sig", "latin1"):
#         try:
#             df = pd.read_csv(csv_path, encoding=enc)
#             df.columns = df.columns.str.strip()
#             return df
#         except Exception:
#             continue
#     return None

# def _collect_frame_landmarks(df: pd.DataFrame):
#     try:
#         lm_cols = [f"{ax}_{i}" for i in range(68) for ax in ("x", "y")]
#         if not set(lm_cols).issubset(df.columns):
#             return None, None
#         lms = df[lm_cols].values.reshape((-1, 68, 2)).astype(np.float32)
#         success = df["success"].astype(np.float32).values if "success" in df.columns else np.ones((len(df),), dtype=np.float32)
#         return lms, success
#     except Exception:
#         return None, None

# def _map_frames_to_of_rows(df: Optional[pd.DataFrame], frame_indices: List[int], fps_video: float) -> List[int]:
#     if df is None:
#         return [0 for _ in frame_indices]
#     n = len(df)
#     if n == 0:
#         return [0 for _ in frame_indices]
#     if "timestamp" in df.columns:
#         ts = df["timestamp"].to_numpy(dtype=np.float64)
#         rows = []
#         for fidx in frame_indices:
#             t = float(fidx) / max(fps_video, 1e-6)
#             j = int(np.searchsorted(ts, t, side="left"))
#             if j == n:
#                 j = n - 1
#             elif j > 0 and (t - ts[j - 1]) < (ts[j] - t):
#                 j -= 1
#             rows.append(max(min(j, n - 1), 0))
#         return rows
#     if "frame" in df.columns:
#         of_frames = df["frame"].to_numpy(dtype=np.int64)
#         return [int(np.argmin(np.abs(of_frames - fidx))) for fidx in frame_indices]
#     return [max(min(int(f), n - 1), 0) for f in frame_indices]

# # === NEW: OpenFace multithread precompute for any dataset/list
# def _ensure_openface_csv_batch(
#     video_paths: List[str],
#     cache_dir: str,
#     openface_binary: str,
#     max_workers: int = 8,
#     show_progress: bool = True,
# ) -> Dict[str, bool]:
#     """
#     Ensure CSV for each video via thread pool.
#     Returns {video_path: created_now_bool}.
#     """
#     if not openface_binary or not os.path.isfile(openface_binary):
#         return {vp: False for vp in video_paths}
#     os.makedirs(cache_dir or ".", exist_ok=True)

#     def _task(vp):
#         try:
#             base = os.path.splitext(os.path.basename(vp))[0]
#             out_csv = os.path.join(cache_dir, f"{base}.csv")
#             if os.path.isfile(out_csv):
#                 return vp, False
#             cmd = [openface_binary, "-f", vp, "-aus", "-2Dfp", "-out_dir", cache_dir]
#             subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#             return vp, os.path.isfile(out_csv)
#         except Exception:
#             return vp, False

#     futs = []
#     created = {}
#     with ThreadPoolExecutor(max_workers=max_workers) as ex:
#         for vp in video_paths:
#             futs.append(ex.submit(_task, vp))
#         if show_progress and (sys.stdout.isatty() or sys.stderr.isatty()):
#             for f in tqdm(as_completed(futs), total=len(futs), desc="OpenFace precompute", ncols=100, leave=False):
#                 vp, ok = f.result()
#                 created[vp] = ok
#         else:
#             for f in as_completed(futs):
#                 vp, ok = f.result()
#                 created[vp] = ok
#     return created

# # =========================
# # Visual-75 utilities
# # =========================
# def _summary_stats(x: np.ndarray) -> Dict[str, float]:
#     x = np.asarray(x, dtype=np.float32)
#     x = x[np.isfinite(x)]
#     if x.size == 0:
#         return {"mean": 0.0, "std": 0.0, "skew": 0.0, "kurt": 0.0, "p50": 0.0, "iqr": 0.0}
#     from scipy.stats import skew, kurtosis
#     p25, p50, p75 = np.percentile(x, [25, 50, 75])
#     return {
#         "mean": float(np.mean(x)),
#         "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
#         "skew": float(skew(x, bias=False)) if x.size > 2 else 0.0,
#         "kurt": float(kurtosis(x, bias=False)) if x.size > 3 else 0.0,
#         "p50": float(p50),
#         "iqr": float(p75 - p25),
#     }

# def _ar1(x: np.ndarray) -> float:
#     x = np.asarray(x, dtype=np.float32)
#     if x.size < 3:
#         return 0.0
#     x = x - np.mean(x)
#     den = float(np.dot(x[:-1], x[:-1])) + 1e-8
#     num = float(np.dot(x[:-1], x[1:]))
#     return float(num / den)

# def _psd_three_band_stats(x: np.ndarray, fs: float) -> List[float]:
#     x = np.asarray(x, dtype=np.float32)
#     if x.size < 8 or not np.any(np.isfinite(x)):
#         return [0.0] * 9
#     from scipy.signal import welch
#     f, Pxx = welch(x, fs=fs, nperseg=min(256, len(x)))
#     if Pxx.size == 0 or np.all(~np.isfinite(Pxx)):
#         return [0.0] * 9
#     fmax = f[-1] if f.size > 0 else fs / 2.0
#     b1 = f <= (fmax / 3.0)
#     b2 = (f > (fmax / 3.0)) & (f <= (2.0 * fmax / 3.0))
#     b3 = f > (2.0 * fmax / 3.0)
#     def band_stats(mask):
#         vals = Pxx[mask]
#         if vals.size == 0:
#             return 0.0, 0.0
#         return float(np.mean(vals)), float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
#     low_m, low_s = band_stats(b1)
#     mid_m, mid_s = band_stats(b2)
#     high_m, high_s = band_stats(b3)
#     def r(a, b): return float(a / (b + 1e-6))
#     return [low_m, mid_m, high_m, low_s, mid_s, high_s, r(low_m, mid_m), r(mid_m, high_m), r(low_m, high_m)]

# def _video_len(path: str) -> int:
#     cap = cv2.VideoCapture(path)
#     if not cap.isOpened():
#         return 0
#     total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
#     cap.release()
#     return max(total, 0)

# def _fps(path: str) -> float:
#     cap = cv2.VideoCapture(path)
#     if not cap.isOpened():
#         return 25.0
#     fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
#     cap.release()
#     return fps if fps > 1e-6 else 25.0

# def _safe_bbox_from_pts(pts, W, H, expand):
#     good = np.isfinite(pts).all(axis=1)
#     if good.sum() < 3:
#         return None
#     p = pts[good]
#     x_min = int(np.floor(p[:, 0].min()))
#     y_min = int(np.floor(p[:, 1].min()))
#     x_max = int(np.ceil(p[:, 0].max()))
#     y_max = int(np.ceil(p[:, 1].max()))
#     w = x_max - x_min
#     h = y_max - y_min
#     if w <= 0 or h <= 0:
#         return None
#     x_exp = int(round(w * expand))
#     y_exp = int(round(h * expand))
#     x1 = max(0, x_min - x_exp)
#     y1 = max(0, y_min - y_exp)
#     x2 = min(x_max + x_exp, W)
#     y2 = min(y_max + y_exp, H)
#     if x2 <= x1 or y2 <= y1:
#         return None
#     return x1, y1, x2, y2

# def _visual75_from_frames_and_openface(
#     video_path: str,
#     frame_indices: List[int],
#     fps_video: float,
#     openface_cache_csv: str,
#     mouth_expand_ratio: float = 0.4,
#     flow_bins: int = 12
# ) -> Optional[np.ndarray]:
#     df = _read_openface_csv(openface_cache_csv)
#     if df is None:
#         return None
#     lms_all, success = _collect_frame_landmarks(df)
#     if lms_all is None:
#         return None

#     rows = _map_frames_to_of_rows(df, frame_indices, fps_video)
#     if len(rows) < 2:
#         rows = rows + rows
#     if len(rows) < 2:
#         return None

#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         return None

#     flow_energy, flow_hists = [], []
#     for ti in range(len(rows) - 1):
#         r0 = int(rows[ti]); r1 = int(rows[ti + 1])
#         if success is not None:
#             if r0 >= len(success) or r1 >= len(success) or (success[r0] < 0.5 or success[r1] < 0.5):
#                 flow_energy.append(0.0); flow_hists.append(np.zeros((flow_bins,), dtype=np.float32)); continue
#         cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_indices[ti])); ok0, f0 = cap.read()
#         cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_indices[ti + 1])); ok1, f1 = cap.read()
#         if not ok0 or not ok1 or f0 is None or f1 is None:
#             flow_energy.append(0.0); flow_hists.append(np.zeros((flow_bins,), dtype=np.float32)); continue
#         Hf, Wf = f0.shape[:2]
#         lm = lms_all[r0]
#         if not np.all(np.isfinite(lm)):
#             flow_energy.append(0.0); flow_hists.append(np.zeros((flow_bins,), dtype=np.float32)); continue
#         pts = lm[MOUTH_OUT]
#         bb = _safe_bbox_from_pts(pts, Wf, Hf, expand=mouth_expand_ratio)
#         if bb is None:
#             flow_energy.append(0.0); flow_hists.append(np.zeros((flow_bins,), dtype=np.float32)); continue
#         x1, y1, x2, y2 = bb
#         g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
#         g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
#         try:
#             flow = cv2.calcOpticalFlowFarneback(
#                 g0, g1, None, pyr_scale=0.5, levels=3, winsize=15, iterations=3,
#                 poly_n=5, poly_sigma=1.2, flags=0
#             )
#             u = flow[y1:y2, x1:x2, 0]; v = flow[y1:y2, x1:x2, 1]
#             mag = np.sqrt(u * u + v * v).astype(np.float32)
#             if mag.size == 0:
#                 flow_energy.append(0.0); flow_hists.append(np.zeros((flow_bins,), dtype=np.float32)); continue
#             q95 = float(np.percentile(mag, 95)) if np.isfinite(mag).all() else 0.0
#             scale = max(q95, 1e-6)
#             mag_n = np.clip(mag / scale, 0.0, 1.0)
#             flow_energy.append(float(mag_n.mean()))
#             hist, _ = np.histogram(mag_n, bins=flow_bins, range=(0.0, 1.0), density=False)
#             hist = hist.astype(np.float32); s = float(hist.sum())
#             if s > 0: hist /= s
#             flow_hists.append(hist)
#         except Exception:
#             flow_energy.append(0.0); flow_hists.append(np.zeros((flow_bins,), dtype=np.float32))
#     cap.release()

#     flow_energy = np.asarray(flow_energy, dtype=np.float32)
#     flow_hists = np.asarray(flow_hists, dtype=np.float32)
#     if flow_hists.ndim != 2 or flow_hists.shape[1] != 12:
#         flow_hists = np.zeros((max(len(rows) - 1, 1), 12), dtype=np.float32)

#     sse = _summary_stats(flow_energy)
#     energy_feats = [sse["mean"], sse["std"], sse["skew"], sse["kurt"], sse["p50"], sse["iqr"]]
#     dyn_feats = []
#     for k in range(12):
#         series = flow_hists[:, k]
#         ss = _summary_stats(series)
#         dyn_feats.extend([ss["mean"], ss["std"], _ar1(series)])
#     if flow_hists.shape[0] >= 2:
#         d1 = np.diff(flow_hists, axis=0)
#         d1_std = [float(np.std(d1[:, k], ddof=1)) if d1.shape[0] > 1 else float(np.std(d1[:, k])) for k in range(12)]
#     else:
#         d1_std = [0.0] * 12
#     if flow_hists.shape[0] >= 3:
#         d2 = np.diff(flow_hists, n=2, axis=0)
#         d2_std = [float(np.std(d2[:, k], ddof=1)) if d2.shape[0] > 1 else float(np.std(d2[:, k])) for k in range(12)]
#     else:
#         d2_std = [0.0] * 12
#     band9 = _psd_three_band_stats(flow_energy, fs=float(fps_video))
#     visual75 = np.array(energy_feats + dyn_feats + d1_std + d2_std + band9, dtype=np.float32)
#     if visual75.shape[0] != 75:
#         return None
#     if not np.all(np.isfinite(visual75)):
#         return None
#     return visual75

# # =========================
# # Audio helpers
# # =========================
# def _have_torchaudio():
#     try:
#         import torchaudio  # noqa
#         return True
#     except Exception:
#         return False

# def _have_librosa():
#     try:
#         import librosa  # noqa
#         return True
#     except Exception:
#         return False

# def _ffmpeg_read_mono_16k(path: str, sr: int = 16000) -> Optional[np.ndarray]:
#     try:
#         with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
#             cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr), "-f", "wav", tmp.name]
#             subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#             try:
#                 import soundfile as sf
#                 audio, file_sr = sf.read(tmp.name, dtype="float32", always_2d=False)
#                 if file_sr != sr and _have_librosa():
#                     import librosa
#                     audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
#                 return audio.astype(np.float32, copy=False)
#             except Exception:
#                 import wave
#                 with wave.open(tmp.name, "rb") as wf:
#                     n = wf.getnframes()
#                     frames = wf.readframes(n)
#                     audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
#                 return audio
#     except Exception:
#         return None

# def _load_audio_mono_16k(path: str, target_sr: int = 16000) -> Optional[np.ndarray]:
#     if _have_torchaudio():
#         try:
#             import torchaudio
#             s, sr = torchaudio.load(path)
#             if s.ndim == 2 and s.size(0) > 1:
#                 s = s.mean(dim=0, keepdim=True)
#             elif s.ndim == 1:
#                 s = s.unsqueeze(0)
#             if sr != target_sr:
#                 s = torchaudio.functional.resample(s, sr, target_sr)
#             return s.squeeze(0).numpy().astype(np.float32, copy=False)
#         except Exception:
#             pass
#     if _have_librosa():
#         try:
#             import librosa
#             s, sr = librosa.load(path, sr=target_sr, mono=True)
#             return s.astype(np.float32, copy=False)
#         except Exception:
#             pass
#     return _ffmpeg_read_mono_16k(path, sr=target_sr)

# def _stft_mag(audio: np.ndarray, sr: int, n_fft: int, hop: int, win: int) -> np.ndarray:
#     if _have_librosa():
#         import librosa
#         S = librosa.stft(audio, n_fft=n_fft, hop_length=hop, win_length=win, window="hann", center=True)
#         return np.abs(S).T.astype(np.float32)
#     if _have_torchaudio():
#         import torch as _t
#         wav = _t.from_numpy(audio).unsqueeze(0)
#         spec = _t.stft(
#             wav, n_fft=n_fft, hop_length=hop, win_length=win,
#             window=_t.hann_window(win), return_complex=True
#         )
#         return spec.abs().squeeze(0).transpose(0, 1).cpu().numpy().astype(np.float32)
#     frames = np.lib.stride_tricks.sliding_window_view(audio, win)[::hop]
#     window = np.hanning(win)
#     S = np.fft.rfft(frames * window, n=n_fft)
#     return np.abs(S).astype(np.float32)

# def _logmel(audio: np.ndarray, sr: int, n_fft: int, hop: int, win: int, n_mels: int) -> np.ndarray:
#     if _have_librosa():
#         import librosa
#         M = librosa.feature.melspectrogram(
#             y=audio, sr=sr, n_fft=n_fft, hop_length=hop, win_length=win,
#             n_mels=n_mels, power=2.0
#         )
#         return librosa.power_to_db(M, ref=np.max).T.astype(np.float32)
#     if _have_torchaudio():
#         import torch as _t
#         import torchaudio
#         wav = _t.from_numpy(audio).unsqueeze(0)
#         mel = torchaudio.transforms.MelSpectrogram(
#             sample_rate=sr, n_fft=n_fft, hop_length=hop, win_length=win, n_mels=n_mels
#         )(wav).squeeze(0)
#         return _t.log(mel + 1e-6).transpose(0, 1).cpu().numpy().astype(np.float32)
#     S = _stft_mag(audio, sr, n_fft, hop, win) ** 2
#     return np.log(S + 1e-6)

# def _mfcc_block(audio: np.ndarray, sr: int, n_mfcc: int, n_fft: int, hop: int, win: int) -> Dict[str, np.ndarray]:
#     if _have_librosa():
#         import librosa
#         MF = librosa.feature.mfcc(
#             y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft,
#             hop_length=hop, win_length=win
#         ).T.astype(np.float32)
#         D1 = librosa.feature.delta(MF.T).T.astype(np.float32)
#         D2 = librosa.feature.delta(MF.T, order=2).T.astype(np.float32)
#         return {"mfcc": MF, "d_mfcc": D1, "dd_mfcc": D2}
#     else:
#         from scipy.fftpack import dct
#         mel = _logmel(audio, sr, n_fft, hop, win, n_mels=max(40, n_mfcc))
#         MF = dct(mel, type=2, norm="ortho", axis=1)[:, :n_mfcc]
#         def delta(x):
#             x = np.asarray(x, dtype=np.float32)
#             pad = np.pad(x, ((1, 1), (0, 0)), mode="edge")
#             return (pad[2:] - pad[:-2]) / 2.0
#         D1 = delta(MF); D2 = delta(D1)
#         return {"mfcc": MF.astype(np.float32), "d_mfcc": D1.astype(np.float32), "dd_mfcc": D2.astype(np.float32)}

# def _audio_feature_pack(
#     audio_seg: np.ndarray, sr: int,
#     n_fft: int, hop: int, win: int,
#     n_mels: int, n_mfcc: int
# ) -> Tuple[np.ndarray, np.ndarray, int]:
#     S_mag = _stft_mag(audio_seg, sr, n_fft, hop, win)  # [F,S]
#     F = int(S_mag.shape[0])

#     # RMS 6
#     rms_series = np.sqrt(np.sum(S_mag ** 2, axis=1))
#     ss = _summary_stats(rms_series)
#     rms_feats = [ss["mean"], ss["std"], ss["skew"], ss["kurt"], ss["p50"], ss["iqr"]]

#     # MFCC 12*(mean,std,ar1)=36 + delta std 12 + dd std 12 = 60
#     mfcc_blk = _mfcc_block(audio_seg, sr, n_mfcc=max(13, n_mfcc), n_fft=n_fft, hop=hop, win=win)
#     MF = mfcc_blk["mfcc"]
#     if MF.shape[1] < 13:
#         pad = np.zeros((MF.shape[0], 13 - MF.shape[1]), dtype=np.float32)
#         MF = np.concatenate([MF, pad], axis=1)
#     MF_12 = MF[:, 1:13].astype(np.float32)
#     D1 = mfcc_blk["d_mfcc"]; D2 = mfcc_blk["dd_mfcc"]
#     if D1.shape[1] < 13:
#         pad = np.zeros((D1.shape[0], 13 - D1.shape[1]), dtype=np.float32)
#         D1 = np.concatenate([D1, pad], axis=1)
#     if D2.shape[1] < 13:
#         pad = np.zeros((D2.shape[0], 13 - D2.shape[1]), dtype=np.float32)
#         D2 = np.concatenate([D2, pad], axis=1)
#     D1_12 = D1[:, 1:13].astype(np.float32)
#     D2_12 = D2[:, 1:13].astype(np.float32)

#     mfcc_feats = []
#     for k in range(12):
#         xk = MF_12[:, k]
#         ssk = _summary_stats(xk)
#         mfcc_feats.extend([ssk["mean"], ssk["std"], _ar1(xk)])

#     d_feats  = [float(np.std(D1_12[:, k], ddof=1)) if D1_12.shape[0] > 1 else 0.0 for k in range(12)]
#     dd_feats = [float(np.std(D2_12[:, k], ddof=1)) if D2_12.shape[0] > 1 else 0.0 for k in range(12)]

#     # Mel tri-band (9)
#     M = _logmel(audio_seg, sr, n_fft, hop, win, n_mels=n_mels)
#     M = M if M.ndim == 2 else np.atleast_2d(M)
#     Mmean = M.mean(axis=0) if M.shape[0] > 0 else np.zeros((n_mels,), dtype=np.float32)
#     Mstd = M.std(axis=0, ddof=1) if M.shape[0] > 1 else np.zeros_like(Mmean)

#     def band(idx0, idx1):
#         i0 = max(0, idx0); i1 = min(n_mels - 1, idx1)
#         if i1 < i0: return np.array([0.0]), np.array([0.0])
#         sl = slice(i0, i1 + 1)
#         return Mmean[sl], Mstd[sl]

#     low_m, low_s = band(0, 10)
#     mid_m, mid_s = band(11, 30)
#     hig_m, hig_s = band(31, n_mels - 1)
#     low_mean, mid_mean, hig_mean = float(low_m.mean()), float(mid_m.mean()), float(hig_m.mean())
#     low_std, mid_std, hig_std = float(low_s.mean()), float(mid_s.mean()), float(hig_s.mean())

#     def safe_ratio(a, b): return float(a / (b + 1e-6))
#     mel_feats = [
#         low_mean, mid_mean, hig_mean,
#         low_std,  mid_std,  hig_std,
#         safe_ratio(low_mean, mid_mean),
#         safe_ratio(mid_mean, hig_mean),
#         safe_ratio(low_mean, hig_mean),
#     ]

#     audio_feats_75 = np.array(rms_feats + mfcc_feats + d_feats + dd_feats + mel_feats, dtype=np.float32)
#     assert audio_feats_75.shape[0] == 75
#     return audio_feats_75, S_mag.astype(np.float32), F

# # =========================
# # Frame selection
# # =========================
# def _pick_indices_window(start_f: int, end_f: int, frames_per_clip: int, stride: int) -> List[int]:
#     eff = max(1, int(stride))
#     need = (frames_per_clip - 1) * eff + 1
#     if end_f - start_f + 1 < need:
#         end = min(end_f, start_f + need - 1)
#         return list(range(start_f, end + 1, eff))
#     s = start_f; e = s + need - 1
#     return list(range(s, e + 1, eff))

# def _tile_span_indices(a: int, b: int, need: int, eff: int) -> List[int]:
#     if b < a:
#         return []
#     base = list(range(a, b + 1, eff))
#     if len(base) == 0:
#         base = [a]
#     reps = (need + len(base) - 1) // len(base)
#     tiled = (base * reps)[:need]
#     return tiled

# # =========================
# # Face cropper (YOLOv8n-face with OF alignment fallback)
# # =========================
# class _FaceCropper:
#     def __init__(self, weights_path: str = "yolov8n-face.pt", img_size: int = 224, conf: float = 0.25, iou: float = 0.5):
#         self.weights_path = weights_path
#         self.img_size = int(img_size)
#         self.conf = float(conf); self.iou = float(iou)
#         self._yolo = None; self._ready = False
#         try:
#             from ultralytics import YOLO
#             self._yolo = YOLO(self.weights_path)
#             self._ready = True
#         except Exception:
#             self._yolo = None; self._ready = False

#     def _crop_from_box(self, frame_bgr: np.ndarray, box_xyxy: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
#         x1, y1, x2, y2 = [int(v) for v in box_xyxy]
#         H, W = frame_bgr.shape[:2]
#         x1, y1 = max(0, x1), max(0, y1)
#         x2, y2 = min(W - 1, x2), min(H - 1, y2)
#         if x2 <= x1 or y2 <= y1: return None
#         face = frame_bgr[y1:y2, x1:x2, :]
#         face = cv2.resize(face, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
#         return face

#     def detect_and_crop(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
#         if not self._ready:
#             return None
#         try:
#             res = self._yolo.predict(
#                 source=frame_bgr[:, :, ::-1], imgsz=self.img_size,
#                 conf=self.conf, iou=self.iou, verbose=False
#             )
#             if (not res) or (res[0].boxes is None) or (res[0].boxes.xyxy is None) or (len(res[0].boxes.xyxy) == 0):
#                 return None
#             boxes = res[0].boxes.xyxy.cpu().numpy()
#             confs = res[0].boxes.conf.cpu().numpy()
#             idx = int(np.argmax(confs))
#             x1, y1, x2, y2 = boxes[idx].astype(int).tolist()
#             return self._crop_from_box(frame_bgr, (x1, y1, x2, y2))
#         except Exception:
#             return None

# def _align_crop_from_landmarks(frame_bgr: np.ndarray, lm_68: np.ndarray, out_size: int = 224) -> Optional[np.ndarray]:
#     try:
#         left_eye = lm_68[EYE_L].mean(axis=0); right_eye = lm_68[EYE_R].mean(axis=0)
#         dy, dx = right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]
#         angle = np.degrees(np.arctan2(dy, dx))
#         eyes_center = ((left_eye[0] + right_eye[0]) * 0.5, (left_eye[1] + right_eye[1]) * 0.5)
#         M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
#         rotated = cv2.warpAffine(frame_bgr, M, (frame_bgr.shape[1], frame_bgr.shape[0]), flags=cv2.INTER_LINEAR)
#         dist = np.hypot(dx, dy)
#         box_w = int(dist * 2.0); box_h = int(dist * 2.5)
#         x1 = int(eyes_center[0] - box_w / 2); y1 = int(eyes_center[1] - box_h * 0.4)
#         x2 = x1 + box_w; y2 = y1 + box_h
#         H, W = rotated.shape[:2]
#         x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(W - 1, x2), min(H - 1, y2)
#         if x2 <= x1 or y2 <= y1: return None
#         face = rotated[y1:y2, x1:x2, :]
#         face = cv2.resize(face, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
#         return face
#     except Exception:
#         return None

# def _fallback_center_crop(frame_bgr: np.ndarray, out_size: int = 224) -> np.ndarray:
#     H, W = frame_bgr.shape[:2]
#     side = min(H, W)
#     cx, cy = W // 2, H // 2
#     x1 = max(0, cx - side // 2); y1 = max(0, cy - side // 2)
#     x2 = min(W, x1 + side); y2 = min(H, y1 + side)
#     crop = frame_bgr[y1:y2, x1:x2, :]
#     return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)

# # =========================
# # Quick ffprobe gate
# # =========================
# def _ffprobe_quick_check(path: str) -> bool:
#     if not os.path.isfile(path):
#         return False
#     try:
#         cmd = [
#             "ffprobe", "-v", "error", "-select_streams", "v:0",
#             "-show_entries", "stream=nb_frames", "-of", "csv=p=0", path
#         ]
#         r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         if r.returncode == 0:
#             out = (r.stdout or b"").decode("utf-8", "ignore").strip()
#             if out.isdigit() and int(out) > 0:
#                 return True
#         cmd = [
#             "ffprobe", "-v", "error", "-show_entries", "format=duration",
#             "-of", "default=nw=1:nk=1", path
#         ]
#         r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         if r.returncode == 0:
#             dur = (r.stdout or b"").decode("utf-8", "ignore").strip()
#             try:
#                 return float(dur) > 0.1
#             except Exception:
#                 return False
#         return False
#     except Exception:
#         return os.path.isfile(path)

# # =========================
# # Multimodal Dataset (returns vis20, vis75, face, stft, aud75, labels)
# # =========================
# class UnifiedAVDataset(Dataset):
#     """
#     Each sample returns:
#       (
#         x_vis20[ B, Dv20 ],
#         x_vis75[ B, 75   ],
#         face[ ... ],
#         stft[ B, F, S ],
#         x_aud[ B, 75 ],
#         label_mm[ B ],
#         label_a[ B ]|None,
#         label_v[ B ]|None,
#         a_len[ B ],
#         video_path
#       )
#     """

#     def __init__(
#         self,
#         *,
#         root_dir: str,
#         mode: str,                 # 'fakeavceleb'|'lavdf'|'av_deepfake1m'
#         subset: str = "train",
#         csv_path: Optional[str] = None,
#         json_path: Optional[str] = None,
#         # visual controls
#         frames_per_clip: int = 25,
#         stride: int = 1,
#         use_fake_periods: bool = True,
#         openface_binary: str = "../OpenFace/build/bin/FeatureExtraction",
#         favc_au_cache_dir: str = "../OpenFace_Cache",
#         lavdf_au_cache_dir: str = "../OpenFace_Cache_LAVDF",
#         avdf1m_au_cache_dir: str = "../OpenFace_Cache_AVDF1M",
#         mouth_expand_ratio: float = 0.4,
#         flow_bins: int = 12,
#         precomputed_dir: Optional[str] = None,
#         compute_if_missing: bool = True,
#         # visual selection
#         feature_set: str = "bins9_11_pack",
#         feature_indices: Optional[List[int]] = None,
#         selection_json_path: Optional[str] = None,
#         selection_key: Optional[str] = None,
#         enforce_vis20: bool = True,
#         # face crop
#         face_detector_weights: str = "yolov8n-face.pt",
#         face_img_size: int = 224,
#         # temporal face clip
#         return_face_seq: bool = False,
#         face_seq_len: int = 8,
#         face_seq_stride: int = 3,
#         # audio params
#         audio_sr: int = 16000,
#         stft_n_fft: int = 400,
#         stft_hop: int = 160,
#         stft_win: int = 400,
#         n_mels: int = 64,
#         n_mfcc: int = 13,
#         min_stft_frames: int = 16,
#         # misc
#         balance_minority: bool = True,
#         seed: int = 42,
#         silent_missing: bool = True,
#         visual_feat_mode: str = "vis20",
#         # failure logging
#         fail_log_dir: Optional[str] = None,
#         # === NEW: delete OpenFace CSV/HOG/TXT after each sample (defaults True per your request)
#         openface_delete_csv: bool = True,
#     ):
#         super().__init__()
#         self.root_dir = root_dir
#         self.mode = mode.lower().strip()
#         assert self.mode in {"fakeavceleb", "lavdf", "av_deepfake1m"}
#         self.subset_req = str(subset).strip().lower()
#         self.frames_per_clip = int(frames_per_clip)
#         self.stride = max(1, int(stride))
#         self.use_fake_periods = bool(use_fake_periods)
#         self.openface_binary = openface_binary
#         self.favc_au_cache_dir = favc_au_cache_dir
#         self.lavdf_au_cache_dir = lavdf_au_cache_dir
#         self.avdf1m_au_cache_dir = avdf1m_au_cache_dir
#         self.mouth_expand_ratio = float(mouth_expand_ratio)
#         self.flow_bins = int(flow_bins)
#         self.precomputed_dir = precomputed_dir
#         self.compute_if_missing = bool(compute_if_missing)
#         self.face_img_size = int(face_img_size)
#         self.face_cropper = _FaceCropper(weights_path=face_detector_weights, img_size=self.face_img_size)

#         self.return_face_seq = bool(return_face_seq)
#         self.face_seq_len = max(1, int(face_seq_len))
#         self.face_seq_stride = max(1, int(face_seq_stride))

#         self.audio_sr = int(audio_sr)
#         self.stft_n_fft = int(stft_n_fft)
#         self.stft_hop = int(stft_hop)
#         self.stft_win = int(stft_win)
#         self.n_mels = int(n_mels)
#         self.n_mfcc = int(n_mfcc)
#         self.min_stft_frames = int(min_stft_frames)

#         self.rng = random.Random(seed)
#         self.silent_missing = bool(silent_missing)
#         self._warn_count = 0
#         self._warn_cap = 100
#         self.balance_minority = bool(balance_minority)

#         self.visual_feat_mode = visual_feat_mode.lower().strip()
#         self.fail_log_dir = fail_log_dir
#         self._fail_log_file = None
#         self.openface_delete_csv = bool(openface_delete_csv)

#         if self.fail_log_dir:
#             os.makedirs(self.fail_log_dir, exist_ok=True)
#             self._fail_log_file = os.path.join(
#                 self.fail_log_dir,
#                 f"fails_{self.subset_req}_{self.mode}_{os.getpid()}.jsonl"
#             )

#         # Visual feature selection priority: explicit -> JSON -> default
#         choice = None
#         if feature_indices is not None and len(feature_indices) > 0:
#             choice = list(feature_indices)
#         if choice is None:
#             sel_from_json = _load_vis20_indices(selection_json_path, key=selection_key, mode_hint=self.mode)
#             if sel_from_json:
#                 choice = list(sel_from_json)
#         if choice is None:
#             choice = list(DEFAULT_VISUAL20_INDICES) if enforce_vis20 else list(RECOMMENDED_SETS.get(feature_set, []))
#         if enforce_vis20 and len(choice) > 20:
#             choice = choice[:20]
#         self.sel_idx = sorted(set(int(i) for i in choice))
#         self.Dv20 = len(self.sel_idx)
#         self.Dv = self.Dv20
#         self.Dv75 = 75

#         if self.precomputed_dir:
#             os.makedirs(self.precomputed_dir, exist_ok=True)

#         desired = _desired_splits(self.mode, self.subset_req)
#         folder_hints = _folder_aliases_for_files(self.mode, self.subset_req)

#         base_real: List[Tuple[str, int, Optional[List[List[float]]], Optional[int], Optional[int]]] = []
#         base_fake: List[Tuple[str, int, Optional[List[List[float]]], Optional[int], Optional[int]]] = []
#         missing_debug = []

#         if self.mode == "fakeavceleb":
#             if not csv_path:
#                 raise ValueError("csv_path required for FakeAVCeleb")
#             df = _read_favc_csv(csv_path)
#             df = df[df["split"].isin(desired)]
#             for _, row in df.iterrows():
#                 p = str(row["file_path"]).strip()
#                 lab = int(row["label"])
#                 abs_path = _resolve_visual_path(self.root_dir, p, subset_hints=folder_hints)
#                 if not abs_path or not os.path.isfile(abs_path):
#                     if len(missing_debug) < 25:
#                         missing_debug.append(p)
#                     if not self.silent_missing:
#                         warnings.warn(f"[{self.subset_req}] missing media: {p}")
#                     continue
#                 (base_real if lab == 0 else base_fake).append((abs_path, lab, None, None, None))

#         elif self.mode == "lavdf":
#             if not json_path:
#                 raise ValueError("json_path required for LAV-DF")
#             for entry in _lavdf_iter_entries(json_path):
#                 if entry["split"] not in desired:
#                     continue
#                 rel = entry["file"]
#                 abs_path = rel if os.path.isabs(rel) else os.path.join(self.root_dir, rel)
#                 if not os.path.isfile(abs_path):
#                     if len(missing_debug) < 25:
#                         missing_debug.append(rel)
#                     if not self.silent_missing:
#                         warnings.warn(f"[{self.subset_req}] missing: {abs_path}")
#                     continue
#                 y_a = 1 if entry.get("modify_audio", False) else 0
#                 y_v = 1 if entry.get("modify_video", False) else 0
#                 lab = 1 if (y_a or y_v) else 0
#                 segs = entry.get("fake_periods", []) or []
#                 (base_real if lab == 0 else base_fake).append((abs_path, lab, segs if segs else None, y_a, y_v))

#         else:  # av_deepfake1m
#             if not json_path:
#                 raise ValueError("json_path required for AV-Deepfake1M")
#             for entry in _avdf1m_iter_entries(json_path):
#                 if entry["split"] not in desired:
#                     continue
#                 rel = entry["file"]
#                 abs_path = rel if os.path.isabs(rel) else os.path.join(self.root_dir, rel)
#                 if not os.path.isfile(abs_path):
#                     if len(missing_debug) < 25:
#                         missing_debug.append(rel)
#                     if not self.silent_missing:
#                         warnings.warn(f"[{self.subset_req}] missing: {abs_path}")
#                     continue
#                 mt = (entry.get("modify_type", "") or "").strip().lower()
#                 y_a = 1 if mt in {"audio_modified", "both_modified"} else 0
#                 y_v = 1 if mt in {"visual_modified", "both_modified"} else 0
#                 lab = 1 if (y_a or y_v) else 0
#                 segs = entry.get("fake_segments", []) or []
#                 (base_real if lab == 0 else base_fake).append((abs_path, lab, segs if segs else None, y_a, y_v))

#         if (len(base_real) + len(base_fake)) == 0:
#             msg = (f"No samples for mode={self.mode}, subset={self.subset_req} (accepted={desired}) under root={self.root_dir}.")
#             if missing_debug:
#                 msg += "\nExamples (first 25):\n  - " + "\n  - ".join(missing_debug)
#             raise RuntimeError(msg)

#         samples: List[Tuple[str, int, Optional[List[List[float]]], Optional[int], Optional[int]]] = base_real + base_fake
#         if self.balance_minority:
#             n0, n1 = len(base_real), len(base_fake)
#             if n0 != n1:
#                 minority = base_fake if n0 > n1 else base_real
#                 gap = abs(n0 - n1)
#                 for _ in range(gap):
#                     src = self.rng.choice(minority)
#                     segs = [s[:] for s in (src[2] or [])] if src[2] else None
#                     samples.append((src[0], src[1], segs, src[3], src[4]))

#         self.samples = samples
#         n_real = sum(1 for _, lab, _, _, _ in self.samples if lab == 0)
#         n_fake = len(self.samples) - n_real
#         print(f"[{self.subset_req.upper()}][{self.mode}] kept={len(self.samples)} "
#               f"(Real={n_real}, Fake={n_fake}) | balance_minority={self.balance_minority} | use_fake_periods={self.use_fake_periods}")
#         print(f"Selected VIS-20 dims: Dv20={self.Dv20} -> idx={self.sel_idx} | visual_feat_mode={self.visual_feat_mode} (output=vis20+vis75)")

#         # === NEW: active chunk view support
#         self._active_indices: Optional[List[int]] = None

#     # --------- helpers ---------
#     def _log_fail(self, video_path: str, code: str, extra: Optional[Dict[str, Any]] = None):
#         if not self._fail_log_file:
#             return
#         rec = {
#             "ts": time.time(),
#             "subset": self.subset_req,
#             "mode": self.mode,
#             "video": video_path,
#             "reason": code,
#         }
#         if extra:
#             rec.update(extra)
#         try:
#             with open(self._fail_log_file, "a", encoding="utf-8") as f:
#                 f.write(json.dumps(rec, ensure_ascii=False) + "\n")
#         except Exception:
#             pass

#     def _of_cache_dir(self) -> str:
#         if self.mode == "lavdf":
#             return self.lavdf_au_cache_dir
#         if self.mode == "av_deepfake1m":
#             return self.avdf1m_au_cache_dir
#         return self.favc_au_cache_dir

#     def _v75_npy_path(self, video_path: str) -> Optional[str]:
#         if not self.precomputed_dir:
#             return None
#         base = os.path.splitext(os.path.basename(video_path))[0]
#         return os.path.join(self.precomputed_dir, base + ".v75.npy")

#     def _select_vis20(self, v75: np.ndarray) -> Optional[np.ndarray]:
#         if v75 is None or v75.size != 75 or not np.all(np.isfinite(v75)):
#             return None
#         return v75[self.sel_idx].astype(np.float32)

#     def _prepare_frame_window(self, video_path: str, label: int, segments_sec: Optional[List[List[float]]]) -> Tuple[List[int], float]:
#         total = _video_len(video_path)
#         if total <= 0:
#             return [], 25.0
#         fps = _fps(video_path)
#         eff = max(1, self.stride)
#         need = (self.frames_per_clip - 1) * eff + 1

#         if self.use_fake_periods and label == 1 and segments_sec:
#             viable = []
#             longest = None; longest_len = -1
#             for s_sec, e_sec in segments_sec:
#                 a = max(int(round(s_sec * fps)), 0)
#                 b = min(int(round(e_sec * fps)), total - 1)
#                 if b > a:
#                     L = b - a + 1
#                     if L > longest_len:
#                         longest, longest_len = (a, b), L
#                     if L >= need:
#                         viable.append((a, b))
#             if viable:
#                 a, b = self.rng.choice(viable)
#                 start = self.rng.randint(a, b - need + 1)
#                 end = start + (need - 1)
#                 return list(range(start, end + 1, eff)), fps
#             if longest is not None:
#                 a, b = longest
#                 idxs = _tile_span_indices(a, b, need, eff)
#                 return idxs, fps

#         start = 0; end = min(total - 1, start + need - 1)
#         return list(range(start, end + 1, eff)), fps

#     # === NEW: set active chunk indices
#     def _set_active_indices(self, indices: List[int]):
#         self._active_indices = list(indices)

#     def _extract_face_at(self, video_path: str, frame_index: int,
#                          of_row: Optional[int], of_df: Optional[pd.DataFrame]) -> Optional[np.ndarray]:
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             return None
#         cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
#         ok, frame = cap.read()
#         cap.release()
#         if not ok or frame is None:
#             return None
#         crop = self.face_cropper.detect_and_crop(frame)
#         if crop is not None:
#             return crop
#         if of_df is not None and of_row is not None and 0 <= of_row < len(of_df):
#             lms, _ = _collect_frame_landmarks(of_df)
#             if lms is not None and of_row < len(lms):
#                 lm = lms[of_row]
#                 if np.all(np.isfinite(lm)):
#                     acrop = _align_crop_from_landmarks(frame, lm, out_size=self.face_img_size)
#                     if acrop is not None:
#                         return acrop
#         return _fallback_center_crop(frame, out_size=self.face_img_size)

#     def _warn(self, msg: str):
#         if not self.silent_missing or self._warn_count < self._warn_cap:
#             warnings.warn(msg)
#             self._warn_count += 1

#     # --------- dataset API ---------
#     def __len__(self):
#         return len(self._active_indices) if self._active_indices is not None else len(self.samples)

#     def __getitem__(self, idx):
#         try:
#             real_idx = self._active_indices[idx] if self._active_indices is not None else idx
#             video_path, label, segments_sec, y_a, y_v = self.samples[real_idx]
#             label = int(label)

#             if not _ffprobe_quick_check(video_path):
#                 self._log_fail(video_path, "ffprobe_failed_bad_container")
#                 return None

#             total = _video_len(video_path)
#             if total <= 0:
#                 self._log_fail(video_path, "unreadable_video_total_frames_0")
#                 self._warn(f"[skip:unreadable_video] {video_path}")
#                 return None

#             frame_idx, fps = self._prepare_frame_window(video_path, label, segments_sec)
#             if not frame_idx:
#                 self._log_fail(video_path, "no_frames_after_selection",
#                                {"label": int(label), "segments_sec": segments_sec})
#                 self._warn(f"[skip:no_frames] {video_path}")
#                 return None
#             start_f = frame_idx[0]; end_f = frame_idx[-1]

#             # OpenFace cache + rows mapping
#             cache_dir = self._of_cache_dir()
#             csv_path, _created_now = _ensure_openface_csv(video_path, cache_dir, self.openface_binary)
#             of_df = _read_openface_csv(csv_path) if csv_path else None
#             of_rows = _map_frames_to_of_rows(of_df, frame_idx, fps)

#             # Visual-75
#             v75 = None
#             npy_path = self._v75_npy_path(video_path)
#             if npy_path and os.path.isfile(npy_path):
#                 try:
#                     v75 = np.load(npy_path).astype(np.float32)
#                 except Exception:
#                     v75 = None
#             else:
#                 if self.compute_if_missing and csv_path is not None:
#                     v75 = _visual75_from_frames_and_openface(
#                         video_path=video_path,
#                         frame_indices=frame_idx,
#                         fps_video=fps,
#                         openface_cache_csv=csv_path,
#                         mouth_expand_ratio=self.mouth_expand_ratio,
#                         flow_bins=self.flow_bins,
#                     )
#                     if v75 is not None and npy_path is not None:
#                         try:
#                             np.save(npy_path, v75)
#                         except Exception:
#                             pass

#             if v75 is None or v75.shape != (75,) or not np.all(np.isfinite(v75)):
#                 self._log_fail(video_path, "visual75_failure",
#                                {"has_csv": bool(csv_path), "csv_path": csv_path,
#                                 "precomputed_exists": bool(npy_path and os.path.isfile(npy_path))})
#                 self._warn(f"[skip:visual75_failure] {video_path}")
#                 # === NEW: cleanup OpenFace artifacts before returning
#                 if self.openface_delete_csv and csv_path:
#                     _cleanup_openface_artifacts(cache_dir, video_path)
#                 return None

#             x_vis75_np = v75
#             x_vis20_np = self._select_vis20(x_vis75_np)
#             if x_vis20_np is None or x_vis20_np.shape[0] != self.Dv20 or not np.all(np.isfinite(x_vis20_np)):
#                 self._log_fail(video_path, "vis20_selection_failure", {"idx": self.sel_idx})
#                 self._warn(f"[skip:vis20_selection_failure] {video_path}")
#                 if self.openface_delete_csv and csv_path:
#                     _cleanup_openface_artifacts(cache_dir, video_path)
#                 return None

#             x_vis75_t = torch.from_numpy(x_vis75_np).to(torch.float32)
#             x_vis20_t = torch.from_numpy(x_vis20_np).to(torch.float32)

#             # Face
#             if not self.return_face_seq:
#                 mid_i = frame_idx[len(frame_idx) // 2]
#                 mid_row = of_rows[len(of_rows) // 2] if len(of_rows) > 0 else None
#                 face_bgr = self._extract_face_at(video_path, mid_i, mid_row, of_df)
#                 if face_bgr is None:
#                     self._log_fail(video_path, "face_crop_failure_single")
#                     self._warn(f"[skip:face_failure] {video_path}")
#                     if self.openface_delete_csv and csv_path:
#                         _cleanup_openface_artifacts(cache_dir, video_path)
#                     return None
#                 face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
#                 face_out = torch.from_numpy(face_rgb.transpose(2, 0, 1)).to(torch.float32) / 255.0
#             else:
#                 seq_idx: List[int] = []
#                 step = int(self.face_seq_stride) if self.face_seq_stride > 0 else 1
#                 for i in range(0, len(frame_idx), step):
#                     seq_idx.append(frame_idx[i])
#                 if len(seq_idx) < self.face_seq_len:
#                     while len(seq_idx) < self.face_seq_len:
#                         seq_idx.append(frame_idx[-1])
#                 if len(seq_idx) > self.face_seq_len:
#                     extra = len(seq_idx) - self.face_seq_len
#                     drop_l = extra // 2
#                     drop_r = extra - drop_l
#                     seq_idx = seq_idx[drop_l: len(seq_idx) - drop_r]
#                 of_rows_seq = _map_frames_to_of_rows(of_df, seq_idx, fps)
#                 faces_list: List[torch.Tensor] = []
#                 for fi, ofr in zip(seq_idx, of_rows_seq):
#                     crop_bgr = self._extract_face_at(video_path, fi, ofr, of_df)
#                     if crop_bgr is None:
#                         self._log_fail(video_path, "face_crop_failure_seq", {"fi": int(fi)})
#                         self._warn(f"[skip:face_seq_failure] {video_path}")
#                         if self.openface_delete_csv and csv_path:
#                             _cleanup_openface_artifacts(cache_dir, video_path)
#                         return None
#                     face_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
#                     t = torch.from_numpy(face_rgb.transpose(2, 0, 1)).to(torch.float32) / 255.0
#                     faces_list.append(t)
#                 face_out = torch.stack(faces_list, dim=0)

#             # Audio strictly aligned
#             a_start_sec = float(start_f) / max(fps, 1e-6)
#             a_end_sec   = float(end_f + 1) / max(fps, 1e-6)

#             audio = _load_audio_mono_16k(video_path, target_sr=self.audio_sr)
#             if audio is None or audio.size == 0:
#                 base = os.path.splitext(video_path)[0]
#                 found = None
#                 for ext in AUDIO_EXTS:
#                     cand = base + ext
#                     if os.path.isfile(cand):
#                         found = cand; break
#                 audio = _load_audio_mono_16k(found, target_sr=self.audio_sr) if found else None
#             if audio is None or audio.size == 0:
#                 self._log_fail(video_path, "audio_missing")
#                 self._warn(f"[skip:audio_missing] {video_path}")
#                 if self.openface_delete_csv and csv_path:
#                     _cleanup_openface_artifacts(cache_dir, video_path)
#                 return None

#             a_start = max(0, int(round(a_start_sec * self.audio_sr)))
#             a_end   = min(audio.shape[0], int(round(a_end_sec * self.audio_sr)))
#             if a_end <= a_start:
#                 approx_samples = int(round((self.frames_per_clip / max(fps, 1e-6)) * self.audio_sr))
#                 a_end = min(audio.shape[0], a_start + approx_samples)
#             if a_end <= a_start:
#                 self._log_fail(video_path, "audio_window_invalid", {"a_start": a_start, "a_end": a_end, "fps": float(fps)})
#                 self._warn(f"[skip:audio_window_invalid] {video_path}")
#                 if self.openface_delete_csv and csv_path:
#                     _cleanup_openface_artifacts(cache_dir, video_path)
#                 return None
#             seg = audio[a_start:a_end]
#             if seg.size <= 16:
#                 self._log_fail(video_path, "audio_too_short", {"seg_len": int(seg.size)})
#                 self._warn(f"[skip:audio_too_short] {video_path}")
#                 if self.openface_delete_csv and csv_path:
#                     _cleanup_openface_artifacts(cache_dir, video_path)
#                 return None

#             audio75, stft, a_len = _audio_feature_pack(
#                 seg, self.audio_sr,
#                 self.stft_n_fft, self.stft_hop, self.stft_win,
#                 self.n_mels, self.n_mfcc
#             )

#             if audio75.shape != (75,) or not np.all(np.isfinite(audio75)):
#                 self._log_fail(video_path, "audio75_invalid")
#                 self._warn(f"[skip:audio75_invalid] {video_path}")
#                 if self.openface_delete_csv and csv_path:
#                     _cleanup_openface_artifacts(cache_dir, video_path)
#                 return None
#             if stft.ndim != 2 or stft.shape[0] <= 0 or stft.shape[1] <= 0 or not np.all(np.isfinite(stft)):
#                 self._log_fail(video_path, "stft_invalid", {"shape": list(stft.shape)})
#                 self._warn(f"[skip:stft_invalid] {video_path}")
#                 if self.openface_delete_csv and csv_path:
#                     _cleanup_openface_artifacts(cache_dir, video_path)
#                 return None

#             stft_t    = torch.from_numpy(stft).to(torch.float32)
#             audio75_t = torch.from_numpy(audio75).to(torch.float32)
#             label_mm_t   = torch.tensor(label, dtype=torch.long)
#             label_a_t = torch.tensor(int(y_a), dtype=torch.long) if y_a is not None else None
#             label_v_t = torch.tensor(int(y_v), dtype=torch.long) if y_v is not None else None

#             # === NEW: final cleanup of OpenFace artifacts for this video
#             if self.openface_delete_csv and csv_path:
#                 _cleanup_openface_artifacts(cache_dir, video_path)

#             return x_vis20_t, x_vis75_t, face_out, stft_t, audio75_t, label_mm_t, label_a_t, label_v_t, int(a_len), video_path

#         except Exception as e:
#             # Attempt cleanup on exception if we can infer video path
#             try:
#                 if self.openface_delete_csv:
#                     cache_dir = self._of_cache_dir()
#                     vp = self.samples[idx][0] if (self._active_indices is None and idx < len(self.samples)) else \
#                          self.samples[self._active_indices[idx]][0]
#                     _cleanup_openface_artifacts(cache_dir, vp)
#             except Exception:
#                 pass
#             self._log_fail(self.samples[0][0] if len(self.samples) > 0 else "unknown",
#                            "exception", {"etype": type(e).__name__, "msg": str(e)})
#             self._warn(f"[skip:exception] {type(e).__name__}: {e}")
#             return None

# # =========================
# # Collate + Loader
# # =========================
# def _pad_time(batch_tensors: List[torch.Tensor]):
#     lengths = [t.size(0) for t in batch_tensors]
#     Fmax = max(lengths); D = batch_tensors[0].size(1)
#     out = torch.zeros((len(batch_tensors), Fmax, D), dtype=torch.float32)
#     for i, t in enumerate(batch_tensors):
#         out[i, :t.size(0)] = t
#     return out, torch.tensor(lengths, dtype=torch.long)

# def _stack_optional(labels_list: List[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
#     if any(l is None for l in labels_list):
#         return None
#     return torch.stack([l if torch.is_tensor(l) else torch.tensor(l) for l in labels_list]).long()

# def collate_unified_av(batch):
#     batch = [b for b in batch if b is not None]
#     if len(batch) == 0:
#         return None

#     x_vis20, x_vis75, faces, stfts, x_aud, labels_mm, labels_a, labels_v, a_lens, paths = zip(*batch)

#     x_vis20 = torch.stack(list(x_vis20), dim=0)
#     x_vis75 = torch.stack(list(x_vis75), dim=0)

#     faces0 = faces[0]
#     if torch.is_tensor(faces0) and faces0.ndim == 3:
#         faces = torch.stack(list(faces), dim=0)
#     elif torch.is_tensor(faces0) and faces0.ndim == 4:
#         faces = torch.stack(list(faces), dim=0)
#     else:
#         raise ValueError(f"Unexpected face tensor shape: {None if faces0 is None else tuple(faces0.shape)}")

#     stft_pad, a_lengths = _pad_time(list(stfts))
#     x_aud   = torch.stack(list(x_aud), dim=0)
#     labels_mm = torch.stack([torch.tensor(l) if not torch.is_tensor(l) else l for l in labels_mm]).long()

#     labels_a = _stack_optional(list(labels_a))
#     labels_v = _stack_optional(list(labels_v))

#     return x_vis20, x_vis75, faces, stft_pad, x_aud, labels_mm, labels_a, labels_v, a_lengths, list(paths)

# class ProgressDataLoader(DataLoader):
#     def __init__(self, *args, show_tqdm: bool = False, desc: str = "", **kwargs):
#         super().__init__(*args, **kwargs)
#         self._show_tqdm = bool(show_tqdm)
#         self._tqdm_desc = str(desc) if desc else "dataloader"

#     def __iter__(self):
#         it = super().__iter__()
#         if self._show_tqdm and (sys.stdout.isatty() or sys.stderr.isatty()):
#             for batch in tqdm(it, total=len(self), desc=self._tqdm_desc, ncols=100, leave=False):
#                 if batch is None:
#                     continue
#                 yield batch
#             return
#         for batch in it:
#             if batch is None:
#                 continue
#             yield batch

# # =========================
# # === NEW: AVDF1M chunk manager + epoch state tracker
# # =========================
# class AVDF1MChunkManager:
#     """
#     Manages N-chunk sampling (default 5000; ~50/50 real-fake) and persistent epoch progress
#     for AV-Deepfake1M. Works by assigning active indices to UnifiedAVDataset.
#     """

#     def __init__(
#         self,
#         *,
#         dataset: "UnifiedAVDataset",
#         chunk_size: int = 5000,
#         fake_ratio: float = 0.5,
#         state_dir: str = "./avdf1m_state",
#         resume: bool = True,
#         start_percent: Optional[float] = None,
#         seed: int = 42,
#         openface_threads: int = 0,  # 0 disables per-chunk precompute
#     ):
#         assert dataset.mode == "av_deepfake1m", "AVDF1MChunkManager requires mode='av_deepfake1m'"
#         self.ds = dataset
#         self.chunk_size = int(chunk_size)
#         self.fake_ratio = float(fake_ratio)
#         self.state_dir = state_dir
#         self.resume = bool(resume)
#         self.rng = random.Random(seed)
#         self.openface_threads = max(0, int(openface_threads))
#         os.makedirs(self.state_dir, exist_ok=True)

#         self.real_idxs = [i for i, (_, y, _, _, _) in enumerate(self.ds.samples) if y == 0]
#         self.fake_idxs = [i for i, (_, y, _, _, _) in enumerate(self.ds.samples) if y == 1]
#         self.total_pool = len(self.ds.samples)

#         self.state_path = os.path.join(self.state_dir, f"state_{self.ds.subset_req}.json")
#         self.state = self._load_state()

#         if (start_percent is not None) and (0.0 <= float(start_percent) <= 100.0):
#             self._seek_to_percent(float(start_percent))

#         if not self.ds._active_indices:
#             self.prepare_next_chunk()

#     def _load_state(self) -> Dict[str, Any]:
#         if self.resume and os.path.isfile(self.state_path):
#             try:
#                 with open(self.state_path, "r", encoding="utf-8") as f:
#                     st = json.load(f)
#                 st.setdefault("epoch_id", 0)
#                 st.setdefault("chunk_no", 0)
#                 st.setdefault("seen_ids", {})
#                 st.setdefault("total_pool", self.total_pool)
#                 st.setdefault("progress_percent", 100.0 * len(st["seen_ids"]) / max(1, self.total_pool))
#                 return st
#             except Exception:
#                 pass
#         return {
#             "epoch_id": 0,
#             "chunk_no": 0,
#             "seen_ids": {},
#             "total_pool": self.total_pool,
#             "progress_percent": 0.0,
#         }

#     def _save_state(self):
#         self.state["total_pool"] = self.total_pool
#         self.state["progress_percent"] = 100.0 * len(self.state["seen_ids"]) / max(1, self.total_pool)
#         try:
#             with open(self.state_path, "w", encoding="utf-8") as f:
#                 json.dump(self.state, f, ensure_ascii=False, indent=2)
#         except Exception:
#             pass

#     def _seek_to_percent(self, pct: float):
#         need_seen = int(round((pct / 100.0) * self.total_pool))
#         all_ids = [_stable_id_from_path(self.ds.samples[i][0]) for i in range(self.total_pool)]
#         self.rng.shuffle(all_ids)
#         pre_seen = {sid: 1 for sid in all_ids[:need_seen]}
#         self.state["seen_ids"] = pre_seen
#         self._save_state()

#     def _filter_unseen(self, idx_list: List[int]) -> List[int]:
#         seen = self.state["seen_ids"]
#         out = []
#         for i in idx_list:
#             p = self.ds.samples[i][0]
#             sid = _stable_id_from_path(p)
#             if sid not in seen:
#                 out.append(i)
#         return out

#     def _mark_seen(self, indices: List[int]):
#         for i in indices:
#             p = self.ds.samples[i][0]
#             sid = _stable_id_from_path(p)
#             self.state["seen_ids"][sid] = 1

#     def percent_complete(self) -> float:
#         return float(self.state.get("progress_percent", 0.0))

#     def chunk_id(self) -> int:
#         return int(self.state.get("chunk_no", 0))

#     def epoch_id(self) -> int:
#         return int(self.state.get("epoch_id", 0))

#     def prepare_next_chunk(self) -> List[int]:
#         avail_real = self._filter_unseen(self.real_idxs)
#         avail_fake = self._filter_unseen(self.fake_idxs)

#         if len(avail_real) + len(avail_fake) < 1:
#             self.state["epoch_id"] += 1
#             self.state["chunk_no"] = 0
#             self.state["seen_ids"] = {}
#             avail_real = list(self.real_idxs)
#             avail_fake = list(self.fake_idxs)

#         self.rng.shuffle(avail_real)
#         self.rng.shuffle(avail_fake)

#         need_fake = int(round(self.chunk_size * self.fake_ratio))
#         need_real = self.chunk_size - need_fake

#         pick_fake = avail_fake[:min(len(avail_fake), need_fake)]
#         pick_real = avail_real[:min(len(avail_real), need_real)]

#         got = len(pick_fake) + len(pick_real)
#         if got < self.chunk_size:
#             remain = self.chunk_size - got
#             pool_extra = avail_real[len(pick_real):] + avail_fake[len(pick_fake):]
#             self.rng.shuffle(pool_extra)
#             pick_extra = pool_extra[:remain]
#             indices = pick_fake + pick_real + pick_extra
#         else:
#             indices = pick_fake + pick_real

#         self.rng.shuffle(indices)

#         self._mark_seen(indices)
#         self.state["chunk_no"] += 1
#         self._save_state()

#         self.ds._set_active_indices(indices)

#         if self.openface_threads > 0:
#             vids = [self.ds.samples[i][0] for i in indices]
#             _ensure_openface_csv_batch(
#                 vids,
#                 cache_dir=self.ds._of_cache_dir(),
#                 openface_binary=self.ds.openface_binary,
#                 max_workers=self.openface_threads,
#                 show_progress=True,
#             )
#         return indices

# # =========================
# # Loader factory
# # =========================
# def get_unified_av_dataloader(
#     *,
#     root_dir: str,
#     mode: str,                         # 'fakeavceleb'|'lavdf'|'av_deepfake1m'
#     subset: str,
#     csv_path: Optional[str] = None,
#     json_path: Optional[str] = None,
#     batch_size: int = 8,
#     shuffle: bool = False,
#     num_workers: int = 0,
#     seed: int = 42,
#     # visual
#     frames_per_clip: int = 25,
#     stride: int = 1,
#     use_fake_periods: bool = True,
#     openface_binary: str = "",
#     favc_au_cache_dir: str = "../OpenFace_Cache",
#     lavdf_au_cache_dir: str = "../OpenFace_Cache_LAVDF",
#     avdf1m_au_cache_dir: str = "../OpenFace_Cache_AVDF1M",
#     mouth_expand_ratio: float = 0.4,
#     flow_bins: int = 12,
#     precomputed_dir: Optional[str] = None,
#     compute_if_missing: bool = False,
#     feature_set: str = "bins9_11_pack",
#     feature_indices: Optional[List[int]] = None,
#     selection_json_path: Optional[str] = None,
#     selection_key: Optional[str] = None,
#     enforce_vis20: bool = True,
#     # face
#     face_detector_weights: str = "yolov8n-face.pt",
#     face_img_size: int = 224,
#     # temporal faces
#     return_face_seq: bool = False,
#     face_seq_len: int = 8,
#     face_seq_stride: int = 3,
#     # audio
#     audio_sr: int = 16000,
#     stft_n_fft: int = 400,
#     stft_hop: int = 160,
#     stft_win: int = 400,
#     n_mels: int = 64,
#     n_mfcc: int = 13,
#     min_stft_frames: int = 16,
#     # misc
#     balance_minority: bool = True,
#     pin_memory: bool = True,
#     show_tqdm: bool = True,
#     visual_feat_mode: str = "vis20",
#     fail_log_dir: Optional[str] = None,
#     # === NEW: delete OpenFace files after use (default True per your request)
#     openface_delete_csv: bool = True,
#     # === NEW: AVDF1M chunk controls
#     avdf1m_enable_chunking: bool = False,
#     avdf1m_chunk_size: int = 5000,
#     avdf1m_fake_ratio: float = 0.5,
#     avdf1m_state_dir: str = "./avdf1m_state",
#     avdf1m_resume: bool = True,
#     avdf1m_start_percent: Optional[float] = None,
#     # === NEW: OpenFace precompute threads
#     openface_threads: int = 0,
#     # === NEW: return the chunk manager (for percent progress & next-chunk advance)
#     return_manager: bool = False,
# ):
#     ds = UnifiedAVDataset(
#         root_dir=root_dir, mode=mode, subset=subset,
#         csv_path=csv_path, json_path=json_path,
#         frames_per_clip=frames_per_clip, stride=stride, use_fake_periods=use_fake_periods,
#         openface_binary=openface_binary,
#         favc_au_cache_dir=favc_au_cache_dir, lavdf_au_cache_dir=lavdf_au_cache_dir, avdf1m_au_cache_dir=avdf1m_au_cache_dir,
#         mouth_expand_ratio=mouth_expand_ratio, flow_bins=flow_bins,
#         precomputed_dir=precomputed_dir, compute_if_missing=compute_if_missing,
#         feature_set=feature_set, feature_indices=feature_indices,
#         selection_json_path=selection_json_path, selection_key=selection_key, enforce_vis20=enforce_vis20,
#         face_detector_weights=face_detector_weights, face_img_size=face_img_size,
#         return_face_seq=return_face_seq, face_seq_len=face_seq_len, face_seq_stride=face_seq_stride,
#         audio_sr=audio_sr, stft_n_fft=stft_n_fft, stft_hop=stft_hop, stft_win=stft_win,
#         n_mels=n_mels, n_mfcc=n_mfcc, min_stft_frames=min_stft_frames,
#         balance_minority=balance_minority, seed=seed,
#         silent_missing=True,
#         visual_feat_mode=visual_feat_mode,
#         fail_log_dir=fail_log_dir,
#         openface_delete_csv=openface_delete_csv,
#     )

#     # AVDF1M chunking (optional)
#     manager = None
#     if (mode.lower().strip() == "av_deepfake1m") and avdf1m_enable_chunking:
#         manager = AVDF1MChunkManager(
#             dataset=ds,
#             chunk_size=avdf1m_chunk_size,
#             fake_ratio=avdf1m_fake_ratio,
#             state_dir=avdf1m_state_dir,
#             resume=avdf1m_resume,
#             start_percent=avdf1m_start_percent,
#             seed=seed,
#             openface_threads=openface_threads,
#         )
#     else:
#         # For non-AVDF1M (or if chunking disabled), optionally precompute OpenFace for *all* items
#         if openface_threads > 0:
#             vids = [p for (p, _, _, _, _) in ds.samples]
#             _ensure_openface_csv_batch(
#                 vids,
#                 cache_dir=ds._of_cache_dir(),
#                 openface_binary=ds.openface_binary,
#                 max_workers=openface_threads,
#                 show_progress=True,
#             )

#     desc = (f"{subset}/{mode} A/V-aligned (VIS-20+VIS-75"
#             f", face{face_img_size}"
#             + (f", T={face_seq_len}@{face_seq_stride}" if return_face_seq else "")
#             + f", sr={audio_sr}, clipT={frames_per_clip})")

#     loader = ProgressDataLoader(
#         ds,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         num_workers=num_workers,
#         pin_memory=pin_memory,
#         persistent_workers=(num_workers > 0),
#         collate_fn=collate_unified_av,
#         drop_last=False,
#         show_tqdm=show_tqdm,
#         desc=desc,
#     )

#     if return_manager:
#         return loader, manager
#     return loader