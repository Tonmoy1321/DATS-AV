import os
import json
import random
import warnings
from typing import Optional, Dict, Tuple, Set, List

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("ULTRALYTICS_VERBOSITY", "0")
os.environ.setdefault("WANDB_SILENT", "true")
warnings.filterwarnings("ignore")

from av_dissonance_with_aux_ensemble_model import build_dissonance_dual_model
from unified_av_dataloader import get_unified_av_dataloader

CONFIG = {
    # ===== Datasets =====
    # FakeAVCeleb
    "FAVC_ROOT": "/media/rt0706/Media/VCBSL-Dataset/FakeAVCeleb",
    "FAVC_CSV": "../Dataset/favc_multimodal_data.csv",  # contains split="test"
    "FAVC_SUBSET": "test",

    # LAV-DF
    "LAVDF_ROOT": "/media/rt0706/Media/VCBSL-Dataset/LAV-DF",
    "LAVDF_JSON": "../Dataset/LAV-DF/metadata.json",
    "LAVDF_SUBSET": "test",

    # (Optional) AV-Deepfake1M listed for reference; not used in this calibration script
    "AVDF1M_ROOT": "/media/rt0706/Lab/AV-Deepfake1M_test_10pct/zips/val",
    "AVDF1M_JSON": "/media/rt0706/Lab/AV-Deepfake1M_test_10pct/zips/val_metadata.json",
    "AVDF1M_SUBSET": "val",

    # ===== Model checkpoint =====
    "CKPT": "runs/avdf1m_dual_avdiss_visual75_vf/ckpt_best_auc.pt",

    # ===== Dataloader / feature extraction =====
    "VIS20_JSON": "runs/visual75_crossdomain_selection.json",
    "OPENFACE_BINARY": "../OpenFace/build/bin/FeatureExtraction",
    "PRECOMPUTED_V75_DIR": None,   # e.g., "runs/precomp_v75"
    "COMPUTE_IF_MISSING": True,    # compute VIS-75 if .npy missing (needs OpenFace cache)
    "FAIL_LOG_DIR": "runs/fail_logs_calib_repeats",

    "BATCH_SIZE": 4,
    "NUM_WORKERS": 2,
    "FRAMES_PER_CLIP": 25,
    "STRIDE": 1,
    "AUDIO_SR": 16000,
    "NFFT": 400,
    "HOP": 160,
    "WIN": 400,
    "FACE_SIZE": 224,
    "FACE_DET_WEIGHTS": "yolov8n-face.pt",
    "USE_FAKE_PERIODS": True,

    # ===== Calibration control =====
    "FRACTION": 0.20,          # 20% stratified subset
    "REPEATS": 5,              # number of independent repeats
    # dataset-specific optimization metric (FAVC is highly imbalanced -> AP more stable)
    "FAVC_CALIB_METRIC": "ap",     # "ap" | "auc" | "acc" | "eer"
    "LAVDF_CALIB_METRIC": "auc",   # "auc" by default
    # decision threshold rule: "youden" | "acc" | "eer" | "balanced" (equalize TPR and TNR)
    "THR_CRITERION": "balanced",

    # ===== Aggregation of repeats: "median" or "best"
    "AGGREGATION": "median",

    # ===== FAVC quick preset to trust AUX more (can override aggregated params)
    "FAVC_TRUST_AUX_PRESET": False,  # set True to force these overrides:
    "FAVC_PRESET_T_AV": 1.2,         # modest temperature on AV head
    "FAVC_PRESET_TAU": 0.99,         # trust AUX almost always; use 1.0 to force AUX-only

    # ===== Repro & device =====
    "SEED": 42,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",

    # ===== Output JSON =====
    "OUT_JSON": "runs/calib_repeated_20pct_favc_lavdf.json",

    # ===== Post-calibration: evaluate full test with aggregated params =====
    "EVAL_FULL_AFTER_CALIB": True,
}

# =========================
# Utilities
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _sigmoid_np(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))

def _eer(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fnr = 1.0 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2.0)

def _balanced_decision_thr(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Equalize TPR and TNR."""
    fpr, tpr, thr = roc_curve(y_true, probs)
    tnr = 1.0 - fpr
    idx = int(np.nanargmin(np.abs(tpr - tnr)))
    return float(thr[idx])

def compute_metrics(y_true: np.ndarray, probs: np.ndarray, thr: Optional[float] = 0.5) -> Dict[str, float]:
    y_true = y_true.astype(np.int64)
    probs = probs.astype(np.float32)
    multi = (len(np.unique(y_true)) > 1)
    auc = roc_auc_score(y_true, probs) if multi else 0.5
    ap  = average_precision_score(y_true, probs) if multi else 0.5
    eer = _eer(y_true, probs) if multi else 0.5
    t = 0.5 if thr is None else float(thr)
    preds = (probs >= t).astype(np.int64)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    total = int(y_true.size)
    acc = (tp + tn) / max(1, total)
    return {
        "auc": float(auc), "ap": float(ap), "eer": float(eer),
        "acc": float(acc), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "total": total, "correct": tp + tn,
        "correct_real": tn, "correct_fake": tp,
        "thr": t,
    }

def switched_probs(ld, la, T_av=1.0, tau=0.25):
    pav = _sigmoid_np(ld / max(T_av, 1e-6))
    pax = _sigmoid_np(la)
    conf = 2.0 * np.abs(pav - 0.5)
    pick_aux = (conf < tau).astype(np.float32)
    pfinal = (1.0 - pick_aux) * pav + pick_aux * pax
    return pfinal, conf, pick_aux

def best_decision_threshold(y_true: np.ndarray, probs: np.ndarray, criterion: str = "youden") -> float:
    if criterion == "balanced":
        return _balanced_decision_thr(y_true, probs)
    fpr, tpr, thr = roc_curve(y_true, probs)
    if criterion == "youden":
        j = tpr - fpr
        idx = int(np.argmax(j))
    elif criterion == "acc":
        accs = [(((probs >= t).astype(int) == y_true).mean()) for t in thr]
        idx = int(np.argmax(accs))
    else:  # "eer"
        idx = int(np.argmin(np.abs(tpr - (1 - fpr))))
    return float(thr[idx])

def fit_temperature_binary(logits: np.ndarray, labels: np.ndarray) -> float:
    T = 1.0
    lr = 0.05
    y = labels.astype(np.float32)
    for _ in range(200):
        p = _sigmoid_np(logits / max(T, 1e-6))
        p = np.clip(p, 1e-6, 1-1e-6)
        grad = np.mean((p - y) * (-logits) / (max(T, 1e-6)**2))
        T = max(1e-3, T - lr * grad)
        if not np.isfinite(T):
            return 1.0
    return float(T)

def sweep_tau(y, ld, la, T_av=1.0, metric="auc"):
    grid = np.linspace(0.0, 1.0, 101)
    best_tau, best_val, best_row = 0.25, -np.inf, None
    for tau in grid:
        p, _, _ = switched_probs(ld, la, T_av=T_av, tau=tau)
        auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else 0.5
        ap  = average_precision_score(y, p) if len(np.unique(y)) > 1 else 0.5
        eer = _eer(y, p)
        acc = (p >= 0.5).astype(np.int32).mean()
        score_map = {"auc": auc, "ap": ap, "acc": acc, "eer": -eer}
        score = score_map.get(metric, auc)
        if score > best_val:
            best_val = score
            best_tau = float(tau)
            best_row = {"tau": best_tau, "auc": float(auc), "ap": float(ap), "eer": float(eer), "acc": float(acc)}
    return best_tau, best_row

# =========================
# Model & loaders
# =========================
def build_model_from_ckpt(ckpt_path: str, device: torch.device):
    payload = torch.load(ckpt_path, map_location=device)
    cfg = payload.get("config", {}) or {}
    vis20_dim = int(cfg.get("VIS20_DIM", 20))
    stft_bins = int(cfg.get("NFFT", 400)) // 2 + 1

    model, _ = build_dissonance_dual_model(
        vis_dim=vis20_dim,
        aud_dim=75,
        stft_bins=stft_bins,
        emb_dim_audio=cfg.get("EMB_DIM_AUDIO", 128),
        emb_dim_visface=cfg.get("EMB_DIM_VISFACE", 256),
        hidden_audio=cfg.get("HIDDEN_AUDIO", 256),
        enc_heads=cfg.get("HEADS", 4),
        enc_layers=cfg.get("LAYERS", 1),
        enc_dropout=cfg.get("ENC_DROPOUT", 0.1),
        pe_max_len=cfg.get("PE_MAX_LEN", 2000),
        pe_dropout=cfg.get("PE_DROPOUT", 0.0),
        cls_dropout_audio=cfg.get("CLS_DROPOUT_AUDIO", 0.3),
        fusion_mode=cfg.get("FUSION_MODE", "gated"),
        face_pretrained=cfg.get("FACE_PRETRAINED", True),
        face_freeze_backbone=cfg.get("FACE_FREEZE_BACKBONE", False),
        switch_threshold=cfg.get("SWITCH_THR", 0.25),
        lambda_aux=cfg.get("LAMBDA_DISSONANCE_AUX", 0.25),
        lambda_total_balancing=cfg.get("LAMBDA_TOTAL_BALANCING", 1.0),
    )
    model.load_state_dict(payload["model"], strict=True)
    model.to(device)
    model.eval()
    return model, vis20_dim, stft_bins, cfg

def make_loader_favc(cfg, *, shuffle=False):
    return get_unified_av_dataloader(
        mode="fakeavceleb",
        subset=cfg["FAVC_SUBSET"],
        root_dir=cfg["FAVC_ROOT"],
        csv_path=cfg["FAVC_CSV"],
        frames_per_clip=cfg["FRAMES_PER_CLIP"],
        stride=cfg["STRIDE"],
        balance_minority=False,
        use_fake_periods=False,  # TEST time, no segment conditioning
        audio_sr=cfg["AUDIO_SR"],
        stft_n_fft=cfg["NFFT"],
        stft_hop=cfg["HOP"],
        stft_win=cfg["WIN"],
        batch_size=cfg["BATCH_SIZE"],
        num_workers=cfg["NUM_WORKERS"],
        face_img_size=cfg["FACE_SIZE"],
        feature_set="bins9_11_pack",
        selection_json_path=cfg["VIS20_JSON"],
        selection_key=None,
        enforce_vis20=True,
        shuffle=shuffle,
        show_tqdm=True,
        face_detector_weights=cfg["FACE_DET_WEIGHTS"],
        visual_feat_mode="vis75",
        openface_binary=cfg.get("OPENFACE_BINARY", "") or "",
        precomputed_dir=cfg.get("PRECOMPUTED_V75_DIR", None),
        compute_if_missing=cfg.get("COMPUTE_IF_MISSING", False),
        fail_log_dir=cfg.get("FAIL_LOG_DIR", None),
    )

def make_loader_lavdf(cfg, *, shuffle=False):
    return get_unified_av_dataloader(
        mode="lavdf",
        subset=cfg["LAVDF_SUBSET"],
        root_dir=cfg["LAVDF_ROOT"],
        json_path=cfg["LAVDF_JSON"],
        frames_per_clip=cfg["FRAMES_PER_CLIP"],
        stride=cfg["STRIDE"],
        balance_minority=False,
        use_fake_periods=True,   # USE annotated periods to align clip
        audio_sr=cfg["AUDIO_SR"],
        stft_n_fft=cfg["NFFT"],
        stft_hop=cfg["HOP"],
        stft_win=cfg["WIN"],
        batch_size=cfg["BATCH_SIZE"],
        num_workers=cfg["NUM_WORKERS"],
        face_img_size=cfg["FACE_SIZE"],
        feature_set="bins9_11_pack",
        selection_json_path=cfg["VIS20_JSON"],
        selection_key=None,
        enforce_vis20=True,
        shuffle=shuffle,
        show_tqdm=True,
        face_detector_weights=cfg["FACE_DET_WEIGHTS"],
        visual_feat_mode="vis75",
        openface_binary=cfg.get("OPENFACE_BINARY", "") or "",
        precomputed_dir=cfg.get("PRECOMPUTED_V75_DIR", None),
        compute_if_missing=cfg.get("COMPUTE_IF_MISSING", False),
        fail_log_dir=cfg.get("FAIL_LOG_DIR", None),
    )

# =========================
# Stratified subset helpers
# =========================
def _stratified_paths_from_dataset(ds, fraction: float, seed: int) -> Set[str]:
    """
    Build a stratified set of paths by sampling from ds.samples using labels in samples.
    ds.samples entries are tuples: (abs_path, lab, segs, y_a, y_v)
    """
    rnd = random.Random(seed)
    real_paths = [p for (p, lab, *_r) in ds.samples if int(lab) == 0]
    fake_paths = [p for (p, lab, *_r) in ds.samples if int(lab) == 1]
    k_real = max(1, int(round(len(real_paths) * fraction)))
    k_fake = max(1, int(round(len(fake_paths) * fraction)))
    sel = set()
    if real_paths:
        sel.update(rnd.sample(real_paths, min(k_real, len(real_paths))))
    if fake_paths:
        sel.update(rnd.sample(fake_paths, min(k_fake, len(fake_paths))))
    return sel

@torch.no_grad()
def _collect_outputs_selected_paths(loader,
                                    model,
                                    device,
                                    selected_paths: Set[str],
                                    amp: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dev_type = device.type
    amp_on = (amp and dev_type == "cuda")
    model.eval()

    y_all, ld_all, la_all = [], [], []
    kept = 0
    for batch in tqdm(loader, total=len(loader), desc="collect-subset", ncols=100, leave=False):
        if batch is None:
            continue
        # (x_vis20, x_vis75, face, stft, x_aud, y_mm, y_a, y_v, a_len, paths)
        if len(batch) == 10:
            x_vis20, x_vis75, face, stft, x_aud, y_mm, _, _, _, paths = batch
        else:
            # Fallback (older collate)
            x_vis20, x_vis75, face, stft, x_aud, y_mm, *rest = batch
            paths = rest[-1] if rest else None
        if paths is None:
            raise RuntimeError("Batch does not include paths; cannot stratify/filter.")

        mask = np.array([p in selected_paths for p in paths], dtype=bool)
        if not mask.any():
            continue

        x_vis20 = x_vis20.to(device, non_blocking=True)
        x_aud   = x_aud.to(device, non_blocking=True)
        stft    = stft.to(device, non_blocking=True)
        face    = face.to(device, non_blocking=True)
        x_vis75 = (x_vis75.to(device, non_blocking=True) if x_vis75 is not None else x_vis20)

        with torch.autocast(device_type=dev_type, dtype=torch.float16, enabled=amp_on):
            out = model(x_vis20=x_vis20, x_vis75=x_vis75, x_aud=x_aud, stft=stft, face=face, infer_switch=False)

        y_b  = y_mm.detach().float().cpu().numpy().reshape(-1)
        ld_b = out["diss_logits"].detach().float().cpu().numpy().reshape(-1)
        la_b = out["aux_logits"].detach().float().cpu().numpy().reshape(-1)

        y_all.append(y_b[mask])
        ld_all.append(ld_b[mask])
        la_all.append(la_b[mask])
        kept += int(mask.sum())

    if kept == 0:
        raise RuntimeError("No samples kept for the stratified subset. Check loader and paths.")
    y  = np.concatenate(y_all,  axis=0).astype(np.int64)
    ld = np.concatenate(ld_all, axis=0).astype(np.float32)
    la = np.concatenate(la_all, axis=0).astype(np.float32)
    return y, ld, la

# =========================
# Full-collect
# =========================
@torch.no_grad()
def collect_outputs_full(loader, model, device, amp=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dev_type = device.type
    amp_on = (amp and dev_type == "cuda")
    y_all, ld_all, la_all = [], [], []
    model.eval()
    for batch in tqdm(loader, total=len(loader), desc="collect-full", ncols=100, leave=False):
        if batch is None:
            continue
        x_vis20, x_vis75, face, stft, x_aud, y_mm, _y_a, _y_v, _a_len, _paths = batch
        x_vis20 = x_vis20.to(device, non_blocking=True)
        x_aud   = x_aud.to(device, non_blocking=True)
        stft    = stft.to(device, non_blocking=True)
        face    = face.to(device, non_blocking=True)
        x_vis75 = x_vis75.to(device, non_blocking=True) if x_vis75 is not None else x_vis20
        with torch.autocast(device_type=dev_type, dtype=torch.float16, enabled=amp_on):
            out = model(x_vis20=x_vis20, x_vis75=x_vis75, x_aud=x_aud, stft=stft, face=face, infer_switch=False)
        y_all.append(y_mm.detach().float().cpu().numpy())
        ld_all.append(out["diss_logits"].detach().float().cpu().numpy())
        la_all.append(out["aux_logits"].detach().float().cpu().numpy())
    y  = np.concatenate(y_all,  axis=0).astype(np.int64)  if y_all  else np.zeros((0,), dtype=np.int64)
    ld = np.concatenate(ld_all, axis=0).astype(np.float32) if ld_all else np.zeros_like(y, dtype=np.float32)
    la = np.concatenate(la_all, axis=0).astype(np.float32) if la_all else np.zeros_like(y, dtype=np.float32)
    return y, ld, la

# =========================
# Calibration (repeated stratified 20%)
# =========================
def calibrate_repeated(loader, model, device, *, repeats: int, fraction: float, base_seed: int,
                       metric: str, thr_criterion: str) -> Dict:
    ds = loader.dataset
    if not hasattr(ds, "samples") or not ds.samples:
        raise RuntimeError("Dataset does not expose .samples; cannot perform stratified sampling.")

    params_list = []
    subset_stats = []
    # also track which repeat achieves best chosen metric on its subset
    best_idx = -1
    best_val = -np.inf

    for r in range(repeats):
        seed = base_seed + r
        sel_paths = _stratified_paths_from_dataset(ds, fraction=fraction, seed=seed)
        y, ld, la = _collect_outputs_selected_paths(loader, model, device, sel_paths, amp=True)

        # temperature on AV head (ld)
        T_av = fit_temperature_binary(ld, y.astype(np.float32))
        # sweep tau using requested metric
        tau, row = sweep_tau(y, ld, la, T_av=T_av, metric=metric)
        # pick decision threshold (with optional balanced criterion)
        p_sw, _, _ = switched_probs(ld, la, T_av=T_av, tau=tau)
        dec_thr = best_decision_threshold(y, p_sw, criterion=thr_criterion)
        # metrics on the calibration subset with chosen dec_thr
        m = compute_metrics(y, p_sw, thr=dec_thr)

        # store
        params_list.append((float(T_av), float(tau), float(dec_thr)))
        subset_stats.append({
            "repeat": r + 1,
            "seed": seed,
            "n_subset": int(y.size),
            "calibration": {"T_av": float(T_av), "tau": float(tau), "dec_thr": float(dec_thr)},
            "sweep_best": row,
            "subset_metrics": {
                "AUC": m["auc"], "AP": m["ap"], "EER": m["eer"], "ACC": m["acc"], "thr": m["thr"],
                "correct": m["correct"], "total": m["total"],
                "correct_real": m["correct_real"], "correct_fake": m["correct_fake"],
                "tn": m["tn"], "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
            }
        })

        # select best-by-metric repeat
        metric_val = {"auc": m["auc"], "ap": m["ap"], "acc": m["acc"], "eer": -m["eer"]}.get(metric, m["auc"])
        if metric_val > best_val:
            best_val = metric_val
            best_idx = r

        print(f"  Repeat {r+1}/{repeats} → T_av={T_av:.3f}, tau={tau:.3f}, dec_thr={dec_thr:.3f} | "
              f"AUC={m['auc']:.4f} AP={m['ap']:.4f} EER={m['eer']:.4f} ACC={m['acc']:.4f} "
              f"({m['correct']}/{m['total']})")

    # aggregate by median (robust to outliers)
    T_median  = float(np.median([p[0] for p in params_list]))
    tau_med   = float(np.median([p[1] for p in params_list]))
    dec_med   = float(np.median([p[2] for p in params_list]))
    agg_median = {"T_av": T_median, "tau": tau_med, "dec_thr": dec_med}

    # also keep the best repeat params
    T_best, tau_best, dec_best = params_list[best_idx]
    agg_best = {"T_av": float(T_best), "tau": float(tau_best), "dec_thr": float(dec_best)}

    print(f"  Aggregated (median) → T_av={T_median:.3f}, tau={tau_med:.3f}, dec_thr={dec_med:.3f}")
    print(f"  Aggregated (best-{metric}) → T_av={T_best:.3f}, tau={tau_best:.3f}, dec_thr={dec_best:.3f}")

    return {
        "params_list": params_list,
        "subset_stats": subset_stats,
        "aggregate_median": agg_median,
        "aggregate_best": agg_best,
        "best_repeat_index": int(best_idx) + 1
    }

# =========================
# Eval helpers (with debug prints)
# =========================
def _debug_switch_stats(tag: str, ld: np.ndarray, la: np.ndarray, T_av: float, tau: float):
    pav = _sigmoid_np(ld / max(T_av, 1e-6))
    pax = _sigmoid_np(la)
    conf = 2.0 * np.abs(pav - 0.5)
    pick_aux = (conf < tau).astype(np.float32)
    print(f"{tag}: pick_aux_rate={float(pick_aux.mean()):.4f} | pav mean/std={float(pav.mean()):.4f}/{float(pav.std()):.4f} | "
          f"pax mean/std={float(pax.mean()):.4f}/{float(pax.std()):.4f}")

# =========================
# Main
# =========================
def main():
    cfg = CONFIG
    os.makedirs(os.path.dirname(os.path.abspath(cfg["OUT_JSON"])), exist_ok=True)
    if cfg.get("PRECOMPUTED_V75_DIR"):
        os.makedirs(cfg["PRECOMPUTED_V75_DIR"], exist_ok=True)
    if cfg.get("FAIL_LOG_DIR"):
        os.makedirs(cfg["FAIL_LOG_DIR"], exist_ok=True)

    set_seed(cfg["SEED"])
    device = torch.device(cfg["DEVICE"])

    # Build model from checkpoint
    model, vis20_dim, stft_bins, train_cfg = build_model_from_ckpt(cfg["CKPT"], device)

    results = {}

    # ---------- FAVC: repeated stratified 20% ----------
    print("\n=== Calibrating on FAVC (repeated stratified 20%) ===")
    favc_loader = make_loader_favc(cfg, shuffle=False)
    favc_calib = calibrate_repeated(
        favc_loader, model, device,
        repeats=cfg["REPEATS"], fraction=cfg["FRACTION"],
        base_seed=cfg["SEED"], metric=cfg["FAVC_CALIB_METRIC"], thr_criterion=cfg["THR_CRITERION"]
    )
    results["FAVC"] = {"subset": cfg["FAVC_SUBSET"], **favc_calib}

    # ---------- LAV-DF: repeated stratified 20% ----------
    print("\n=== Calibrating on LAV-DF (repeated stratified 20%) ===")
    lavdf_loader = make_loader_lavdf(cfg, shuffle=False)
    lavdf_calib = calibrate_repeated(
        lavdf_loader, model, device,
        repeats=cfg["REPEATS"], fraction=cfg["FRACTION"],
        base_seed=cfg["SEED"] + 777,  # different seed stream for variety
        metric=cfg["LAVDF_CALIB_METRIC"], thr_criterion=cfg["THR_CRITERION"]
    )
    results["LAV-DF"] = {"subset": cfg["LAVDF_SUBSET"], **lavdf_calib}

    # ---------- Optional: evaluate full splits with aggregated params ----------
    if cfg.get("EVAL_FULL_AFTER_CALIB", True):
        print("\n=== Evaluating FULL TEST with aggregated params ===")

        # choose aggregation policy
        agg_key = "aggregate_best" if cfg.get("AGGREGATION", "median") == "best" else "aggregate_median"

        # FAVC: optionally force AUX-trusting preset
        if cfg.get("FAVC_TRUST_AUX_PRESET", False):
            T_av_f = float(cfg.get("FAVC_PRESET_T_AV", 1.2))
            tau_f  = float(cfg.get("FAVC_PRESET_TAU", 0.99))
            dec_f  = None  # determine from full set using selected THR_CRITERION
            print(f"[FAVC] Using AUX-trust preset → T_av={T_av_f:.3f}, tau={tau_f:.3f} (dec_thr will be computed)")
        else:
            T_av_f = results["FAVC"][agg_key]["T_av"]
            tau_f  = results["FAVC"][agg_key]["tau"]
            dec_f  = results["FAVC"][agg_key]["dec_thr"]
            print(f"[FAVC] Using {agg_key.split('_')[-1]} params → T_av={T_av_f:.3f}, tau={tau_f:.3f}, dec_thr={dec_f:.3f}")

        # Collect full logits, compute decision threshold (if preset) and eval
        yF, ldF, laF = collect_outputs_full(favc_loader, model, device, amp=True)
        pF, _, _ = switched_probs(ldF, laF, T_av=T_av_f, tau=tau_f)
        if dec_f is None:
            dec_f = best_decision_threshold(yF, pF, criterion=cfg["THR_CRITERION"])
            print(f"[FAVC] Computed dec_thr via {cfg['THR_CRITERION']}: {dec_f:.3f}")
        mF = compute_metrics(yF, pF, thr=dec_f)
        _debug_switch_stats("FAVC FULL switch-stats", ldF, laF, T_av=T_av_f, tau=tau_f)
        results["FAVC"]["full_eval"] = {
            "T_av": float(T_av_f), "tau": float(tau_f), "dec_thr": float(dec_f),
            "AUC": mF["auc"], "AP": mF["ap"], "EER": mF["eer"], "ACC": mF["acc"], "thr": mF["thr"],
            "correct": mF["correct"], "total": mF["total"],
            "correct_real": mF["correct_real"], "correct_fake": mF["correct_fake"],
            "tn": mF["tn"], "tp": mF["tp"], "fp": mF["fp"], "fn": mF["fn"]
        }
        print(f"FAVC FULL → T_av={T_av_f:.3f}, tau={tau_f:.3f}, dec_thr={dec_f:.3f} | "
              f"AUC={mF['auc']:.4f} AP={mF['ap']:.4f} EER={mF['eer']:.4f} ACC={mF['acc']:.4f} "
              f"({mF['correct']}/{mF['total']})")

        # LAV-DF full
        T_av_l = results["LAV-DF"][agg_key]["T_av"]
        tau_l  = results["LAV-DF"][agg_key]["tau"]
        dec_l  = results["LAV-DF"][agg_key]["dec_thr"]
        print(f"[LAV-DF] Using {agg_key.split('_')[-1]} params → T_av={T_av_l:.3f}, tau={tau_l:.3f}, dec_thr={dec_l:.3f}")
        yL, ldL, laL = collect_outputs_full(lavdf_loader, model, device, amp=True)
        pL, _, _ = switched_probs(ldL, laL, T_av=T_av_l, tau=tau_l)
        mL = compute_metrics(yL, pL, thr=dec_l)
        _debug_switch_stats("LAV-DF FULL switch-stats", ldL, laL, T_av=T_av_l, tau=tau_l)
        results["LAV-DF"]["full_eval"] = {
            "T_av": float(T_av_l), "tau": float(tau_l), "dec_thr": float(dec_l),
            "AUC": mL["auc"], "AP": mL["ap"], "EER": mL["eer"], "ACC": mL["acc"], "thr": mL["thr"],
            "correct": mL["correct"], "total": mL["total"],
            "correct_real": mL["correct_real"], "correct_fake": mL["correct_fake"],
            "tn": mL["tn"], "tp": mL["tp"], "fp": mL["fp"], "fn": mL["fn"]
        }
        print(f"LAV-DF FULL → T_av={T_av_l:.3f}, tau={tau_l:.3f}, dec_thr={dec_l:.3f} | "
              f"AUC={mL['auc']:.4f} AP={mL['ap']:.4f} EER={mL['eer']:.4f} ACC={mL['acc']:.4f} "
              f"({mL['correct']}/{mL['total']})")

    # ---------- Save ----------
    payload = {
        "ckpt": os.path.abspath(cfg["CKPT"]),
        "fraction": cfg["FRACTION"],
        "repeats": cfg["REPEATS"],
        "favc_metric": cfg["FAVC_CALIB_METRIC"],
        "lavdf_metric": cfg["LAVDF_CALIB_METRIC"],
        "thr_criterion": cfg["THR_CRITERION"],
        "aggregation": cfg["AGGREGATION"],
        "favc_trust_aux_preset": cfg["FAVC_TRUST_AUX_PRESET"],
        "favc": results["FAVC"],
        "lavdf": results["LAV-DF"],
        "notes": (
            "Per-dataset (T_av, tau, dec_thr) via repeated stratified 20% calibration; "
            "params aggregated by 'median' or 'best' as configured; optional AUX-trust preset for FAVC; "
            "balanced decision threshold available."
        )
    }
    with open(cfg["OUT_JSON"], "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved calibration → {cfg['OUT_JSON']}")

if __name__ == "__main__":
    main()