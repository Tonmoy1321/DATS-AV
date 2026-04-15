import os
import json
import time
import random
import warnings
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from tqdm import tqdm

os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("ULTRALYTICS_VERBOSITY", "0")
os.environ.setdefault("WANDB_SILENT", "true")
warnings.filterwarnings("ignore")

from av_dissonance_with_aux_ensemble_model import build_dissonance_dual_model
from unified_av_dataloader import get_unified_av_dataloader

CONFIG = {
    # ---- data roots / metadata ----
    "AVDF1M_TRAIN_ROOT": "/media/rt0706/Media/VCBSL-Dataset/AV-Deepfake1M/train",
    "AVDF1M_TRAIN_JSON": "/media/rt0706/Media/VCBSL-Dataset/AV-Deepfake1M/train_metadata.json",
    "AVDF1M_VAL_ROOT":   "/media/rt0706/Media/VCBSL-Dataset/AV-Deepfake1M/val",
    "AVDF1M_VAL_JSON":   "/media/rt0706/Media/VCBSL-Dataset/AV-Deepfake1M/val_metadata.json",

    # ---- VIS-20 selection (used to slice VIS-75 to VIS-20 inside loader) ----
    "VIS20_JSON": "runs/visual75_crossdomain_selection.json",
    "VIS20_KEY": None,
    "ENFORCE_EXACT_20": True,

    # ---- dataloader / feature extraction controls ----
    "BATCH_SIZE": 2,
    "NUM_WORKERS": 2,
    "FRAMES_PER_CLIP": 25,
    "STRIDE": 1,
    "AUDIO_SR": 16000,
    "NFFT": 400,
    "HOP": 160,
    "WIN": 400,
    "FACE_SIZE": 224,
    "BALANCE_MINORITY": True,
    "USE_FAKE_PERIODS": True,
    "SHOW_TQDM": True,
    "FACE_DET_WEIGHTS": "yolov8n-face.pt",
    "OPENFACE_BINARY": "../OpenFace/build/bin/FeatureExtraction",
    "PRECOMPUTED_V75_DIR": None,
    "COMPUTE_IF_MISSING": True,
    "FAIL_LOG_DIR": "runs/fail_logs",

    # ---- model ----
    "EMB_DIM_AUDIO": 128,
    "EMB_DIM_VISFACE": 256,
    "HIDDEN_AUDIO": 256,
    "HEADS": 4,
    "LAYERS": 1,
    "ENC_DROPOUT": 0.1,
    "PE_MAX_LEN": 2000,
    "PE_DROPOUT": 0.0,
    "CLS_DROPOUT_AUDIO": 0.3,
    "FUSION_MODE": "gated",
    "FACE_PRETRAINED": True,
    "FACE_FREEZE_BACKBONE": False,
    "SWITCH_THR": 0.25,
    "LAMBDA_DISSONANCE_AUX": 0.25,
    "LAMBDA_TOTAL_BALANCING": 1.0,

    # ---- training ----
    "EPOCHS": 20,
    "LR": 2e-4,
    "WEIGHT_DECAY": 0.01,
    "BETAS": (0.9, 0.999),
    "MAX_GRAD_NORM": 5.0,
    "ACCUM_STEPS": 1,
    "AMP": True,
    "SEED": 42,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "OUT_DIR": "runs/avdf1m_dual_full_avdiss_visual75_vf",
    "SAVE_EVERY_EPOCH": False,

    # ---- calibration / monitoring ----
    "CALIB_EVERY": 1,
    "CALIB_METRIC": "auc",
}

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def _eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2.0)

def best_decision_threshold(y_true: np.ndarray, probs: np.ndarray, criterion: str = "youden") -> float:
    fpr, tpr, thr = roc_curve(y_true, probs)
    if criterion == "youden":
        j = tpr - fpr
        idx = np.argmax(j)
    elif criterion == "acc":
        accs = [(((probs >= t).astype(int) == y_true).mean()) for t in thr]
        idx = int(np.argmax(accs))
    else:
        idx = np.argmin(np.abs(tpr - (1 - fpr)))
    return float(thr[idx])

def compute_metrics(y_true: np.ndarray, logits_or_probs: np.ndarray,
                    already_probs: bool = False, thr: Optional[float] = 0.5) -> Dict[str, float]:
    probs = logits_or_probs.astype(np.float32) if already_probs else _sigmoid(logits_or_probs.astype(np.float32))
    multi = (len(np.unique(y_true)) > 1)
    if multi:
        auc = roc_auc_score(y_true, probs)
        ap  = average_precision_score(y_true, probs)
        eer = _eer(y_true, probs)
    else:
        auc = ap = eer = 0.5
    use_thr = 0.5 if (thr is None) else float(thr)
    preds = (probs >= use_thr).astype(np.int64)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    total = len(y_true)
    acc = (tp + tn) / max(1, total)
    return {
        "auc": float(auc), "ap": float(ap), "eer": float(eer),
        "acc": float(acc), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "total": int(total), "correct": int(tp + tn),
        "correct_real": tn, "correct_fake": tp,
        "thr": use_thr,
    }

def fmt_metrics(tag: str, m: Dict[str, float]) -> str:
    return (f"{tag} | AUC={m['auc']:.4f} AP={m['ap']:.4f} EER={m['eer']:.4f} "
            f"| acc@{m.get('thr', 0.5):.3f}={m['acc']:.4f} ({m['correct']}/{m['total']}) "
            f"| TN={m['correct_real']} TP={m['correct_fake']} FP={m['fp']} FN={m['fn']}")

# ------------------------------- LOADER --------------------------------------

def make_loader_avdf1m(cfg: dict, subset: str, *, shuffle: bool):
    root = cfg["AVDF1M_TRAIN_ROOT"] if subset == "train" else cfg["AVDF1M_VAL_ROOT"]
    meta = cfg["AVDF1M_TRAIN_JSON"] if subset == "train" else cfg["AVDF1M_VAL_JSON"]
    if not root or not meta:
        return None
    return get_unified_av_dataloader(
        mode="av_deepfake1m",
        subset=subset,
        root_dir=root,
        json_path=meta,
        frames_per_clip=cfg["FRAMES_PER_CLIP"],
        stride=cfg["STRIDE"],
        balance_minority=cfg["BALANCE_MINORITY"],
        use_fake_periods=cfg["USE_FAKE_PERIODS"],
        audio_sr=cfg["AUDIO_SR"],
        stft_n_fft=cfg["NFFT"],
        stft_hop=cfg["HOP"],
        stft_win=cfg["WIN"],
        batch_size=cfg["BATCH_SIZE"],
        num_workers=cfg["NUM_WORKERS"],
        face_img_size=cfg["FACE_SIZE"],
        feature_set="bins9_11_pack",
        selection_json_path=cfg["VIS20_JSON"],
        selection_key=cfg["VIS20_KEY"],
        enforce_vis20=cfg["ENFORCE_EXACT_20"],
        shuffle=shuffle,
        show_tqdm=cfg["SHOW_TQDM"],
        face_detector_weights=cfg["FACE_DET_WEIGHTS"],
        visual_feat_mode="vis75",
        openface_binary=cfg.get("OPENFACE_BINARY", ""),
        precomputed_dir=cfg.get("PRECOMPUTED_V75_DIR", None),
        compute_if_missing=cfg.get("COMPUTE_IF_MISSING", False),
        fail_log_dir=cfg.get("FAIL_LOG_DIR", None),
    )

# ------------------------------- MODEL ---------------------------------------

def build_model(cfg: dict, vis20_dim: int, stft_bins: int, device: torch.device):
    model, criterion = build_dissonance_dual_model(
        vis_dim=vis20_dim,
        aud_dim=75,
        stft_bins=stft_bins,
        emb_dim_audio=cfg["EMB_DIM_AUDIO"],
        emb_dim_visface=cfg["EMB_DIM_VISFACE"],
        hidden_audio=cfg["HIDDEN_AUDIO"],
        enc_heads=cfg["HEADS"],
        enc_layers=cfg["LAYERS"],
        enc_dropout=cfg["ENC_DROPOUT"],
        pe_max_len=cfg["PE_MAX_LEN"],
        pe_dropout=cfg["PE_DROPOUT"],
        cls_dropout_audio=cfg["CLS_DROPOUT_AUDIO"],
        fusion_mode=cfg["FUSION_MODE"],
        face_pretrained=cfg["FACE_PRETRAINED"],
        face_freeze_backbone=cfg["FACE_FREEZE_BACKBONE"],
        switch_threshold=cfg["SWITCH_THR"],
        lambda_aux=cfg["LAMBDA_DISSONANCE_AUX"],
        lambda_total_balancing=cfg["LAMBDA_TOTAL_BALANCING"],
    )
    model.to(device)
    return model, criterion

def save_ckpt(path: str, model: nn.Module, optim, epoch: int, config: dict, best_auc: float, scaler=None):
    payload = {
        "model": model.state_dict(),
        "optimizer": optim.state_dict(),
        "scaler": (scaler.state_dict() if scaler is not None else None),
        "epoch": epoch,
        "best_auc": best_auc,
        "config": config,
        "calibration": {
            "T_av": config.get("CALIB_T_AV", 1.0),
            "tau":  config.get("CALIB_TAU", config.get("SWITCH_THR", 0.25)),
            "dec_thr": config.get("CALIB_DEC_THR", 0.5),
        },
        "ts": time.time(),
    }
    torch.save(payload, path)

def _sigmoid_np(x): return 1.0 / (1.0 + np.exp(-x))

# ---- UPDATED: batch unpacker -------------------------------------------------
def _unpack_batch(batch):
    """
    New dataloader returns:
      x_vis20, x_vis75, faces, stft, x_aud,
      y_mm, y_a|None, y_v|None, a_len, paths
    Keep back-compat fallback if older tuple length shows up.
    """
    if len(batch) == 10:
        return batch
    elif len(batch) == 9:
        # (older: no per-modality) -> insert Nones for y_a,y_v
        x_vis20, x_vis75, faces, stft, x_aud, y_mm, a_len, paths = batch
        return x_vis20, x_vis75, faces, stft, x_aud, y_mm, None, None, a_len, paths
    elif len(batch) == 8:
        # very old: single visual tensor twice (keep shape)
        x_vis20, x_vis75, faces, stft, x_aud, y_mm, a_len, paths = batch
        return x_vis20, x_vis75, faces, stft, x_aud, y_mm, None, None, a_len, paths
    else:
        raise RuntimeError(f"Unexpected batch structure of length {len(batch)}")

@torch.no_grad()
def collect_outputs(loader, model, device, amp=True):
    dev_type = device.type
    amp_on = (amp and dev_type == "cuda")
    y_all, ld_all, la_all = [], [], []
    model.eval()
    for batch in tqdm(loader, total=len(loader), desc="collect", ncols=100, leave=False):
        x_vis20, x_vis75, face, stft, x_aud, y_mm, _y_a, _y_v, _a_len, _paths = _unpack_batch(batch)
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
    y  = np.concatenate(y_all,  axis=0).astype(np.int64)
    ld = np.concatenate(ld_all, axis=0).astype(np.float32)
    la = np.concatenate(la_all, axis=0).astype(np.float32)
    return y, ld, la

def switched_probs(ld, la, T_av=1.0, tau=0.25):
    pav = _sigmoid_np(ld / max(T_av, 1e-6))
    pax = _sigmoid_np(la)
    conf = 2.0 * np.abs(pav - 0.5)
    pick_aux = (conf < tau).astype(np.float32)
    pfinal = (1.0 - pick_aux) * pav + pick_aux * pax
    return pfinal, conf, pick_aux

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
        score = score_map[metric if metric in score_map else "auc"]
        if score > best_val:
            best_val = score
            best_tau = float(tau)
            best_row = {"tau": best_tau, "auc": float(auc), "ap": float(ap), "eer": float(eer), "acc": float(acc)}
    return best_tau, best_row

@torch.no_grad()
def run_eval(loader, model, device, amp=True, desc="eval",
             switch_thr: float = 0.25,
             use_external_switch: bool = False,
             T_av: float = 1.0,
             dec_thr: Optional[float] = 0.5):
    if loader is None:
        return None, None, None, None, None
    dev_type = device.type
    amp_on = (amp and dev_type == "cuda")
    logits_diss, logits_aux, logits_sw, labels_all = [], [], [], []
    model.eval()
    for batch in tqdm(loader, total=len(loader), desc=desc, ncols=100, leave=False):
        x_vis20, x_vis75, face, stft, x_aud, y_mm, _y_a, _y_v, _a_len, _paths = _unpack_batch(batch)
        x_vis20 = x_vis20.to(device, non_blocking=True)
        x_aud   = x_aud.to(device, non_blocking=True)
        stft    = stft.to(device, non_blocking=True)
        face    = face.to(device, non_blocking=True)
        x_vis75 = x_vis75.to(device, non_blocking=True) if x_vis75 is not None else x_vis20
        y       = y_mm.to(device, non_blocking=True).float()
        with torch.autocast(device_type=dev_type, dtype=torch.float16, enabled=amp_on):
            out = model(x_vis20=x_vis20, x_vis75=x_vis75, x_aud=x_aud, stft=stft, face=face,
                        infer_switch=(not use_external_switch), switch_threshold=switch_thr)
        logits_diss.append(out["diss_logits"].detach().float().cpu().numpy())
        logits_aux.append(out["aux_logits"].detach().float().cpu().numpy())
        if out.get("logits_switch") is not None and not use_external_switch:
            logits_sw.append(out["logits_switch"].detach().float().cpu().numpy())
        labels_all.append(y.detach().float().cpu().numpy())
    y  = np.concatenate(labels_all, axis=0) if labels_all else np.zeros((0,), dtype=np.float32)
    ld = np.concatenate(logits_diss, axis=0) if logits_diss else np.zeros_like(y)
    la = np.concatenate(logits_aux,   axis=0) if logits_aux   else np.zeros_like(y)
    if use_external_switch:
        p_sw, _, _ = switched_probs(ld, la, T_av=T_av, tau=switch_thr)
        m_main = compute_metrics(y.astype(np.int64), p_sw.astype(np.float32), already_probs=True, thr=dec_thr)
        ls = p_sw
    else:
        ls = np.concatenate(logits_sw, axis=0) if logits_sw else np.zeros((0,), dtype=np.float32)
        m_main = compute_metrics(y.astype(np.int64), ls.astype(np.float32), thr=0.5) if ls.size else None
    m_av  = compute_metrics(y.astype(np.int64), ld.astype(np.float32), thr=0.5) if ld.size else None
    m_aux = compute_metrics(y.astype(np.int64), la.astype(np.float32), thr=0.5) if la.size else None
    return m_main, m_av, m_aux, y, (ls if (ls is not None and len(ls) > 0) else None)

# train loop (passes per-modality labels to criterion)
def train_one_epoch(loader, model, criterion, optimizer, device, scaler, *,
                    amp=True, max_grad_norm=0.0, accum_steps=1):
    model.train()
    losses = []
    dev_type = device.type
    amp_on = (amp and dev_type == "cuda")
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, total=len(loader), desc="train", ncols=100, leave=False)
    for step, batch in enumerate(pbar, start=1):
        x_vis20, x_vis75, face, stft, x_aud, y_mm, y_a, y_v, _a_len, _paths = _unpack_batch(batch)
        x_vis20 = x_vis20.to(device, non_blocking=True)
        x_aud   = x_aud.to(device, non_blocking=True)
        stft    = stft.to(device, non_blocking=True)
        face    = face.to(device, non_blocking=True)
        x_vis75 = x_vis75.to(device, non_blocking=True) if x_vis75 is not None else x_vis20

        # labels dict: always multimodal; add per-modality if available
        labels = {"y_mm": (y_mm if torch.is_tensor(y_mm) else torch.tensor(y_mm)).to(device).float()}
        if y_a is not None and y_v is not None:
            labels["y_a"] = (y_a if torch.is_tensor(y_a) else torch.tensor(y_a)).to(device).float()
            labels["y_v"] = (y_v if torch.is_tensor(y_v) else torch.tensor(y_v)).to(device).float()

        with torch.autocast(device_type=dev_type, dtype=torch.float16, enabled=amp_on):
            out = model(x_vis20=x_vis20, x_vis75=x_vis75, x_aud=x_aud, stft=stft, face=face, infer_switch=False)
            # NEW: criterion expects labels dict (supports per-modality if provided)
            loss_dict = criterion(out, labels, infer_switch=False)
            loss = loss_dict["loss"] / max(1, accum_steps)

        if scaler is not None and amp_on:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if step % max(1, accum_steps) == 0:
            if scaler is not None and amp_on:
                if max_grad_norm and max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if max_grad_norm and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        losses.append(loss_dict["loss"].detach().item())
        pbar.set_postfix(
            loss=np.mean(losses[-50:]),
            diss_acc=float(loss_dict.get("acc_diss", 0.0)),
            aux_acc=float(loss_dict.get("acc_aux", 0.0)),
        )
    return float(np.mean(losses)) if losses else 0.0

def maybe_resume(cfg, model, optimizer, scaler, device):
    ckpt_path = os.path.join(cfg["OUT_DIR"], "ckpt_last.pt")
    if not os.path.isfile(ckpt_path):
        return 1, -1.0
    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["model"], strict=True)
    if payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    start_epoch = int(payload.get("epoch", 0)) + 1
    best_auc = float(payload.get("best_auc", -1.0))
    calib = payload.get("calibration", {}) or {}
    cfg["CALIB_T_AV"]   = float(calib.get("T_av", 1.0))
    cfg["CALIB_TAU"]    = float(calib.get("tau", cfg.get("SWITCH_THR", 0.25)))
    cfg["CALIB_DEC_THR"] = float(calib.get("dec_thr", 0.5))
    print(f"Resumed from {ckpt_path} at epoch={payload.get('epoch', 0)} → start_epoch={start_epoch}")
    print(f"Resume calib: T_av={cfg['CALIB_T_AV']:.3f}, tau={cfg['CALIB_TAU']:.3f}, dec_thr={cfg['CALIB_DEC_THR']:.3f}")
    return start_epoch, best_auc

def main():
    cfg = CONFIG
    set_seed(cfg["SEED"])
    os.makedirs(cfg["OUT_DIR"], exist_ok=True)
    if cfg.get("PRECOMPUTED_V75_DIR"):
        os.makedirs(cfg["PRECOMPUTED_V75_DIR"], exist_ok=True)
    if cfg.get("FAIL_LOG_DIR"):
        os.makedirs(cfg["FAIL_LOG_DIR"], exist_ok=True)

    device = torch.device(cfg["DEVICE"])

    train_loader = make_loader_avdf1m(cfg, subset="train", shuffle=True)
    if train_loader is None:
        raise RuntimeError("Train loader couldn't be built. Check AVDF1M_TRAIN_ROOT/JSON.")
    train_loader_eval = make_loader_avdf1m(cfg, subset="train", shuffle=False)
    val_loader = make_loader_avdf1m(cfg, subset="val", shuffle=False) \
        if (cfg["AVDF1M_VAL_ROOT"] and cfg["AVDF1M_VAL_JSON"]) else None

    # VIS-20 dim from dataset (align with loader's selection)
    sel_idx = getattr(train_loader.dataset, "sel_idx", None)
    if sel_idx is None or len(sel_idx) == 0:
        xb, *_ = next(iter(train_loader))
        vis20_dim = 20 if xb.shape[1] >= 20 else int(xb.shape[1])
    else:
        vis20_dim = int(len(sel_idx))

    stft_bins = cfg["NFFT"] // 2 + 1
    model, criterion = build_model(cfg, vis20_dim, stft_bins, device)

    optimizer = AdamW(model.parameters(), lr=cfg["LR"], betas=cfg["BETAS"], weight_decay=cfg["WEIGHT_DECAY"])
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg["AMP"] and device.type == "cuda"))

    # Calibration defaults
    cfg["CALIB_T_AV"]   = 1.0
    cfg["CALIB_TAU"]    = cfg["SWITCH_THR"]
    cfg["CALIB_DEC_THR"] = 0.5

    start_epoch, best_auc = maybe_resume(cfg, model, optimizer, scaler, device)
    best_epoch = start_epoch - 1 if best_auc >= 0 else -1

    history_path = os.path.join(cfg["OUT_DIR"], "train_history.json")
    if os.path.isfile(history_path):
        try:
            with open(history_path, "r") as f:
                history = json.load(f)
        except Exception:
            history = {"train": [], "val": [], "calib": []}
    else:
        history = {"train": [], "val": [], "calib": []}

    for epoch in range(start_epoch, cfg["EPOCHS"] + 1):
        print(f"\nEpoch {epoch}/{cfg['EPOCHS']}")
        train_loss = train_one_epoch(
            train_loader, model, criterion, optimizer, device, scaler,
            amp=cfg["AMP"], max_grad_norm=cfg["MAX_GRAD_NORM"], accum_steps=cfg["ACCUM_STEPS"]
        )

        # calibration
        if val_loader is not None and (cfg["CALIB_EVERY"] > 0) and (epoch % cfg["CALIB_EVERY"] == 0):
            y_val, ld_val, la_val = collect_outputs(val_loader, model, device, amp=cfg["AMP"])
            T_av = fit_temperature_binary(ld_val, y_val.astype(np.float32))
            best_tau, row = sweep_tau(y_val, ld_val, la_val, T_av=T_av, metric=cfg["CALIB_METRIC"])
            p_sw_val, _, _ = switched_probs(ld_val, la_val, T_av=T_av, tau=best_tau)
            dec_thr = best_decision_threshold(y_val, p_sw_val, criterion="youden")
            cfg["CALIB_T_AV"] = float(T_av)
            cfg["CALIB_TAU"]  = float(best_tau)
            cfg["CALIB_DEC_THR"] = float(dec_thr)
            history["calib"].append({"epoch": epoch, "T_av": float(T_av), "tau": float(best_tau),
                                     "dec_thr": float(dec_thr), **row})
            print(f"[CALIB] epoch={epoch} | T_av={T_av:.3f} | tau={best_tau:.3f} | dec_thr={dec_thr:.3f} | {row}")

        # eval on train (for monitoring) using calibrated external switch if we have val
        use_ext = (val_loader is not None)
        tau_eval = cfg.get("CALIB_TAU", cfg["SWITCH_THR"])
        T_av_eval = cfg.get("CALIB_T_AV", 1.0)
        dec_thr_eval = cfg.get("CALIB_DEC_THR", 0.5)

        m_main_t, m_av_t, m_aux_t, _, _ = run_eval(
            train_loader_eval, model, device, amp=cfg["AMP"], desc="eval-train",
            switch_thr=tau_eval, use_external_switch=use_ext, T_av=T_av_eval, dec_thr=dec_thr_eval
        )
        if m_main_t: print(fmt_metrics("TRAIN(main)", m_main_t))
        if m_av_t:   print(fmt_metrics("TRAIN(av_diss)",   m_av_t))
        if m_aux_t:  print(fmt_metrics("TRAIN(aux)",   m_aux_t))

        history["train"].append({
            "epoch": epoch,
            "loss": train_loss,
            **(m_main_t or {}),
            **({f"av_{k}": v for k, v in (m_av_t or {}).items()}),
            **({f"aux_{k}": v for k, v in (m_aux_t or {}).items()}),
        })

        # eval on val (with external switch based on calibrated tau/T_av)
        if val_loader is not None:
            m_main_v, m_av_v, m_aux_v, _, _ = run_eval(
                val_loader, model, device, amp=cfg["AMP"], desc="eval-val",
                switch_thr=tau_eval, use_external_switch=True, T_av=T_av_eval, dec_thr=dec_thr_eval
            )
            if m_main_v: print(fmt_metrics("VAL(main)", m_main_v))
            if m_av_v:   print(fmt_metrics("VAL(av_diss)",   m_av_v))
            if m_aux_v:  print(fmt_metrics("VAL(aux)",   m_aux_v))
            history["val"].append({
                "epoch": epoch,
                **(m_main_v or {}),
                **({f"av_{k}": v for k, v in (m_av_v or {}).items()}),
                **({f"aux_{k}": v for k, v in (m_aux_v or {}).items()}),
            })
            cur_auc = (m_main_v or m_av_v or m_aux_v or {"auc": 0.0})["auc"]
        else:
            cur_auc = (m_main_t or m_av_t or m_aux_t or {"auc": 0.0})["auc"]

        # checkpoints
        ckpt_last = os.path.join(cfg["OUT_DIR"], "ckpt_last.pt")
        save_ckpt(ckpt_last, model, optimizer, epoch, cfg, best_auc=max(best_auc, cur_auc), scaler=scaler)
        if cfg["SAVE_EVERY_EPOCH"]:
            save_ckpt(os.path.join(cfg["OUT_DIR"], f"ckpt_epoch_{epoch:03d}.pt"),
                      model, optimizer, epoch, cfg, best_auc=max(best_auc, cur_auc), scaler=scaler)

        if cur_auc > best_auc:
            best_auc = cur_auc
            best_epoch = epoch
            ckpt_best = os.path.join(cfg["OUT_DIR"], "ckpt_best_auc.pt")
            save_ckpt(ckpt_best, model, optimizer, epoch, cfg, best_auc=best_auc, scaler=scaler)
            print(f"✓ New best AUC={best_auc:.4f} at epoch {best_epoch}. Saved to {ckpt_best}")

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nTraining done. Best AUC={best_auc:.4f} (epoch {best_epoch}).")
    print(f"Artifacts in: {cfg['OUT_DIR']}")
    print("Files: ckpt_last.pt, ckpt_best_auc.pt, train_history.json")


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

if __name__ == "__main__":
    main()


# #-------------FakeAVCeleb -----------------------#

# # train_dissonance_dual_favc.py
# import os
# import csv
# import json
# import warnings
# from typing import Dict, Optional, Tuple, List

# import numpy as np
# import torch
# import torch.nn as nn
# from torch.optim import AdamW
# from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
# from tqdm import tqdm

# os.environ.setdefault("PYTHONWARNINGS", "ignore")
# os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# os.environ.setdefault("KMP_WARNINGS", "0")
# os.environ.setdefault("ULTRALYTICS_VERBOSITY", "0")
# os.environ.setdefault("WANDB_SILENT", "true")
# warnings.filterwarnings("ignore")

# from av_dissonance_with_aux_ensemble_model import build_dissonance_dual_model
# from unified_av_dataloader import get_unified_av_dataloader

# CONFIG = {
#     "FAVC_TRAIN_ROOT": "/media/rt0706/Media/VCBSL-Dataset/FakeAVCeleb",
#     "FAVC_TRAIN_CSV":  "../Dataset/favc_multimodal_data.csv",
#     "FAVC_VAL_ROOT":   "/media/rt0706/Media/VCBSL-Dataset/FakeAVCeleb",
#     "FAVC_VAL_CSV":    "../Dataset/favc_multimodal_data.csv",

#     "AUX_AUDIO_CSV":   "../Dataset/favc_audio_data.csv",
#     "AUX_VISUAL_CSV":  "../Dataset/favc_visual_data.csv",

#     "VIS20_JSON": "runs/visual75_crossdomain_selection.json",
#     "VIS20_KEY": None,
#     "ENFORCE_EXACT_20": True,

#     "BATCH_SIZE": 2,
#     "NUM_WORKERS": 0,
#     "FRAMES_PER_CLIP": 25,
#     "STRIDE": 1,
#     "AUDIO_SR": 16000,
#     "NFFT": 400,
#     "HOP": 160,
#     "WIN": 400,
#     "FACE_SIZE": 224,
#     "BALANCE_MINORITY": True,
#     "USE_FAKE_PERIODS": False,
#     "SHOW_TQDM": True,
#     "FACE_DET_WEIGHTS": "yolov8n-face.pt",
#     "OPENFACE_BINARY": "../OpenFace/build/bin/FeatureExtraction",
#     "PRECOMPUTED_V75_DIR": "runs/precomputed_vis75",
#     "COMPUTE_IF_MISSING": True,
#     "FAIL_LOG_DIR": "runs/fail_logs_train",

#     "EMB_DIM_AUDIO": 128,
#     "EMB_DIM_VISFACE": 256,
#     "HIDDEN_AUDIO": 256,
#     "HEADS": 4,
#     "LAYERS": 1,
#     "ENC_DROPOUT": 0.1,
#     "PE_MAX_LEN": 2000,
#     "PE_DROPOUT": 0.0,
#     "CLS_DROPOUT_AUDIO": 0.3,
#     "FUSION_MODE": "gated",
#     "FACE_PRETRAINED": True,
#     "FACE_FREEZE_BACKBONE": False,

#     "SWITCH_THR": 0.25,
#     "CALIB_EVERY": 1,
#     "CALIB_METRIC": "auc",
#     "CALIB_INIT_T_AV": 1.0,

#     "EPOCHS": 20,
#     "LR": 2e-4,
#     "WEIGHT_DECAY": 0.01,
#     "BETAS": (0.9, 0.999),
#     "MAX_GRAD_NORM": 5.0,
#     "ACCUM_STEPS": 1,
#     "AMP": True,
#     "SEED": 42,
#     "DEVICE": "cuda:0" if torch.cuda.is_available() else "cpu",
#     "OUT_DIR": "runs/favc_dual_avdiss_visual75_train",
#     "SAVE_EVERY_EPOCH": False,

#     "AUX_LOSS_W": 0.5
# }

# def set_seed(seed: int = 42):
#     import random
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

# def _sigmoid_np(x: np.ndarray) -> np.ndarray:
#     return 1.0 / (1.0 + np.exp(-x))

# def _eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
#     fpr, tpr, _ = roc_curve(y_true, y_score)
#     fnr = 1.0 - tpr
#     i = np.nanargmin(np.abs(fnr - fpr))
#     return float((fpr[i] + fnr[i]) / 2.0)

# def compute_metrics(y_true: np.ndarray, logits_or_probs: np.ndarray,
#                     already_probs: bool = False, thr: Optional[float] = 0.5) -> Dict[str, float]:
#     probs = logits_or_probs.astype(np.float32) if already_probs else _sigmoid_np(logits_or_probs.astype(np.float32))
#     multi = (len(np.unique(y_true)) > 1)
#     if multi:
#         auc = roc_auc_score(y_true, probs)
#         ap  = average_precision_score(y_true, probs)
#         eer = _eer(y_true, probs)
#     else:
#         auc = ap = eer = 0.5
#     use_thr = 0.5 if (thr is None) else float(thr)
#     preds = (probs >= use_thr).astype(np.int64)
#     tp = int(((preds == 1) & (y_true == 1)).sum())
#     tn = int(((preds == 0) & (y_true == 0)).sum())
#     fp = int(((preds == 1) & (y_true == 0)).sum())
#     fn = int(((preds == 0) & (y_true == 1)).sum())
#     total = len(y_true)
#     acc = (tp + tn) / max(1, total)
#     return {
#         "auc": float(auc), "ap": float(ap), "eer": float(eer),
#         "acc": float(acc), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
#         "total": int(total), "correct": int(tp + tn),
#         "correct_real": tn, "correct_fake": tp,
#         "thr": use_thr,
#     }

# def fmt_metrics(tag: str, m: Dict[str, float], y_true: Optional[np.ndarray] = None) -> str:
#     line = (f"{tag} | AUC={m['auc']:.4f} AP={m['ap']:.4f} EER={m['eer']:.4f} "
#             f"| acc@{m.get('thr', 0.5):.3f}={m['acc']:.4f} ({m['correct']}/{m['total']}) "
#             f"| TN={m['correct_real']} TP={m['correct_fake']} FP={m['fp']} FN={m['fn']}")
#     if y_true is not None and y_true.size:
#         n_real = int((y_true == 0).sum()); n_fake = int((y_true == 1).sum())
#         line += f" | real: {m['correct_real']}/{n_real} | fake: {m['correct_fake']}/{n_fake}"
#     return line

# def _unpack_batch_safe(batch):
#     if batch is None:
#         return None
#     if not isinstance(batch, (list, tuple)):
#         return None
#     L = len(batch)
#     if L == 10:
#         return batch
#     elif L in (8, 9):
#         try:
#             x_vis20, x_vis75, face, stft, x_aud, y_mm, a_len, paths = batch[:8]
#             return x_vis20, x_vis75, face, stft, x_aud, y_mm, None, None, a_len, paths
#         except Exception:
#             return None
#     else:
#         return None

# def _get_aux_logits_from_out(out: Dict[str, torch.Tensor]) -> torch.Tensor:
#     if "aux_logits" in out:
#         return out["aux_logits"]
#     if ("aux_a_logits" in out) and ("aux_v_logits" in out):
#         z_a = out["aux_a_logits"]
#         z_v = out["aux_v_logits"]
#         g   = out.get("aux_gate", None)
#         if g is not None:
#             while g.ndim < z_a.ndim:
#                 g = g.unsqueeze(-1)
#             return (g >= 0.5).float() * z_a + (g < 0.5).float() * z_v
#         return torch.where(z_a >= z_v, z_a, z_v)
#     raise KeyError("No AUX logits found")

# def _norm_key(p: str) -> str:
#     if p is None:
#         return ""
#     p2 = p.replace("\\", "/")
#     p2 = os.path.normpath(p2)
#     return p2.lower()

# def _read_label_map(csv_path: str) -> Dict[str, int]:
#     if not csv_path or not os.path.isfile(csv_path):
#         return {}
#     with open(csv_path, "r", newline="", encoding="utf-8") as f:
#         reader = csv.DictReader(f)
#         cols = [c.lower().strip() for c in reader.fieldnames or []]
#         path_cols = [c for c in cols if "path" in c or "file" in c]
#         label_cols = [c for c in cols if c in ("label", "target", "y", "cls")]
#         rows = list(reader)
#     data = {}
#     for r in rows:
#         rp = {k.lower().strip(): v for k, v in r.items()}
#         path_val = ""
#         for c in path_cols or []:
#             if c in rp and rp[c]:
#                 path_val = rp[c]
#                 break
#         lab_val = None
#         for c in label_cols or []:
#             if c in rp and rp[c] != "":
#                 lab_val = rp[c]
#                 break
#         if path_val == "" or lab_val is None:
#             continue
#         try:
#             y = int(lab_val)
#         except Exception:
#             try:
#                 y = 1 if str(lab_val).strip().lower() in ("1", "true", "fake", "pos") else 0
#             except Exception:
#                 continue
#         key_full = _norm_key(path_val)
#         key_base = os.path.basename(key_full)
#         data[key_full] = int(y)
#         if key_base:
#             data[key_base] = int(y)
#     return data

# class AuxLabeler:
#     def __init__(self, audio_csv: Optional[str], visual_csv: Optional[str]):
#         self.audio_map = _read_label_map(audio_csv or "")
#         self.visual_map = _read_label_map(visual_csv or "")

#     def _lookup_one(self, path: str, which: str) -> Optional[int]:
#         m = self.audio_map if which == "a" else self.visual_map
#         if not m:
#             return None
#         k = _norm_key(path)
#         b = os.path.basename(k)
#         if k in m:
#             return m[k]
#         if b in m:
#             return m[b]
#         return None

#     def batch_lookup(self, paths: List[str]) -> Tuple[List[Optional[int]], List[Optional[int]]]:
#         ya, yv = [], []
#         for p in paths:
#             ya.append(self._lookup_one(p, "a"))
#             yv.append(self._lookup_one(p, "v"))
#         return ya, yv

# @torch.no_grad()
# def collect_outputs(loader, model, device, amp=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     dev_type = device.type
#     amp_on = (amp and dev_type == "cuda")
#     y_all, ld_all, la_all = [], [], []
#     model.eval()
#     iterator = tqdm(loader, total=len(loader), desc="collect", ncols=100, leave=False)
#     for batch in iterator:
#         unpacked = _unpack_batch_safe(batch)
#         if unpacked is None:
#             continue
#         x_vis20, x_vis75, face, stft, x_aud, y_mm, _y_a, _y_v, _a_len, _paths = unpacked
#         x_vis20 = x_vis20.to(device, non_blocking=True)
#         x_aud   = x_aud.to(device, non_blocking=True)
#         stft    = stft.to(device, non_blocking=True)
#         face    = face.to(device, non_blocking=True)
#         x_vis75 = x_vis75.to(device, non_blocking=True) if x_vis75 is not None else x_vis20
#         with torch.autocast(device_type=dev_type, dtype=torch.float16, enabled=amp_on):
#             out = model(x_vis20=x_vis20, x_vis75=x_vis75, x_aud=x_aud, stft=stft, face=face, infer_switch=False)
#         y_all.append(y_mm.detach().float().cpu().numpy())
#         ld_all.append(out["diss_logits"].detach().float().cpu().numpy())
#         la_all.append(_get_aux_logits_from_out(out).detach().float().cpu().numpy())
#     y  = np.concatenate(y_all,  axis=0).astype(np.int64)  if y_all  else np.zeros((0,), dtype=np.int64)
#     ld = np.concatenate(ld_all, axis=0).astype(np.float32) if ld_all else np.zeros_like(y, dtype=np.float32)
#     la = np.concatenate(la_all, axis=0).astype(np.float32) if la_all else np.zeros_like(y, dtype=np.float32)
#     return y, ld, la

# def switched_probs(ld: np.ndarray, la: np.ndarray, T_av=1.0, tau=0.25):
#     pav = _sigmoid_np(ld / max(T_av, 1e-6))
#     pax = _sigmoid_np(la)
#     conf = 2.0 * np.abs(pav - 0.5)
#     pick_aux = (conf < tau).astype(np.float32)
#     pfinal = (1.0 - pick_aux) * pav + pick_aux * pax
#     return pfinal, conf, pick_aux

# def fit_temperature_binary(logits: np.ndarray, labels: np.ndarray, max_iter: int = 50) -> float:
#     if logits.size == 0 or len(np.unique(labels)) < 2:
#         return 1.0
#     T = 1.0
#     for _ in range(max_iter):
#         z = logits / max(T, 1e-6)
#         p = _sigmoid_np(z)
#         grad = np.sum((p - labels) * (-logits / (max(T, 1e-6) ** 2)))
#         hess = np.sum(p * (1 - p) * (logits**2) / (max(T, 1e-6) ** 4)) + 1e-6
#         step = grad / hess
#         T_new = max(1e-3, T - step)
#         if abs(T_new - T) < 1e-4:
#             T = T_new
#             break
#         T = T_new
#     return float(max(1e-3, min(100.0, T)))

# def sweep_tau(y: np.ndarray, ld: np.ndarray, la: np.ndarray, T_av: float, metric: str = "auc") -> Tuple[float, Dict[str, float]]:
#     if y.size == 0:
#         return 0.25, {"auc": 0.5, "ap": 0.5}
#     taus = np.linspace(0.0, 0.5, 51)
#     best_tau, best_val, best_ap = 0.25, -1.0, 0.5
#     for t in taus:
#         p, _, _ = switched_probs(ld, la, T_av=T_av, tau=t)
#         try:
#             auc = roc_auc_score(y, p)
#             ap  = average_precision_score(y, p)
#         except Exception:
#             auc, ap = 0.5, 0.5
#         score = auc if metric.lower() == "auc" else ap
#         if score > best_val:
#             best_val, best_tau, best_ap = score, float(t), float(ap)
#     return best_tau, {"auc": float(best_val if metric.lower()=="auc" else roc_auc_score(y, switched_probs(ld, la, T_av, best_tau)[0])), "ap": best_ap}

# def best_decision_threshold(y: np.ndarray, probs: np.ndarray, criterion: str = "youden") -> float:
#     if y.size == 0:
#         return 0.5
#     fpr, tpr, thr = roc_curve(y, probs)
#     j = tpr - fpr
#     i = int(np.nanargmax(j))
#     return float(np.clip(thr[i], 0.0, 1.0))

# def make_loader_favc(cfg: dict, subset: str, *, shuffle: bool):
#     if subset == "train":
#         root = cfg["FAVC_TRAIN_ROOT"]
#         csv_path = cfg["FAVC_TRAIN_CSV"]
#     else:
#         root = cfg["FAVC_VAL_ROOT"]
#         csv_path = cfg["FAVC_VAL_CSV"]
#     return get_unified_av_dataloader(
#         mode="fakeavceleb",
#         subset=subset,
#         root_dir=root,
#         csv_path=csv_path,
#         frames_per_clip=cfg["FRAMES_PER_CLIP"],
#         stride=cfg["STRIDE"],
#         balance_minority=cfg["BALANCE_MINORITY"],
#         use_fake_periods=cfg["USE_FAKE_PERIODS"],
#         audio_sr=cfg["AUDIO_SR"],
#         stft_n_fft=cfg["NFFT"],
#         stft_hop=cfg["HOP"],
#         stft_win=cfg["WIN"],
#         batch_size=cfg["BATCH_SIZE"],
#         num_workers=cfg["NUM_WORKERS"],
#         face_img_size=cfg["FACE_SIZE"],
#         feature_set="bins9_11_pack",
#         selection_json_path=cfg["VIS20_JSON"],
#         selection_key=cfg["VIS20_KEY"],
#         enforce_vis20=cfg["ENFORCE_EXACT_20"],
#         shuffle=shuffle,
#         show_tqdm=cfg["SHOW_TQDM"],
#         face_detector_weights=cfg["FACE_DET_WEIGHTS"],
#         visual_feat_mode="vis75",
#         openface_binary=cfg.get("OPENFACE_BINARY", ""),
#         precomputed_dir=cfg.get("PRECOMPUTED_V75_DIR", None),
#         compute_if_missing=cfg.get("COMPUTE_IF_MISSING", False),
#         fail_log_dir=cfg.get("FAIL_LOG_DIR", None),
#     )

# def build_model(cfg: dict, vis20_dim: int, stft_bins: int, device: torch.device):
#     model, _criterion = build_dissonance_dual_model(
#         vis_dim=vis20_dim,
#         aud_dim=75,
#         stft_bins=stft_bins,
#         emb_dim_audio=cfg["EMB_DIM_AUDIO"],
#         emb_dim_visface=cfg["EMB_DIM_VISFACE"],
#         hidden_audio=cfg["HIDDEN_AUDIO"],
#         enc_heads=cfg["HEADS"],
#         enc_layers=cfg["LAYERS"],
#         enc_dropout=cfg["ENC_DROPOUT"],
#         pe_max_len=cfg["PE_MAX_LEN"],
#         pe_dropout=cfg["PE_DROPOUT"],
#         cls_dropout_audio=cfg["CLS_DROPOUT_AUDIO"],
#         fusion_mode=cfg["FUSION_MODE"],
#         face_pretrained=cfg["FACE_PRETRAINED"],
#         face_freeze_backbone=cfg["FACE_FREEZE_BACKBONE"],
#     )
#     model.to(device)
#     criterion = nn.BCEWithLogitsLoss()
#     return model, criterion

# def train_one_epoch(loader, model, criterion, optimizer, device, scaler,
#                     aux_labeler: AuxLabeler, aux_loss_w: float,
#                     amp=True, max_grad_norm=5.0, accum_steps=1) -> float:
#     model.train()
#     dev_type = device.type
#     amp_on = (amp and dev_type == "cuda")
#     running = 0.0
#     n_steps = 0
#     optimizer.zero_grad(set_to_none=True)
#     iterator = tqdm(loader, desc="train", ncols=100)
#     bce = nn.BCEWithLogitsLoss()
#     for step, batch in enumerate(iterator, 1):
#         unpacked = _unpack_batch_safe(batch)
#         if unpacked is None:
#             continue
#         x_vis20, x_vis75, face, stft, x_aud, y_mm, _y_a, _y_v, _a_len, paths = unpacked
#         x_vis20 = x_vis20.to(device, non_blocking=True)
#         x_aud   = x_aud.to(device, non_blocking=True)
#         stft    = stft.to(device, non_blocking=True)
#         face    = face.to(device, non_blocking=True)
#         x_vis75 = x_vis75.to(device, non_blocking=True) if x_vis75 is not None else x_vis20
#         y_mm    = y_mm.to(device, dtype=torch.float32, non_blocking=True).view(-1, 1)

#         ya_list, yv_list = aux_labeler.batch_lookup(list(paths))
#         ya = torch.tensor([0 if v is None else int(v) for v in ya_list], dtype=torch.float32, device=device).view(-1, 1)
#         yv = torch.tensor([0 if v is None else int(v) for v in yv_list], dtype=torch.float32, device=device).view(-1, 1)

#         with torch.autocast(device_type=dev_type, dtype=torch.float16, enabled=amp_on):
#             out = model(x_vis20=x_vis20, x_vis75=x_vis75, x_aud=x_aud, stft=stft, face=face, infer_switch=False)
#             z_main = out["diss_logits"]
#             if z_main.dim() == 1:
#                 z_main = z_main.view(-1, 1)
#             loss_main = criterion(z_main, y_mm)

#             loss_aux = 0.0
#             if "aux_a_logits" in out:
#                 za = out["aux_a_logits"]
#                 if za.dim() == 1: za = za.view(-1, 1)
#                 loss_aux = loss_aux + bce(za, ya)
#             if "aux_v_logits" in out:
#                 zv = out["aux_v_logits"]
#                 if zv.dim() == 1: zv = zv.view(-1, 1)
#                 loss_aux = loss_aux + bce(zv, yv)

#             loss = loss_main + aux_loss_w * loss_aux

#         if scaler.is_enabled():
#             scaler.scale(loss).backward()
#         else:
#             loss.backward()

#         if (step % accum_steps) == 0:
#             if scaler.is_enabled():
#                 scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
#             if scaler.is_enabled():
#                 scaler.step(optimizer)
#                 scaler.update()
#             else:
#                 optimizer.step()
#             optimizer.zero_grad(set_to_none=True)

#         running += float(loss.detach().item())
#         n_steps += 1
#     return running / max(1, n_steps)

# def run_eval(loader, model, device, *, amp: bool, desc: str,
#              switch_thr: float, use_external_switch: bool, T_av: float, dec_thr: float):
#     y, ld, la = collect_outputs(loader, model, device, amp=amp)
#     if y.size == 0:
#         print(f"{desc}: no usable samples.")
#         return None, None, None, None, None
#     if use_external_switch:
#         p_sw, conf, pick = switched_probs(ld, la, T_av=T_av, tau=switch_thr)
#         m_main = compute_metrics(y, p_sw, already_probs=True, thr=dec_thr)
#     else:
#         p_av = _sigmoid_np(ld)
#         m_main = compute_metrics(y, p_av, already_probs=True, thr=0.5)
#         conf = 2.0 * np.abs(p_av - 0.5)
#         pick = (conf < switch_thr).astype(np.float32)
#     m_av  = compute_metrics(y, ld, thr=0.5)
#     m_aux = compute_metrics(y, la, thr=0.5)
#     routing = {"total": int(y.shape[0]), "aux_count": int(pick.sum()), "av_count": int(y.shape[0] - int(pick.sum()))}
#     print(f"[{desc}] Routing: AV={routing['av_count']}/{routing['total']} ({100.0*routing['av_count']/routing['total']:.2f}%) | AUX={routing['aux_count']}/{routing['total']} ({100.0*routing['aux_count']/routing['total']:.2f}%)")
#     return m_main, m_av, m_aux, y, (ld, la)

# def save_ckpt(path, model, optimizer, epoch, cfg, best_auc: float, scaler=None):
#     payload = {
#         "model": model.state_dict(),
#         "optimizer": optimizer.state_dict(),
#         "epoch": epoch,
#         "best_auc": float(best_auc),
#         "config": cfg,
#         "calibration": {
#             "T_av": float(cfg.get("CALIB_T_AV", 1.0)),
#             "tau": float(cfg.get("CALIB_TAU", cfg.get("SWITCH_THR", 0.25))),
#             "dec_thr": float(cfg.get("CALIB_DEC_THR", 0.5)),
#         },
#     }
#     if scaler is not None:
#         payload["scaler"] = scaler.state_dict()
#     torch.save(payload, path)

# def maybe_resume(cfg, model, optimizer, scaler, device):
#     ckpt_last = os.path.join(cfg["OUT_DIR"], "ckpt_last.pt")
#     if not os.path.isfile(ckpt_last):
#         return 1, -1.0
#     payload = torch.load(ckpt_last, map_location=device)
#     try:
#         model.load_state_dict(payload["model"], strict=True)
#         optimizer.load_state_dict(payload["optimizer"])
#         if "scaler" in payload and scaler is not None:
#             try:
#                 scaler.load_state_dict(payload["scaler"])
#             except Exception:
#                 pass
#         cfg_cal = payload.get("calibration", {}) or {}
#         cfg["CALIB_T_AV"]   = float(cfg_cal.get("T_av", cfg["CALIB_INIT_T_AV"]))
#         cfg["CALIB_TAU"]    = float(cfg_cal.get("tau", cfg["SWITCH_THR"]))
#         cfg["CALIB_DEC_THR"]= float(cfg_cal.get("dec_thr", 0.5))
#         start_epoch = int(payload.get("epoch", 0)) + 1
#         best_auc = float(payload.get("best_auc", -1.0))
#         print(f"Resumed from {ckpt_last} at epoch {start_epoch-1} (best_auc={best_auc:.4f}).")
#         return start_epoch, best_auc
#     except Exception as e:
#         print(f"Resume failed: {e}. Starting fresh.")
#         return 1, -1.0

# def build_model(cfg: dict, vis20_dim: int, stft_bins: int, device: torch.device):
#     model, _criterion = build_dissonance_dual_model(
#         vis_dim=vis20_dim,
#         aud_dim=75,
#         stft_bins=stft_bins,
#         emb_dim_audio=cfg["EMB_DIM_AUDIO"],
#         emb_dim_visface=cfg["EMB_DIM_VISFACE"],
#         hidden_audio=cfg["HIDDEN_AUDIO"],
#         enc_heads=cfg["HEADS"],
#         enc_layers=cfg["LAYERS"],
#         enc_dropout=cfg["ENC_DROPOUT"],
#         pe_max_len=cfg["PE_MAX_LEN"],
#         pe_dropout=cfg["PE_DROPOUT"],
#         cls_dropout_audio=cfg["CLS_DROPOUT_AUDIO"],
#         fusion_mode=cfg["FUSION_MODE"],
#         face_pretrained=cfg["FACE_PRETRAINED"],
#         face_freeze_backbone=cfg["FACE_FREEZE_BACKBONE"],
#     )
#     model.to(device)
#     criterion = nn.BCEWithLogitsLoss()
#     return model, criterion

# def main():
#     cfg = CONFIG
#     set_seed(cfg["SEED"])
#     os.makedirs(cfg["OUT_DIR"], exist_ok=True)
#     os.makedirs(cfg["FAIL_LOG_DIR"], exist_ok=True)
#     os.makedirs(cfg["PRECOMPUTED_V75_DIR"], exist_ok=True)
#     device = torch.device(cfg["DEVICE"])
#     stft_bins = cfg["NFFT"] // 2 + 1

#     aux_labeler = AuxLabeler(cfg.get("AUX_AUDIO_CSV"), cfg.get("AUX_VISUAL_CSV"))

#     train_loader = make_loader_favc(cfg, subset="train", shuffle=True)
#     if train_loader is None:
#         raise RuntimeError("Train loader couldn't be built. Check FAVC_TRAIN_ROOT/FAVC_TRAIN_CSV.")
#     val_loader = make_loader_favc(cfg, subset="eval", shuffle=False)

#     sel_idx = getattr(train_loader.dataset, "sel_idx", None)
#     if sel_idx is None or len(sel_idx) == 0:
#         vis20_dim = 20
#         for _batch in train_loader:
#             up = _unpack_batch_safe(_batch)
#             if up is None:
#                 continue
#             xb, *_ = up
#             vis20_dim = 20 if xb.shape[1] >= 20 else int(xb.shape[1])
#             break
#     else:
#         vis20_dim = int(len(sel_idx))

#     model, criterion = build_model(cfg, vis20_dim=vis20_dim, stft_bins=stft_bins, device=device)
#     optimizer = AdamW(model.parameters(), lr=cfg["LR"], betas=cfg["BETAS"], weight_decay=cfg["WEIGHT_DECAY"])
#     scaler = torch.cuda.amp.GradScaler(enabled=(cfg["AMP"] and device.type == "cuda"))

#     cfg["CALIB_T_AV"]    = float(cfg["CALIB_INIT_T_AV"])
#     cfg["CALIB_TAU"]     = float(cfg["SWITCH_THR"])
#     cfg["CALIB_DEC_THR"] = 0.5

#     start_epoch, best_auc = maybe_resume(cfg, model, optimizer, scaler, device)
#     best_epoch = start_epoch - 1 if best_auc >= 0 else -1

#     history_path = os.path.join(cfg["OUT_DIR"], "train_history.json")
#     if os.path.isfile(history_path):
#         try:
#             with open(history_path, "r") as f:
#                 history = json.load(f)
#         except Exception:
#             history = {"train": [], "eval": [], "calib": []}
#     else:
#         history = {"train": [], "eval": [], "calib": []}

#     for epoch in range(start_epoch, cfg["EPOCHS"] + 1):
#         print(f"\nEpoch {epoch}/{cfg['EPOCHS']}")
#         train_loss = train_one_epoch(
#             train_loader, model, criterion, optimizer, device, scaler,
#             aux_labeler=aux_labeler, aux_loss_w=float(cfg.get("AUX_LOSS_W", 0.5)),
#             amp=cfg["AMP"], max_grad_norm=cfg["MAX_GRAD_NORM"], accum_steps=cfg["ACCUM_STEPS"]
#         )

#         if val_loader is not None and (cfg["CALIB_EVERY"] > 0) and (epoch % cfg["CALIB_EVERY"] == 0):
#             y_val, ld_val, la_val = collect_outputs(val_loader, model, device, amp=cfg["AMP"])
#             T_av = fit_temperature_binary(ld_val, y_val.astype(np.float32)) if y_val.size else 1.0
#             best_tau, row = sweep_tau(y_val, ld_val, la_val, T_av=T_av, metric=cfg["CALIB_METRIC"])
#             p_sw_val, _, _ = switched_probs(ld_val, la_val, T_av=T_av, tau=best_tau)
#             dec_thr = best_decision_threshold(y_val, p_sw_val, criterion="youden") if y_val.size else 0.5
#             cfg["CALIB_T_AV"] = float(T_av)
#             cfg["CALIB_TAU"]  = float(best_tau)
#             cfg["CALIB_DEC_THR"] = float(dec_thr)
#             history["calib"].append({"epoch": epoch, "T_av": float(T_av), "tau": float(best_tau), "dec_thr": float(dec_thr), **row})
#             print(f"[CALIB] epoch={epoch} | T_av={T_av:.3f} | tau={best_tau:.3f} | dec_thr={dec_thr:.3f} | {row}")

#         use_ext = (val_loader is not None)
#         tau_eval = cfg.get("CALIB_TAU", cfg["SWITCH_THR"])
#         T_av_eval = cfg.get("CALIB_T_AV", 1.0)
#         dec_thr_eval = cfg.get("CALIB_DEC_THR", 0.5)

#         m_main_t, m_av_t, m_aux_t, _, _ = run_eval(
#             train_loader, model, device, amp=cfg["AMP"], desc="eval-train",
#             switch_thr=tau_eval, use_external_switch=use_ext, T_av=T_av_eval, dec_thr=dec_thr_eval
#         )
#         if m_main_t: print(fmt_metrics("TRAIN(main)", m_main_t))
#         if m_av_t:   print(fmt_metrics("TRAIN(av_diss)", m_av_t))
#         if m_aux_t:  print(fmt_metrics("TRAIN(aux)", m_aux_t))

#         history["train"].append({
#             "epoch": epoch, "loss": train_loss,
#             **(m_main_t or {}),
#             **({f"av_{k}": v for k, v in (m_av_t or {}).items()}),
#             **({f"aux_{k}": v for k, v in (m_aux_t or {}).items()}),
#         })

#         if val_loader is not None:
#             m_main_v, m_av_v, m_aux_v, _, _ = run_eval(
#                 val_loader, model, device, amp=cfg["AMP"], desc="eval-val",
#                 switch_thr=tau_eval, use_external_switch=True, T_av=T_av_eval, dec_thr=dec_thr_eval
#             )
#             if m_main_v: print(fmt_metrics("VAL(main)", m_main_v))
#             if m_av_v:   print(fmt_metrics("VAL(av_diss)", m_av_v))
#             if m_aux_v:  print(fmt_metrics("VAL(aux)", m_aux_v))
#             history["eval"].append({
#                 "epoch": epoch,
#                 **(m_main_v or {}),
#                 **({f"av_{k}": v for k, v in (m_av_v or {}).items()}),
#                 **({f"aux_{k}": v for k, v in (m_aux_v or {}).items()}),
#             })
#             cur_auc = (m_main_v or m_av_v or m_aux_v or {"auc": 0.0})["auc"]
#         else:
#             cur_auc = (m_main_t or m_av_t or m_aux_t or {"auc": 0.0})["auc"]

#         ckpt_last = os.path.join(cfg["OUT_DIR"], "ckpt_last.pt")
#         torch.save({}, ckpt_last) if False else None
#         def _save(pth, best):
#             payload = {
#                 "model": model.state_dict(),
#                 "optimizer": optimizer.state_dict(),
#                 "epoch": epoch,
#                 "best_auc": float(best),
#                 "config": cfg,
#                 "calibration": {
#                     "T_av": float(cfg.get("CALIB_T_AV", 1.0)),
#                     "tau": float(cfg.get("CALIB_TAU", cfg.get("SWITCH_THR", 0.25))),
#                     "dec_thr": float(cfg.get("CALIB_DEC_THR", 0.5)),
#                 },
#                 "scaler": scaler.state_dict() if scaler is not None else None,
#             }
#             torch.save(payload, pth)

#         _save(ckpt_last, max(best_auc, cur_auc))
#         if cfg["SAVE_EVERY_EPOCH"]:
#             _save(os.path.join(cfg["OUT_DIR"], f"ckpt_epoch_{epoch:03d}.pt"), max(best_auc, cur_auc))

#         if cur_auc > best_auc:
#             best_auc = cur_auc
#             best_epoch = epoch
#             _save(os.path.join(cfg["OUT_DIR"], "ckpt_best_auc.pt"), best_auc)
#             print(f"✓ New best AUC={best_auc:.4f} at epoch {best_epoch}. Saved to runs/favc_dual_avdiss_visual75_train/ckpt_best_auc.pt")

#         with open(history_path, "w") as f:
#             json.dump(history, f, indent=2)

#     print(f"\nTraining done. Best AUC={best_auc:.4f} (epoch {best_epoch}).")
#     print(f"Artifacts in: {cfg['OUT_DIR']}")
#     print("Files: ckpt_last.pt, ckpt_best_auc.pt, train_history.json")

# if __name__ == "__main__":
#     main()