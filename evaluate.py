# import os
# import json
# import warnings
# from typing import Dict, Optional, Tuple

# import numpy as np
# import torch
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

# # =========================
# # Config (edit paths below)
# # =========================
# CONFIG = {
#     # ==== Shared loader/config (match your train settings as needed) ====
#     "BATCH_SIZE": 4,
#     "NUM_WORKERS": 2,
#     "FRAMES_PER_CLIP": 25,
#     "STRIDE": 1,
#     "AUDIO_SR": 16000,
#     "NFFT": 400,
#     "HOP": 160,
#     "WIN": 400,
#     "FACE_SIZE": 224,
#     "SHOW_TQDM": True,
#     "FACE_DET_WEIGHTS": "yolov8n-face.pt",
#     "OPENFACE_BINARY": "../OpenFace/build/bin/FeatureExtraction",
#     "PRECOMPUTED_V75_DIR": None,     # e.g. "runs/v75_cache" if you cached VIS-75
#     "COMPUTE_IF_MISSING": True,      # compute VIS-75 if .npy missing (needs OpenFace CSV)
#     "FAIL_LOG_DIR": "runs/fail_logs_test",

#     # VIS-20 selection to slice VIS-75 inside the dataloader
#     "VIS20_JSON": "runs/visual75_crossdomain_selection.json",
#     "VIS20_KEY": None,
#     "ENFORCE_EXACT_20": True,

#     # ==== Model (must match training) ====
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

#     # ==== Runtime ====
#     "AMP": True,
#     "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",

#     # ==== Checkpoint dir ====
#     "CKPT_DIR": "runs/avdf1m_dual_avdiss_visual75_vf",
#     "CKPT_BEST": "ckpt_best_auc.pt",
#     "CKPT_LAST": "ckpt_last.pt",

#     # ===== Datasets =====
#     # FakeAVCeleb
#     "FAVC_ROOT": "/media/rt0706/Media/VCBSL-Dataset/FakeAVCeleb",
#     "FAVC_CSV": "../Dataset/favc_multimodal_data.csv",  # contains split="test"
#     "FAVC_SUBSET": "test",

#     # LAV-DF
#     "LAVDF_ROOT": "/media/rt0706/Media/VCBSL-Dataset/LAV-DF",
#     "LAVDF_JSON": "../Dataset/LAV-DF/metadata.json",
#     "LAVDF_SUBSET": "test",

#     # AV-Deepfake1M (evaluation split)
#     "AVDF1M_ROOT": "/media/rt0706/Lab/AV-Deepfake1M_test_10pct/zips/val",
#     "AVDF1M_JSON": "/media/rt0706/Lab/AV-Deepfake1M_test_10pct/zips/val_metadata.json",
#     "AVDF1M_SUBSET": "val",
# }

# # =========================
# # Utils
# # =========================
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
#         n_real = int((y_true == 0).sum())
#         n_fake = int((y_true == 1).sum())
#         line += f" | real: {m['correct_real']}/{n_real} | fake: {m['correct_fake']}/{n_fake}"
#     return line

# @torch.no_grad()
# def collect_outputs(loader, model, device, amp=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     dev_type = device.type
#     amp_on = (amp and dev_type == "cuda")
#     y_all, ld_all, la_all = [], [], []
#     model.eval()
#     for batch in tqdm(loader, total=len(loader), desc="collect", ncols=100, leave=False):
#         # Dataloader returns:
#         # (x_vis20, x_vis75, face, stft, x_aud, y_mm, y_a|None, y_v|None, a_len, path)
#         x_vis20, x_vis75, face, stft, x_aud, y_mm, _y_a, _y_v, _a_len, _paths = batch
#         x_vis20 = x_vis20.to(device, non_blocking=True)
#         x_aud   = x_aud.to(device, non_blocking=True)
#         stft    = stft.to(device, non_blocking=True)
#         face    = face.to(device, non_blocking=True)
#         x_vis75 = x_vis75.to(device, non_blocking=True) if x_vis75 is not None else x_vis20
#         with torch.autocast(device_type=dev_type, dtype=torch.float16, enabled=amp_on):
#             out = model(x_vis20=x_vis20, x_vis75=x_vis75, x_aud=x_aud, stft=stft, face=face, infer_switch=False)
#         y_all.append(y_mm.detach().float().cpu().numpy())
#         ld_all.append(out["diss_logits"].detach().float().cpu().numpy())
#         la_all.append(out["aux_logits"].detach().float().cpu().numpy())
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

# # =========================
# # Dataloaders (test subsets)
# # =========================
# def make_loader_favc(cfg: dict):
#     return get_unified_av_dataloader(
#         mode="fakeavceleb",
#         subset=cfg["FAVC_SUBSET"],
#         root_dir=cfg["FAVC_ROOT"],
#         csv_path=cfg["FAVC_CSV"],
#         frames_per_clip=cfg["FRAMES_PER_CLIP"],
#         stride=cfg["STRIDE"],
#         # do NOT rebalance at test time
#         balance_minority=False,
#         use_fake_periods=False,
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
#         shuffle=False,
#         show_tqdm=cfg["SHOW_TQDM"],
#         face_detector_weights=cfg["FACE_DET_WEIGHTS"],
#         visual_feat_mode="vis75",
#         openface_binary=cfg.get("OPENFACE_BINARY", ""),
#         precomputed_dir=cfg.get("PRECOMPUTED_V75_DIR", None),
#         compute_if_missing=cfg.get("COMPUTE_IF_MISSING", False),
#         fail_log_dir=cfg.get("FAIL_LOG_DIR", None),
#     )

# def make_loader_lavdf(cfg: dict):
#     return get_unified_av_dataloader(
#         mode="lavdf",
#         subset=cfg["LAVDF_SUBSET"],
#         root_dir=cfg["LAVDF_ROOT"],
#         json_path=cfg["LAVDF_JSON"],
#         frames_per_clip=cfg["FRAMES_PER_CLIP"],
#         stride=cfg["STRIDE"],
#         balance_minority=False,
#         use_fake_periods=True, 
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
#         shuffle=False,
#         show_tqdm=cfg["SHOW_TQDM"],
#         face_detector_weights=cfg["FACE_DET_WEIGHTS"],
#         visual_feat_mode="vis75",
#         openface_binary=cfg.get("OPENFACE_BINARY", ""),
#         precomputed_dir=cfg.get("PRECOMPUTED_V75_DIR", None),
#         compute_if_missing=cfg.get("COMPUTE_IF_MISSING", False),
#         fail_log_dir=cfg.get("FAIL_LOG_DIR", None),
#     )

# def make_loader_avdf1m(cfg: dict):
#     return get_unified_av_dataloader(
#         mode="av_deepfake1m",
#         subset=cfg["AVDF1M_SUBSET"],
#         root_dir=cfg["AVDF1M_ROOT"],
#         json_path=cfg["AVDF1M_JSON"],
#         frames_per_clip=cfg["FRAMES_PER_CLIP"],
#         stride=cfg["STRIDE"],
#         balance_minority=False,
#         use_fake_periods=True,
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
#         shuffle=False,
#         show_tqdm=cfg["SHOW_TQDM"],
#         face_detector_weights=cfg["FACE_DET_WEIGHTS"],
#         visual_feat_mode="vis75",
#         openface_binary=cfg.get("OPENFACE_BINARY", ""),
#         precomputed_dir=cfg.get("PRECOMPUTED_V75_DIR", None),
#         compute_if_missing=cfg.get("COMPUTE_IF_MISSING", False),
#         fail_log_dir=cfg.get("FAIL_LOG_DIR", None),
#     )

# # =========================
# # Model / ckpt helpers
# # =========================
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
#     return model

# def load_checkpoint(cfg: dict, model: torch.nn.Module, device: torch.device):
#     ckpt_best = os.path.join(cfg["CKPT_DIR"], cfg["CKPT_BEST"])
#     ckpt_last = os.path.join(cfg["CKPT_DIR"], cfg["CKPT_LAST"])
#     path = ckpt_best if os.path.isfile(ckpt_best) else ckpt_last
#     if not os.path.isfile(path):
#         raise FileNotFoundError(f"No checkpoint found at {ckpt_best} or {ckpt_last}")
#     payload = torch.load(path, map_location=device)
#     model.load_state_dict(payload["model"], strict=True)
#     calib = payload.get("calibration", {}) or {}
#     T_av = float(calib.get("T_av", 1.0))
#     tau  = float(calib.get("tau", 0.25))
#     dec_thr = float(calib.get("dec_thr", 0.5))
#     print(f"Loaded checkpoint: {path}")
#     print(f"Calibration → T_av={T_av:.3f}, tau={tau:.3f}, dec_thr={dec_thr:.3f}")
#     return T_av, tau, dec_thr

# # =========================
# # Main test flow
# # =========================
# def evaluate_one(loader, model, device, *, T_av: float, tau: float, dec_thr: float, amp: bool, tag: str):
#     if loader is None:
#         print(f"{tag}: loader is None (skipping).")
#         return None

#     # Determine VIS-20 dim (align with loader's selection)
#     sel_idx = getattr(loader.dataset, "sel_idx", None)
#     if sel_idx is None or len(sel_idx) == 0:
#         # Peek a batch to infer
#         xb, *_ = next(iter(loader))
#         vis20_dim = 20 if xb.shape[1] >= 20 else int(xb.shape[1])
#     else:
#         vis20_dim = int(len(sel_idx))

#     # Ensure model has matching input dim
#     # (Assumes you loaded a matching checkpoint; skip rebuilding for simplicity)

#     # Collect logits
#     y, ld, la = collect_outputs(loader, model, device, amp=amp)
#     if y.size == 0:
#         print(f"{tag}: no usable samples (all filtered).")
#         return None

#     # External switch inference (calibrated)
#     p_sw, _, _ = switched_probs(ld, la, T_av=T_av, tau=tau)
#     m_main = compute_metrics(y.astype(np.int64), p_sw.astype(np.float32), already_probs=True, thr=dec_thr)
#     print(fmt_metrics(f"{tag} (SWITCHED)", m_main, y_true=y))
#     # Also report the two heads individually (threshold 0.5)
#     m_av  = compute_metrics(y.astype(np.int64), ld.astype(np.float32), thr=0.5)
#     m_aux = compute_metrics(y.astype(np.int64), la.astype(np.float32), thr=0.5)
#     print(fmt_metrics(f"{tag} (AV-Dissonance)", m_av,  y_true=y))
#     print(fmt_metrics(f"{tag} (AUX)",          m_aux, y_true=y))
#     return {"main": m_main, "av": m_av, "aux": m_aux}

# def main():
#     cfg = CONFIG
#     os.makedirs(cfg["FAIL_LOG_DIR"], exist_ok=True)

#     device = torch.device(cfg["DEVICE"])
#     stft_bins = cfg["NFFT"] // 2 + 1

#     # Build loaders
#     favc_loader  = make_loader_favc(cfg)
#     lavdf_loader = make_loader_lavdf(cfg)
#     avdf1m_loader = make_loader_avdf1m(cfg)

#     # Get VIS-20 dim based on any loader (all use same selection JSON)
#     probe_loader = avdf1m_loader or lavdf_loader or favc_loader
#     if probe_loader is None:
#         raise RuntimeError("No dataset loaders could be constructed. Check your paths.")

#     sel_idx = getattr(probe_loader.dataset, "sel_idx", None)
#     if sel_idx is None or len(sel_idx) == 0:
#         xb, *_ = next(iter(probe_loader))
#         vis20_dim = 20 if xb.shape[1] >= 20 else int(xb.shape[1])
#     else:
#         vis20_dim = int(len(sel_idx))

#     # Build model & load checkpoint
#     model = build_model(cfg, vis20_dim=vis20_dim, stft_bins=stft_bins, device=device)
#     T_av, tau, dec_thr = load_checkpoint(cfg, model, device)

#     # Eval each dataset
#     print("\n========== Inference ==========")
#     if favc_loader is not None:
#         evaluate_one(favc_loader, model, device, T_av=T_av, tau=tau, dec_thr=dec_thr, amp=cfg["AMP"], tag="FAVC")
#     if lavdf_loader is not None:
#         evaluate_one(lavdf_loader, model, device, T_av=T_av, tau=tau, dec_thr=dec_thr, amp=cfg["AMP"], tag="LAV-DF")
#     if avdf1m_loader is not None:
#         evaluate_one(avdf1m_loader, model, device, T_av=T_av, tau=tau, dec_thr=dec_thr, amp=cfg["AMP"], tag="AV-Deepfake1M")

# if __name__ == "__main__":
#     main()

# # ------------------ USE THIS FOR NORMAL--------------## 

# import os
# import warnings
# from typing import Dict, Optional, Tuple

# import numpy as np
# import torch
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
#     "BATCH_SIZE": 8,
#     "NUM_WORKERS": 2,
#     "FRAMES_PER_CLIP": 25,
#     "STRIDE": 1,
#     "AUDIO_SR": 16000,
#     "NFFT": 400,
#     "HOP": 160,
#     "WIN": 400,
#     "FACE_SIZE": 224,
#     "SHOW_TQDM": True,
#     "FACE_DET_WEIGHTS": "yolov8n-face.pt",
#     "OPENFACE_BINARY": "../OpenFace/build/bin/FeatureExtraction",
#     "PRECOMPUTED_V75_DIR": None,     # e.g. "runs/v75_cache" if you cached VIS-75
#     "COMPUTE_IF_MISSING": True,      # compute VIS-75 if .npy missing (needs OpenFace CSV)
#     "FAIL_LOG_DIR": "runs/fail_logs_test",

#     # VIS-20 selection to slice VIS-75 inside the dataloader
#     "VIS20_JSON": "runs/visual75_crossdomain_selection.json",
#     "VIS20_KEY": None,
#     "ENFORCE_EXACT_20": True,

#     # ==== Model (must match training) ====
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

#     # ==== Runtime ====
#     "AMP": True,
#     "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",

#     # ==== Checkpoint dir ====
#     "CKPT_DIR": "runs/avdf1m_dual_avdiss_visual75_vf",
#     "CKPT_BEST": "ckpt_best_auc.pt",
#     "CKPT_LAST": "ckpt_last.pt",

#     # ===== Datasets =====
#     # FakeAVCeleb
#     "FAVC_ROOT": "/media/rt0706/Media/VCBSL-Dataset/FakeAVCeleb",
#     "FAVC_CSV": "../Dataset/favc_multimodal_data.csv",  # contains split="test"
#     "FAVC_SUBSET": "test",

#     # LAV-DF
#     "LAVDF_ROOT": "/media/rt0706/Media/VCBSL-Dataset/LAV-DF",
#     "LAVDF_JSON": "../Dataset/LAV-DF/metadata.json",
#     "LAVDF_SUBSET": "test",

#     # AV-Deepfake1M (evaluation split)
#     "AVDF1M_ROOT": "/media/rt0706/Media/VCBSL-Dataset/AV-Deepfake1M/val",
#     "AVDF1M_JSON": "/media/rt0706/Media/VCBSL-Dataset/AV-Deepfake1M/val_metadata.json",
#     "AVDF1M_SUBSET": "val",
# }


# PER_DATASET_PARAMS = {
#     # From repeated stratified 20% (median)
#     "FAVC":   {"T_av": 5.020,   "tau": 0.590, "dec_thr": 0.069},
#     # From two repeats (midpoint/median-of-two)
#     "LAV-DF": {"T_av": 2.3205,  "tau": 0.010, "dec_thr": 0.139},
#     # AVDF1M uses checkpoint calibration
# }

# # =========================
# # Utils
# # =========================
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
#         n_real = int((y_true == 0).sum())
#         n_fake = int((y_true == 1).sum())
#         line += f" | real: {m['correct_real']}/{n_real} | fake: {m['correct_fake']}/{n_fake}"
#     return line

# @torch.no_grad()
# def collect_outputs(loader, model, device, amp=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     dev_type = device.type
#     amp_on = (amp and dev_type == "cuda")
#     y_all, ld_all, la_all = [], [], []
#     model.eval()
#     for batch in tqdm(loader, total=len(loader), desc="collect", ncols=100, leave=False):
#         # (x_vis20, x_vis75, face, stft, x_aud, y_mm, y_a|None, y_v|None, a_len, path)
#         x_vis20, x_vis75, face, stft, x_aud, y_mm, _y_a, _y_v, _a_len, _paths = batch
#         x_vis20 = x_vis20.to(device, non_blocking=True)
#         x_aud   = x_aud.to(device, non_blocking=True)
#         stft    = stft.to(device, non_blocking=True)
#         face    = face.to(device, non_blocking=True)
#         x_vis75 = x_vis75.to(device, non_blocking=True) if x_vis75 is not None else x_vis20
#         with torch.autocast(device_type=dev_type, dtype=torch.float16, enabled=amp_on):
#             out = model(x_vis20=x_vis20, x_vis75=x_vis75, x_aud=x_aud, stft=stft, face=face, infer_switch=False)
#         y_all.append(y_mm.detach().float().cpu().numpy())
#         ld_all.append(out["diss_logits"].detach().float().cpu().numpy())
#         la_all.append(out["aux_logits"].detach().float().cpu().numpy())
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


# def make_loader_favc(cfg: dict):
#     return get_unified_av_dataloader(
#         mode="fakeavceleb",
#         subset=cfg["FAVC_SUBSET"],
#         root_dir=cfg["FAVC_ROOT"],
#         csv_path=cfg["FAVC_CSV"],
#         frames_per_clip=cfg["FRAMES_PER_CLIP"],
#         stride=cfg["STRIDE"],
#         balance_minority=False,
#         use_fake_periods=False,
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
#         shuffle=False,
#         show_tqdm=cfg["SHOW_TQDM"],
#         face_detector_weights=cfg["FACE_DET_WEIGHTS"],
#         visual_feat_mode="vis75",
#         openface_binary=cfg.get("OPENFACE_BINARY", ""),
#         precomputed_dir=cfg.get("PRECOMPUTED_V75_DIR", None),
#         compute_if_missing=cfg.get("COMPUTE_IF_MISSING", False),
#         fail_log_dir=cfg.get("FAIL_LOG_DIR", None),
#     )

# def make_loader_lavdf(cfg: dict):
#     return get_unified_av_dataloader(
#         mode="lavdf",
#         subset=cfg["LAVDF_SUBSET"],
#         root_dir=cfg["LAVDF_ROOT"],
#         json_path=cfg["LAVDF_JSON"],
#         frames_per_clip=cfg["FRAMES_PER_CLIP"],
#         stride=cfg["STRIDE"],
#         balance_minority=False,
#         use_fake_periods=True,
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
#         shuffle=False,
#         show_tqdm=cfg["SHOW_TQDM"],
#         face_detector_weights=cfg["FACE_DET_WEIGHTS"],
#         visual_feat_mode="vis75",
#         openface_binary=cfg.get("OPENFACE_BINARY", ""),
#         precomputed_dir=cfg.get("PRECOMPUTED_V75_DIR", None),
#         compute_if_missing=cfg.get("COMPUTE_IF_MISSING", False),
#         fail_log_dir=cfg.get("FAIL_LOG_DIR", None),
#     )

# def make_loader_avdf1m(cfg: dict):
#     return get_unified_av_dataloader(
#         mode="av_deepfake1m",
#         subset=cfg["AVDF1M_SUBSET"],
#         root_dir=cfg["AVDF1M_ROOT"],
#         json_path=cfg["AVDF1M_JSON"],
#         frames_per_clip=cfg["FRAMES_PER_CLIP"],
#         stride=cfg["STRIDE"],
#         balance_minority=False,
#         use_fake_periods=True,
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
#         shuffle=False,
#         show_tqdm=cfg["SHOW_TQDM"],
#         face_detector_weights=cfg["FACE_DET_WEIGHTS"],
#         visual_feat_mode="vis75",
#         openface_binary=cfg.get("OPENFACE_BINARY", ""),
#         precomputed_dir=cfg.get("PRECOMPUTED_V75_DIR", None),
#         compute_if_missing=cfg.get("COMPUTE_IF_MISSING", False),
#         fail_log_dir=cfg.get("FAIL_LOG_DIR", None),
#     )

# # =========================
# # Model / ckpt helpers
# # =========================
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
#     return model

# def load_checkpoint(cfg: dict, model: torch.nn.Module, device: torch.device):
#     ckpt_best = os.path.join(cfg["CKPT_DIR"], cfg["CKPT_BEST"])
#     ckpt_last = os.path.join(cfg["CKPT_DIR"], cfg["CKPT_LAST"])
#     path = ckpt_best if os.path.isfile(ckpt_best) else ckpt_last
#     if not os.path.isfile(path):
#         raise FileNotFoundError(f"No checkpoint found at {ckpt_best} or {ckpt_last}")
#     payload = torch.load(path, map_location=device)
#     model.load_state_dict(payload["model"], strict=True)
#     calib = payload.get("calibration", {}) or {}
#     T_av = float(calib.get("T_av", 1.0))
#     tau  = float(calib.get("tau", 0.25))
#     dec_thr = float(calib.get("dec_thr", 0.5))
#     print(f"Loaded checkpoint: {path}")
#     print(f"Calibration → T_av={T_av:.3f}, tau={tau:.3f}, dec_thr={dec_thr:.3f}")
#     return T_av, tau, dec_thr

# # =========================
# # Main test flow
# # =========================
# def evaluate_one(loader, model, device, *, T_av: float, tau: float, dec_thr: float, amp: bool, tag: str):
#     if loader is None:
#         print(f"{tag}: loader is None (skipping).")
#         return None

#     y, ld, la = collect_outputs(loader, model, device, amp=amp)
#     if y.size == 0:
#         print(f"{tag}: no usable samples (all filtered).")
#         return None

#     p_sw, _, _ = switched_probs(ld, la, T_av=T_av, tau=tau)
#     m_main = compute_metrics(y.astype(np.int64), p_sw.astype(np.float32), already_probs=True, thr=dec_thr)
#     print(f"Using params → T_av={T_av:.3f}, tau={tau:.3f}, dec_thr={dec_thr:.3f}")
#     print(fmt_metrics(f"{tag} (SWITCHED)", m_main, y_true=y))
#     m_av  = compute_metrics(y.astype(np.int64), ld.astype(np.float32), thr=0.5)
#     m_aux = compute_metrics(y.astype(np.int64), la.astype(np.float32), thr=0.5)
#     print(fmt_metrics(f"{tag} (AV-Dissonance)", m_av,  y_true=y))
#     print(fmt_metrics(f"{tag} (AUX)",          m_aux, y_true=y))
#     return {"main": m_main, "av": m_av, "aux": m_aux}

# def main():
#     cfg = CONFIG
#     os.makedirs(cfg["FAIL_LOG_DIR"], exist_ok=True)

#     device = torch.device(cfg["DEVICE"])
#     stft_bins = cfg["NFFT"] // 2 + 1

#     # Build loaders
#     favc_loader   = make_loader_favc(cfg)
#     lavdf_loader  = make_loader_lavdf(cfg)
#     avdf1m_loader = make_loader_avdf1m(cfg)

#     # VIS-20 dim from any available loader (they share selection JSON)
#     probe_loader = avdf1m_loader or lavdf_loader or favc_loader
#     if probe_loader is None:
#         raise RuntimeError("No dataset loaders could be constructed. Check your paths.")
#     sel_idx = getattr(probe_loader.dataset, "sel_idx", None)
#     if sel_idx is None or len(sel_idx) == 0:
#         xb, *_ = next(iter(probe_loader))
#         vis20_dim = 20 if xb.shape[1] >= 20 else int(xb.shape[1])
#     else:
#         vis20_dim = int(len(sel_idx))

#     # Build model & load checkpoint (AVDF1M calibration stays as-is)
#     model = build_model(cfg, vis20_dim=vis20_dim, stft_bins=stft_bins, device=device)
#     T_av_ckpt, tau_ckpt, dec_thr_ckpt = load_checkpoint(cfg, model, device)

#     # Use hardcoded best params for FAVC & LAV-DF; checkpoint params for AVDF1M
#     favc_params   = PER_DATASET_PARAMS["FAVC"]
#     lavdf_params  = PER_DATASET_PARAMS["LAV-DF"]
#     avdf1m_params = {"T_av": T_av_ckpt, "tau": tau_ckpt, "dec_thr": dec_thr_ckpt}

#     print("\n========== Inference ==========")
#     if favc_loader is not None:
#         evaluate_one(favc_loader, model, device,
#                      T_av=favc_params["T_av"], tau=favc_params["tau"], dec_thr=favc_params["dec_thr"],
#                      amp=cfg["AMP"], tag="FAVC")
#     if lavdf_loader is not None:
#         evaluate_one(lavdf_loader, model, device,
#                      T_av=lavdf_params["T_av"], tau=lavdf_params["tau"], dec_thr=lavdf_params["dec_thr"],
#                      amp=cfg["AMP"], tag="LAV-DF")
#     if avdf1m_loader is not None:
#         evaluate_one(avdf1m_loader, model, device,
#                      T_av=avdf1m_params["T_av"], tau=avdf1m_params["tau"], dec_thr=avdf1m_params["dec_thr"],
#                      amp=cfg["AMP"], tag="AV-Deepfake1M")

# if __name__ == "__main__":
#     main()

##---------------------ROUTING SUMMARY------------------##

import os
import json
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from tqdm import tqdm

# Quieter logs
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("ULTRALYTICS_VERBOSITY", "0")
os.environ.setdefault("WANDB_SILENT", "true")
warnings.filterwarnings("ignore")

# ==== mirror training imports ====
from av_dissonance_with_aux_ensemble_model import build_dissonance_dual_model
from unified_av_dataloader import get_unified_av_dataloader

# =========================
# Config (edit paths below)
# =========================
CONFIG = {
    # ==== Shared loader/config ====
    "BATCH_SIZE": 4,
    "NUM_WORKERS": 2,
    "FRAMES_PER_CLIP": 25,
    "STRIDE": 1,
    "AUDIO_SR": 16000,
    "NFFT": 400,
    "HOP": 160,
    "WIN": 400,
    "FACE_SIZE": 224,
    "SHOW_TQDM": True,
    "FACE_DET_WEIGHTS": "yolov8n-face.pt",
    "OPENFACE_BINARY": "../OpenFace/build/bin/FeatureExtraction",
    "PRECOMPUTED_V75_DIR": None,
    "COMPUTE_IF_MISSING": True,
    "FAIL_LOG_DIR": "runs/fail_logs_test",

    "VIS20_JSON": "runs/visual75_crossdomain_selection.json",
    "VIS20_KEY": None,
    "ENFORCE_EXACT_20": True,

    # ==== Model (must match training) ====
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

    # ==== Runtime ====
    "AMP": True,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",

    # ==== Checkpoint dir ====
    "CKPT_DIR": "runs/avdf1m_dual_avdiss_visual75_vf",
    "CKPT_BEST": "ckpt_best_auc.pt",
    "CKPT_LAST": "ckpt_last.pt",

    # ===== Datasets =====
    "FAVC_ROOT": "/media/rt0706/Media/VCBSL-Dataset/FakeAVCeleb",
    "FAVC_CSV": "../Dataset/favc_multimodal_data.csv",
    "FAVC_SUBSET": "test",

    "LAVDF_ROOT": "/media/rt0706/Media/VCBSL-Dataset/LAV-DF",
    "LAVDF_JSON": "../Dataset/LAV-DF/metadata.json",
    "LAVDF_SUBSET": "test",

    "AVDF1M_ROOT": "/media/rt0706/Lab/AV-Deepfake1M_test_10pct/zips/val",
    "AVDF1M_JSON": "/media/rt0706/Lab/AV-Deepfake1M_test_10pct/zips/val_metadata.json",
    "AVDF1M_SUBSET": "val",

    # -------- Per-dataset fractional subsampling (via Subset wrapper) --------
    "SUBSAMPLE": {
        "FAVC":   {"fraction": 1.00, "balanced": False},
        "LAV-DF": {"fraction": 0.20, "balanced": False},
        "AVDF1M": {"fraction": 0.20, "balanced": False},
        "SEED": 1337,
    },
}

# =========================
# Utils
# =========================
def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def _eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2.0)

def compute_metrics(y_true: np.ndarray, logits_or_probs: np.ndarray,
                    already_probs: bool = False, thr: Optional[float] = 0.5) -> Dict[str, float]:
    probs = logits_or_probs.astype(np.float32) if already_probs else _sigmoid_np(logits_or_probs.astype(np.float32))
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

def fmt_metrics(tag: str, m: Dict[str, float], y_true: Optional[np.ndarray] = None) -> str:
    line = (f"{tag} | AUC={m['auc']:.4f} AP={m['ap']:.4f} EER={m['eer']:.4f} "
            f"| acc@{m.get('thr', 0.5):.3f}={m['acc']:.4f} ({m['correct']}/{m['total']}) "
            f"| TN={m['correct_real']} TP={m['correct_fake']} FP={m['fp']} FN={m['fn']}")
    if y_true is not None and y_true.size:
        n_real = int((y_true == 0).sum()); n_fake = int((y_true == 1).sum())
        line += f" | real: {m['correct_real']}/{n_real} | fake: {m['correct_fake']}/{n_fake}"
    return line

# ---- tolerant batch unpacker ----
def _unpack_batch_safe(batch):
    """
    Expected dataloader return (training convention):
      x_vis20, x_vis75, face, stft, x_aud, y_mm, y_a|None, y_v|None, a_len, paths

    Returns a 10-tuple or None if the batch is unusable.
    """
    if batch is None:
        return None
    if not isinstance(batch, (list, tuple)):
        return None
    L = len(batch)
    if L == 10:
        return batch
    elif L in (8, 9):
        # Older/leaner collate: (x_vis20, x_vis75, face, stft, x_aud, y_mm, a_len, paths)
        try:
            x_vis20, x_vis75, face, stft, x_aud, y_mm, a_len, paths = batch[:8]
            return x_vis20, x_vis75, face, stft, x_aud, y_mm, None, None, a_len, paths
        except Exception:
            return None
    else:
        return None

# ---- AUX adapter: single aux_logits or a/v heads + gate ----
def _get_aux_logits_from_out(out: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "aux_logits" in out:
        return out["aux_logits"]
    # Fall back: choose audio if gate>=0.5 else visual
    if ("aux_a_logits" in out) and ("aux_v_logits" in out):
        z_a = out["aux_a_logits"]
        z_v = out["aux_v_logits"]
        g   = out.get("aux_gate", None)
        if g is not None:
            # Align shapes by broadcasting over trailing dims
            while g.ndim < z_a.ndim:
                g = g.unsqueeze(-1)
            return (g >= 0.5).float() * z_a + (g < 0.5).float() * z_v
        # Fallback: elementwise max
        return torch.where(z_a >= z_v, z_a, z_v)
    raise KeyError("No AUX logits found: expected 'aux_logits' or ('aux_a_logits','aux_v_logits'[,'aux_gate']).")

@torch.no_grad()
def collect_outputs(loader, model, device, amp=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dev_type = device.type
    amp_on = (amp and dev_type == "cuda")

    y_all, ld_all, la_all = [], [], []
    model.eval()

    for batch in tqdm(loader, total=len(loader), desc="collect", ncols=100, leave=False):
        unpacked = _unpack_batch_safe(batch)
        if unpacked is None:
            # Skip unusable batch (e.g., failed face/audio extraction or None collate)
            continue

        x_vis20, x_vis75, face, stft, x_aud, y_mm, _y_a, _y_v, _a_len, _paths = unpacked

        x_vis20 = x_vis20.to(device, non_blocking=True)
        x_aud   = x_aud.to(device, non_blocking=True)
        stft    = stft.to(device, non_blocking=True)
        face    = face.to(device, non_blocking=True)
        x_vis75 = x_vis75.to(device, non_blocking=True) if x_vis75 is not None else x_vis20

        with torch.autocast(device_type=dev_type, dtype=torch.float16, enabled=amp_on):
            out = model(
                x_vis20=x_vis20, x_vis75=x_vis75,
                x_aud=x_aud, stft=stft, face=face,
                infer_switch=False
            )

        y_all.append(y_mm.detach().float().cpu().numpy())
        ld_all.append(out["diss_logits"].detach().float().cpu().numpy())
        la_all.append(_get_aux_logits_from_out(out).detach().float().cpu().numpy())

    y  = np.concatenate(y_all,  axis=0).astype(np.int64)  if y_all  else np.zeros((0,), dtype=np.int64)
    ld = np.concatenate(ld_all, axis=0).astype(np.float32) if ld_all else np.zeros_like(y, dtype=np.float32)
    la = np.concatenate(la_all, axis=0).astype(np.float32) if la_all else np.zeros_like(y, dtype=np.float32)
    return y, ld, la

def switched_probs(ld: np.ndarray, la: np.ndarray, T_av=1.0, tau=0.25):
    pav = _sigmoid_np(ld / max(T_av, 1e-6))
    pax = _sigmoid_np(la)
    conf = 2.0 * np.abs(pav - 0.5)
    pick_aux = (conf < tau).astype(np.float32)
    pfinal = (1.0 - pick_aux) * pav + pick_aux * pax
    return pfinal, conf, pick_aux

# =========================
# Subsampling wrapper
# =========================
def subsample_loader(loader, fraction: float, seed: int = 1337, balanced: bool = False):
    """
    Wrap an existing DataLoader by replacing its dataset with a Subset of indices.
    - fraction in (0,1]; if >=1.0, returns the original loader.
    - balanced=True tries to split evenly across labels if dataset exposes labels; otherwise falls back to unbalanced.
    """
    from torch.utils.data import DataLoader, Subset

    if fraction >= 1.0:
        return loader

    ds = loader.dataset
    n = len(ds)
    k = max(1, int(np.floor(n * fraction)))
    rng = np.random.RandomState(seed)

    # Try to read labels from common attribute names; otherwise unbalanced
    labels = None
    for attr in ("labels", "targets", "y", "y_all", "label_list", "ys"):
        if hasattr(ds, attr):
            try:
                arr = getattr(ds, attr)
                labels = np.asarray(arr)
                break
            except Exception:
                pass

    if balanced and (labels is not None) and (set(np.unique(labels)) >= {0, 1}):
        idx0 = np.where(labels == 0)[0]
        idx1 = np.where(labels == 1)[0]
        if len(idx0) == 0 or len(idx1) == 0:
            sel = rng.choice(n, k, replace=False)
            print("[subsample] Balanced requested but one class missing; using unbalanced sampling.")
        else:
            take0 = min(len(idx0), k // 2)
            take1 = min(len(idx1), k - take0)
            sel = np.concatenate([
                rng.choice(idx0, take0, replace=False),
                rng.choice(idx1, take1, replace=False)
            ])
    else:
        sel = rng.choice(n, k, replace=False)
        if balanced and labels is None:
            print("[subsample] Balanced requested but labels not found; using unbalanced sampling.")

    sel = np.sort(sel)
    # Preserve collate_fn/pin_memory if present
    collate_fn = getattr(loader, "collate_fn", None)
    new_loader = DataLoader(
        Subset(ds, sel.tolist()),
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=loader.num_workers,
        pin_memory=getattr(loader, "pin_memory", False),
        collate_fn=collate_fn,
        drop_last=False,
        persistent_workers=getattr(loader, "persistent_workers", False),
    )
    print(f"[subsample] Using {k}/{n} samples ({fraction*100:.2f}%).")
    return new_loader

# =========================
# Dataloaders (build, then optionally subsample)
# =========================
def make_loader_favc(cfg: dict):
    loader = get_unified_av_dataloader(
        mode="fakeavceleb",
        subset=cfg["FAVC_SUBSET"],
        root_dir=cfg["FAVC_ROOT"],
        csv_path=cfg["FAVC_CSV"],
        frames_per_clip=cfg["FRAMES_PER_CLIP"],
        stride=cfg["STRIDE"],
        balance_minority=False,
        use_fake_periods=False,
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
        shuffle=False,
        show_tqdm=cfg["SHOW_TQDM"],
        face_detector_weights=cfg["FACE_DET_WEIGHTS"],
        visual_feat_mode="vis75",
        openface_binary=cfg.get("OPENFACE_BINARY", ""),
        precomputed_dir=cfg.get("PRECOMPUTED_V75_DIR", None),
        compute_if_missing=cfg.get("COMPUTE_IF_MISSING", False),
        fail_log_dir=cfg.get("FAIL_LOG_DIR", None),
    )
    pol = cfg["SUBSAMPLE"]["FAVC"]
    return subsample_loader(loader, pol["fraction"], cfg["SUBSAMPLE"]["SEED"], pol.get("balanced", False))

def make_loader_lavdf(cfg: dict):
    loader = get_unified_av_dataloader(
        mode="lavdf",
        subset=cfg["LAVDF_SUBSET"],
        root_dir=cfg["LAVDF_ROOT"],
        json_path=cfg["LAVDF_JSON"],
        frames_per_clip=cfg["FRAMES_PER_CLIP"],
        stride=cfg["STRIDE"],
        balance_minority=False,
        use_fake_periods=True,
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
        shuffle=False,
        show_tqdm=cfg["SHOW_TQDM"],
        face_detector_weights=cfg["FACE_DET_WEIGHTS"],
        visual_feat_mode="vis75",
        openface_binary=cfg.get("OPENFACE_BINARY", ""),
        precomputed_dir=cfg.get("PRECOMPUTED_V75_DIR", None),
        compute_if_missing=cfg.get("COMPUTE_IF_MISSING", False),
        fail_log_dir=cfg.get("FAIL_LOG_DIR", None),
    )
    pol = cfg["SUBSAMPLE"]["LAV-DF"]
    return subsample_loader(loader, pol["fraction"], cfg["SUBSAMPLE"]["SEED"], pol.get("balanced", False))

def make_loader_avdf1m(cfg: dict):
    loader = get_unified_av_dataloader(
        mode="av_deepfake1m",
        subset=cfg["AVDF1M_SUBSET"],
        root_dir=cfg["AVDF1M_ROOT"],
        json_path=cfg["AVDF1M_JSON"],
        frames_per_clip=cfg["FRAMES_PER_CLIP"],
        stride=cfg["STRIDE"],
        balance_minority=False,
        use_fake_periods=True,
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
        shuffle=False,
        show_tqdm=cfg["SHOW_TQDM"],
        face_detector_weights=cfg["FACE_DET_WEIGHTS"],
        visual_feat_mode="vis75",
        openface_binary=cfg.get("OPENFACE_BINARY", ""),
        precomputed_dir=cfg.get("PRECOMPUTED_V75_DIR", None),
        compute_if_missing=cfg.get("COMPUTE_IF_MISSING", False),
        fail_log_dir=cfg.get("FAIL_LOG_DIR", None),
    )
    pol = cfg["SUBSAMPLE"]["AVDF1M"]
    return subsample_loader(loader, pol["fraction"], cfg["SUBSAMPLE"]["SEED"], pol.get("balanced", False))

# =========================
# Model / ckpt helpers
# =========================
def build_model(cfg: dict, vis20_dim: int, stft_bins: int, device: torch.device):
    model, _criterion = build_dissonance_dual_model(
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
    )
    model.to(device)
    return model

def load_checkpoint(cfg: dict, model: torch.nn.Module, device: torch.device):
    ckpt_best = os.path.join(cfg["CKPT_DIR"], cfg["CKPT_BEST"])
    ckpt_last = os.path.join(cfg["CKPT_DIR"], cfg["CKPT_LAST"])
    path = ckpt_best if os.path.isfile(ckpt_best) else ckpt_last
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No checkpoint found at {ckpt_best} or {ckpt_last}")
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model"], strict=True)
    calib = payload.get("calibration", {}) or {}
    T_av = float(calib.get("T_av", 1.0))
    tau  = float(calib.get("tau", 0.25))
    dec_thr = float(calib.get("dec_thr", 0.5))
    print(f"Loaded checkpoint: {path}")
    print(f"Calibration → T_av={T_av:.3f}, tau={tau:.3f}, dec_thr={dec_thr:.3f}")
    return T_av, tau, dec_thr

# =========================
# Main test flow
# =========================
def evaluate_one(loader, model, device, *, T_av: float, tau: float, dec_thr: float, amp: bool, tag: str):
    if loader is None:
        print(f"{tag}: loader is None (skipping).")
        return None

    y, ld, la = collect_outputs(loader, model, device, amp=amp)
    if y.size == 0:
        print(f"{tag}: no usable samples (all filtered).")
        return None

    # Switching
    p_sw, conf, pick_aux = switched_probs(ld, la, T_av=T_av, tau=tau)
    m_main = compute_metrics(y.astype(np.int64), p_sw.astype(np.float32), already_probs=True, thr=dec_thr)

    # Routing summary
    total = int(y.shape[0])
    n_aux = int(pick_aux.sum())
    n_av  = total - n_aux
    pct_av  = (n_av  / max(1, total)) * 100.0
    pct_aux = (n_aux / max(1, total)) * 100.0
    print(f"[{tag}] Routing: AV={n_av}/{total} ({pct_av:.2f}%) | AUX={n_aux}/{total} ({pct_aux:.2f}%)")

    print(f"Using params → T_av={T_av:.3f}, tau={tau:.3f}, dec_thr={dec_thr:.3f}")
    print(fmt_metrics(f"{tag} (SWITCHED)", m_main, y_true=y))

    # Heads separately
    m_av  = compute_metrics(y.astype(np.int64), ld.astype(np.float32), thr=0.5)
    m_aux = compute_metrics(y.astype(np.int64), la.astype(np.float32), thr=0.5)
    print(fmt_metrics(f"{tag} (AV-Dissonance)", m_av,  y_true=y))
    print(fmt_metrics(f"{tag} (AUX)",          m_aux, y_true=y))

    return {
        "main": m_main, "av": m_av, "aux": m_aux,
        "routing": {"total": total, "av_count": n_av, "aux_count": n_aux,
                    "av_pct": pct_av, "aux_pct": pct_aux}
    }

def _probe_vis20_dim_from_loader(loader) -> int:
    """
    Robustly determine VIS-20 dimension:
      - If dataset exposes 'sel_idx', use its length.
      - Else scan batches until a valid one is found; fallback to 20.
    """
    if loader is None:
        return 20
    sel_idx = getattr(loader.dataset, "sel_idx", None)
    if sel_idx is not None and len(sel_idx) > 0:
        return int(len(sel_idx))
    # Fallback by peeking batches (skip None/invalid)
    for _batch in loader:
        unpacked = _unpack_batch_safe(_batch)
        if unpacked is None:
            continue
        xb, *_ = unpacked
        return 20 if xb.shape[1] >= 20 else int(xb.shape[1])
    return 20

def main():
    cfg = CONFIG
    os.makedirs(cfg["FAIL_LOG_DIR"], exist_ok=True)

    device = torch.device(cfg["DEVICE"])
    stft_bins = cfg["NFFT"] // 2 + 1

    # Build loaders (then subsample via wrapper)
    favc_loader   = make_loader_favc(cfg)
    lavdf_loader  = make_loader_lavdf(cfg)
    avdf1m_loader = make_loader_avdf1m(cfg)

    # VIS-20 dim from any available loader (robust to None batches)
    probe_loader = avdf1m_loader or lavdf_loader or favc_loader
    if probe_loader is None:
        raise RuntimeError("No dataset loaders could be constructed. Check your paths.")
    vis20_dim = _probe_vis20_dim_from_loader(probe_loader)

    # Build model & load checkpoint (this still gives the "default" calibration)
    model = build_model(cfg, vis20_dim=vis20_dim, stft_bins=stft_bins, device=device)
    T_av, tau, dec_thr = load_checkpoint(cfg, model, device)

    # ---- Override calibration ONLY for LAV-DF ----
    lavdf_hparams = {
        "T_av": 2.3205,
        "tau": 0.010,
        "dec_thr": 0.139,
    }

    if lavdf_loader is not None:
        evaluate_one(
            lavdf_loader, model, device,
            T_av=lavdf_hparams["T_av"],
            tau=lavdf_hparams["tau"],
            dec_thr=lavdf_hparams["dec_thr"],
            amp=cfg["AMP"], tag="LAV-DF"
        )

    print("\n========== Inference ==========")
    if favc_loader is not None:
        evaluate_one(
            favc_loader, model, device,
            T_av=T_av, tau=tau, dec_thr=dec_thr,
            amp=cfg["AMP"], tag="FAVC"
        )

    if avdf1m_loader is not None:
        evaluate_one(
            avdf1m_loader, model, device,
            T_av=T_av, tau=tau, dec_thr=dec_thr,
            amp=cfg["AMP"], tag="AV-Deepfake1M"
        )

if __name__ == "__main__":
    main()