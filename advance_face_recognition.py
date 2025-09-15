#!/usr/bin/env python
"""
FRDS2 Face Verification Demo (2 users)

What this script does
---------------------
1) Scans a dataset laid out as:
   FRDS2/
     <USER_A>/img1.jpg ...
     <USER_B>/imgX.png ...
     ...

2) Picks 2 users (either provided via --users or the first 2 valid folders).

3) Splits each user's images into enrollment ("offline training") and test ("online verification") sets.

4) Offline enrollment builds compact templates per user from many images by:
   - computing face embeddings with `face_recognition`
   - simple quality gate (blur + min face size)
   - optional dedup
   - building either a single centroid or K-means K centroids (recommended K=3..8)
   - persisting templates to disk as .npy files

5) Online verification evaluates speed/accuracy by simulating claimed-identity checks
   (image + claimed_name -> accept/reject) and produces metrics:
   - ROC / threshold sweep (TPR, FPR, FNR, Precision/Recall/F1)
   - EER estimate, best-F1 threshold, and a threshold at target FAR
   - Latency estimates: detection+embed and scoring time per image

Usage
-----
python fr_verify_demo.py \
  --root "FRDS2" \
  --users "Alice,Bob" \
  --k 5 \
  --enroll-ratio 0.7 \
  --seed 42 \
  --target-far 0.001

If --users is omitted, the first two valid user folders will be used.

Dependencies
------------
- face_recognition (dlib-backed)
- numpy, scikit-learn, opencv-python, tqdm

Notes
-----
- This script assumes **one main identity per folder** and usually **single face per image**.
- For images with multiple faces, it picks the **largest** detected face.
- For speed, detection uses HOG; switch to CNN for accuracy if you have CUDA.
"""

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

try:
    import face_recognition
except Exception as e:
    raise SystemExit("face_recognition is required. pip install face_recognition dlib")

try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None  # We'll guard use


# -----------------------------
# Utility dataclasses
# -----------------------------
@dataclass
class ImageRecord:
    path: Path
    user: str

@dataclass
class Split:
    enroll: List[ImageRecord]
    test: List[ImageRecord]


# -----------------------------
# Face utils
# -----------------------------

def l2norm(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32, copy=False)
    n = np.linalg.norm(v) + 1e-12
    return v / n


def largest_box(boxes: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    # boxes are (top, right, bottom, left)
    if not boxes:
        return None
    areas = [max(0, (b - t)) * max(0, (r - l)) for (t, r, b, l) in boxes]
    return boxes[int(np.argmax(areas))]


def laplacian_var(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def face_embed_from_path(path: Path,
                         model: str = "hog",
                         min_blur: float = 40.0,
                         min_size: int = 64) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Return (embedding(128,), chosen_box) or (None, None) if not usable.
    - model: "hog" | "cnn"
    - min_blur: discard too blurry images
    - min_size: minimal face box side to keep
    """
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        return None, None
    if laplacian_var(img_bgr) < min_blur:
        return None, None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = np.ascontiguousarray(img_rgb)
    boxes = face_recognition.face_locations(img_rgb, model=model)
    if not boxes:
        return None, None
    box = largest_box(boxes)
    if box is None:
        return None, None
    t, r, b, l = box
    if min(b - t, r - l) < min_size:
        return None, None

    encs = face_recognition.face_encodings(img_rgb, [box])
    if not encs:
        return None, None
    return encs[0].astype(np.float32), box


# -----------------------------
# Enrollment: build templates
# -----------------------------

def dedup_embeddings(E: np.ndarray, near_cos: float = 0.995) -> np.ndarray:
    """Drop near-duplicate embeddings by greedy filtering using cosine sim threshold."""
    if E.shape[0] <= 1:
        return E
    keep = [0]
    for i in range(1, E.shape[0]):
        if np.max(E[i] @ E[:i].T) < near_cos:
            keep.append(i)
    return E[keep]


def build_templates(embeds: List[np.ndarray], k_max: int = 5,
                    min_per_cluster: int = 12) -> np.ndarray:
    """Return array [K, D] of normalized templates."""
    E = np.stack([l2norm(e) for e in embeds], axis=0)
    E = dedup_embeddings(E, near_cos=0.995)
    if E.shape[0] < min_per_cluster or k_max <= 1 or KMeans is None:
        return np.expand_dims(l2norm(np.mean(E, axis=0)), 0)
    k = min(k_max, max(2, E.shape[0] // min_per_cluster))
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    km.fit(E)
    centers = np.stack([l2norm(c) for c in km.cluster_centers_], axis=0)
    return centers


# -----------------------------
# Dataset & splitting
# -----------------------------

def find_users(root: Path) -> List[str]:
    users = [p.name for p in root.iterdir() if p.is_dir()]
    users.sort()
    return users


def list_images(user_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p for p in user_dir.rglob("*") if p.suffix.lower() in exts]
    files.sort()
    return files


def make_split(paths: List[Path], user: str, ratio: float, seed: int) -> Split:
    rnd = random.Random(seed)
    paths = paths.copy()
    rnd.shuffle(paths)
    n_enroll = max(1, int(len(paths) * ratio))
    enroll = [ImageRecord(p, user) for p in paths[:n_enroll]]
    test = [ImageRecord(p, user) for p in paths[n_enroll:]]
    return Split(enroll=enroll, test=test)


# -----------------------------
# Verification & evaluation
# -----------------------------

def score_against_templates(q: np.ndarray, T: np.ndarray) -> float:
    """Return max cosine similarity between query (normalized) and templates [K,D]."""
    q = l2norm(q)
    return float(np.max(T @ q))


def simulate_verification(test_images: List[ImageRecord],
                          templates: Dict[str, np.ndarray],
                          target_far: float = 0.001,
                          model: str = "hog"):
    """Simulate claimed-name verification for positives and impostors.
    Returns metrics and per-image timing.
    """
    users = list(templates.keys())
    if len(users) != 2:
        raise ValueError("This demo expects exactly 2 users for clear positive/negative trials.")

    # Gather scores
    y_true = []  # 1 for genuine, 0 for impostor
    scores = []
    det_embed_times = []
    score_times = []

    # For each test image of user U, run two claims: claim U (genuine), claim other (impostor)
    for rec in tqdm(test_images, desc="Scoring test images"):
        emb = None
        t0 = time.perf_counter()
        emb, _ = face_embed_from_path(rec.path, model=model)
        t1 = time.perf_counter()
        if emb is None:
            # skip unusable image
            continue
        for claimed in users:
            t2 = time.perf_counter()
            s = score_against_templates(emb, templates[claimed])
            t3 = time.perf_counter()
            scores.append(s)
            y_true.append(1 if (claimed == rec.user) else 0)
            det_embed_times.append(t1 - t0)
            score_times.append(t3 - t2)

    y_true = np.array(y_true, dtype=np.int32)
    scores = np.array(scores, dtype=np.float32)
    det_embed_times = np.array(det_embed_times, dtype=np.float32)
    score_times = np.array(score_times, dtype=np.float32)

    # Threshold sweep
    if scores.size == 0:
        raise RuntimeError("No test scores computed. Check image quality/detection.")

    ths = np.linspace(-1.0, 1.0, num=1001)
    tprs, fprs, fnrs, precs, recs, f1s = [], [], [], [], [], []
    for th in ths:
        y_pred = (scores >= th).astype(np.int32)
        TP = int(np.sum((y_pred == 1) & (y_true == 1)))
        FP = int(np.sum((y_pred == 1) & (y_true == 0)))
        TN = int(np.sum((y_pred == 0) & (y_true == 0)))
        FN = int(np.sum((y_pred == 0) & (y_true == 1)))
        P = max(1, TP + FN)
        N = max(1, TN + FP)
        tpr = TP / P
        fpr = FP / N
        fnr = FN / P
        prec = TP / max(1, TP + FP)
        rec = tpr
        f1 = 2 * prec * rec / max(1e-12, (prec + rec))
        tprs.append(tpr)
        fprs.append(fpr)
        fnrs.append(fnr)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)

    tprs = np.array(tprs)
    fprs = np.array(fprs)
    fnrs = np.array(fnrs)
    f1s = np.array(f1s)

    # EER (where FPR ~= FNR)
    idx_eer = int(np.argmin(np.abs(fprs - fnrs)))
    eer = 0.5 * (fprs[idx_eer] + fnrs[idx_eer])
    th_eer = ths[idx_eer]

    # Best F1 threshold
    idx_f1 = int(np.argmax(f1s))
    th_best_f1 = ths[idx_f1]
    best_f1 = f1s[idx_f1]

    # Threshold at target FAR
    # Find smallest threshold such that FPR <= target_far
    valid = np.where(fprs <= target_far)[0]
    if len(valid) == 0:
        th_far = float("nan")
        tpr_far = float("nan")
        fpr_far = float("nan")
    else:
        idx_far = valid[0]
        th_far = ths[idx_far]
        tpr_far = tprs[idx_far]
        fpr_far = fprs[idx_far]

    timings = {
        "det_embed_ms_mean": float(np.mean(det_embed_times) * 1000.0),
        "det_embed_ms_p50": float(np.median(det_embed_times) * 1000.0),
        "det_embed_ms_p95": float(np.percentile(det_embed_times, 95) * 1000.0),
        "score_ms_mean": float(np.mean(score_times) * 1000.0),
        "score_ms_p50": float(np.median(score_times) * 1000.0),
        "score_ms_p95": float(np.percentile(score_times, 95) * 1000.0),
    }

    metrics = {
        "eer": float(eer),
        "th_eer": float(th_eer),
        "best_f1": float(best_f1),
        "th_best_f1": float(th_best_f1),
        "th_at_target_far": float(th_far),
        "tpr_at_target_far": float(tpr_far),
        "fpr_at_target_far": float(fpr_far),
        "n_trials": int(len(scores)),
    }

    return metrics, timings


# -----------------------------
# Main orchestrator
# -----------------------------

# -----------------------------
# Runtime API: single-image verification
# -----------------------------

def load_templates_for_user(user: str, out_dir: Path) -> np.ndarray:
    """Load templates for a given user from out_dir/<user>.npy; ensure correct dtype/shape and L2-normalize rows."""
    npy_path = out_dir / f"{user}.npy"
    if not npy_path.exists():
        raise FileNotFoundError(f"Templates not found for user '{user}': {npy_path}")
    T = np.load(npy_path)
    if T.ndim != 2 or T.shape[1] not in (128, 512):
        raise ValueError(f"Template shape invalid for {user}: {T.shape}")
    T = T.astype(np.float32)
    # Ensure rows are normalized (some external templates may not be)
    norms = np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
    T = T / norms
    return T


def verify_user_image(claimed_user: str,
                      image_path: str,
                      out_dir: str = "user_templates",
                      model: str = "hog",
                      threshold: float = 0.95,
                      return_all: bool = False):
    """
    Verify a single image against a claimed user.

    Parameters
    ----------
    claimed_user : str
        Username whose templates to load (expects <out_dir>/<user>.npy).
    image_path : str
        Path to the input image.
    out_dir : str
        Directory containing the saved templates.
    model : str
        face_recognition face locator: 'hog' (CPU, fast) or 'cnn' (GPU/slow but accurate).
    threshold : float
        Cosine similarity threshold to accept as the claimed user.
    return_all : bool
        If True, return a list of results for all detected faces; otherwise return the best matching face.

    Returns
    -------
    dict or list[dict]
        Each result dict has keys: {
          'claimed_user', 'decision' (True/False), 'score' (float),
          'threshold' (float), 'box' (t, r, b, l), 'timing': {detect_ms, embed_ms, score_ms, pipeline_ms},
          'k_templates' (int)
        }
    """
    T = load_templates_for_user(claimed_user, Path(out_dir))

    t0 = time.perf_counter()
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return {
            "claimed_user": claimed_user,
            "decision": False,
            "score": None,
            "threshold": threshold,
            "box": None,
            "timing": {"detect_ms": 0.0, "embed_ms": 0.0, "score_ms": 0.0, "pipeline_ms": 0.0},
            "k_templates": int(T.shape[0]),
            "reason": "image_not_found",
        }

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = np.ascontiguousarray(img_rgb)

    t_det0 = time.perf_counter()
    boxes = face_recognition.face_locations(img_rgb, model=model)
    t_det1 = time.perf_counter()
    if not boxes:
        return {
            "claimed_user": claimed_user,
            "decision": False,
            "score": None,
            "threshold": threshold,
            "box": None,
            "timing": {"detect_ms": (t_det1 - t_det0) * 1000.0, "embed_ms": 0.0, "score_ms": 0.0, "pipeline_ms": (t_det1 - t_det0) * 1000.0},
            "k_templates": int(T.shape[0]),
            "reason": "no_face_detected",
        }

    # Encode all detected faces
    t_emb0 = time.perf_counter()
    encs = face_recognition.face_encodings(img_rgb, boxes)
    t_emb1 = time.perf_counter()

    detect_ms = (t_det1 - t_det0) * 1000.0
    embed_ms = (t_emb1 - t_emb0) * 1000.0

    results = []
    for box, enc in zip(boxes, encs):
        q = l2norm(enc.astype(np.float32))
        t_s0 = time.perf_counter()
        score = float(np.max(T @ q))  # cosine similarity
        t_s1 = time.perf_counter()
        score_ms = (t_s1 - t_s0) * 1000.0
        pipeline_ms = detect_ms + embed_ms + score_ms
        results.append({
            "claimed_user": claimed_user,
            "decision": bool(score >= threshold),
            "score": score,
            "threshold": threshold,
            "box": box,
            "timing": {
                "detect_ms": detect_ms,
                "embed_ms": embed_ms,
                "score_ms": score_ms,
                "pipeline_ms": pipeline_ms,
            },
            "k_templates": int(T.shape[0]),
        })

    if return_all:
        return results
    best = max(results, key=lambda r: -1.0 if r["score"] is None else r["score"])
    return best


# -----------------------------
# Runtime API: enroll a new user from a folder
# -----------------------------

def enroll_user_from_folder(user: str,
                            folder_path: str,
                            out_dir: str = "user_templates",
                            model: str = "hog",
                            k_max: int = 5,
                            min_per_cluster: int = 12,
                            overwrite: bool = False,
                            verbose: bool = False,
                            min_blur: float = 40.0,
                            min_size: int = 64):
    """
    Create or refresh templates for a user from a folder of images.

    Parameters
    ----------
    user : str
        Target username. Templates are stored at <out_dir>/<user>.npy
    folder_path : str
        Directory containing images (searched recursively).
    out_dir : str
        Directory to save templates.
    model : str
        face_recognition face locator: 'hog' or 'cnn'.
    k_max : int
        Max number of KMeans centroids to produce.
    min_per_cluster : int
        Minimum samples per cluster to enable multi-centroid.
    overwrite : bool
        If False and templates exist, do not overwrite; return status='exists'.
        If True, overwrite existing templates.
    verbose : bool
        Print diagnostics.
    min_blur : float, min_size : int
        Quality gates for enrollment.

    Returns
    -------
    dict with keys: {
        'user', 'status' ('created'|'overwritten'|'exists'|'no_images'|'no_usable_images'),
        'templates_path', 'k', 'n_images', 'n_usable'
    }
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    npy_path = out / f"{user}.npy"

    if npy_path.exists() and not overwrite:
        return {
            "user": user,
            "status": "exists",
            "templates_path": str(npy_path),
            "k": None,
            "n_images": None,
            "n_usable": None,
        }

    # Gather images
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        return {
            "user": user,
            "status": "no_images",
            "templates_path": None,
            "k": None,
            "n_images": 0,
            "n_usable": 0,
        }
    paths = list_images(folder)
    if len(paths) == 0:
        return {
            "user": user,
            "status": "no_images",
            "templates_path": None,
            "k": None,
            "n_images": 0,
            "n_usable": 0,
        }

    # Encode and filter
    embeds = []
    if verbose:
        print(f"[Enroll] {user}: scanning {len(paths)} images in {folder}")
    for p in tqdm(paths, desc=f"Embedding {user}"):
        e, _ = face_embed_from_path(p, model=model, min_blur=min_blur, min_size=min_size)
        if e is not None:
            embeds.append(e)
    n_usable = len(embeds)
    if n_usable == 0:
        return {
            "user": user,
            "status": "no_usable_images",
            "templates_path": None,
            "k": None,
            "n_images": len(paths),
            "n_usable": 0,
        }

    # Build templates
    T = build_templates(embeds, k_max=k_max, min_per_cluster=min_per_cluster).astype(np.float32)
    if npy_path.exists() and overwrite:
        try:
            npy_path.unlink()
        except Exception:
            pass
    np.save(npy_path, T)

    return {
        "user": user,
        "status": "overwritten" if overwrite and npy_path.exists() else "created",
        "templates_path": str(npy_path),
        "k": int(T.shape[0]),
        "n_images": len(paths),
        "n_usable": n_usable,
    }


def is_name_not_in_list(name: str) -> bool:

    denied_users = [
        "Adriana Lima",
        "Alex Lawther",
        "Alvaro Morte",
        "alycia dabnem carey",
        "Anthony Mackie",
        "Avril Lavigne",
        "barack obama",
        "Ben Affleck",
        "Brenton Thwaites",
        "Brian J. Smith",
        "camila mendes",
        "Chris Evans",
        "Chris Pratt",
        "Dominic Purcell",
        "elizabeth olsen",
        "ellen page",
        "elon musk",
        "Emilia Clarke",
        "Emma Stone",
        "Emma Watson",
        "gal gadot",
        "grant gustin",
        "Henry Cavil",
        "Hugh Jackman",
        "Inbar Lavi",
        "jeff bezos",
        "Jennifer Lawrence",
        "Jeremy Renner",
        "Jessica Barden",
        "Katharine Mcphee",
        "Katherine Langford",
        "Keanu Reeves",
        "kiernen shipka",
        "Krysten Ritter",
        "Lili Reinhart",
        "Lionel Messi",
        "Logan Lerman",
        "Maisie Williams",
        "Maria Pedraza",
        "Miley Cyrus",
        "Morena Baccarin",
        "Natalie Dormer",
        "Neil Patrick Harris",
        "Pedro Alonso",
        "Rami Malek",
        "Richard Harmon",
        "Robert De Niro",
        "scarlett johansson",
        "Shakira Isabel Mebarak",
        "Stephen Amell",
        "Tom Holland",
        "Tuppence Middleton",
        "Ursula Corbero",
        "Wentworth Miller",
        "Zac Efron",
        "Zendaya",
        "Zoe Saldana",
    ]

    # compare everything in lowercase
    return name.lower() not in (n.lower() for n in denied_users)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, required=True, help='Path to FRDS2 root folder')
    ap.add_argument('--users', type=str, default='', help='Comma-separated two users to include')
    ap.add_argument('--enroll-ratio', type=float, default=0.7, help='Fraction of images for enrollment')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--k', type=int, default=5, help='Max K-means centroids per user (templates)')
    ap.add_argument('--model', type=str, default='hog', choices=['hog', 'cnn'], help='face_locations model')
    ap.add_argument('--target-far', type=float, default=0.001, help='Target false accept rate for threshold selection')
    ap.add_argument('--out', type=str, default='user_templates', help='Directory to save templates and splits')
    ap.add_argument('--min-per-cluster', type=int, default=12, help='Minimum samples per KMeans cluster before using multi-centroid')
    ap.add_argument('--verbose', action='store_true', help='Print extra diagnostics during enrollment/verification')
    ap.add_argument('--skip-enroll', action='store_true', help='Skip building embeddings/templates and load existing templates from --out directory')
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    users = []
    if args.users:
        users = [u.strip() for u in args.users.split(',') if u.strip()]
    else:
        users = find_users(root)[:2]

    if len(users) != 2:
        raise SystemExit(f"Need exactly 2 users. Got: {users}")

    # Build splits
    splits: Dict[str, Split] = {}
    for u in users:
        paths = list_images(root / u)
        if len(paths) < 10:
            raise SystemExit(f"User {u} has too few images ({len(paths)}). Need >= 10.")
        splits[u] = make_split(paths, u, ratio=args.enroll_ratio, seed=args.seed)

    # Persist split for reproducibility
    split_json = {
        u: {
            'enroll': [str(rec.path) for rec in splits[u].enroll],
            'test': [str(rec.path) for rec in splits[u].test],
        } for u in users
    }
    (out_dir / 'split.json').write_text(json.dumps(split_json, indent=2))

    # Offline enrollment: compute templates and save
    templates: Dict[str, np.ndarray] = {}
    if args.skip_enroll:
        # Load precomputed templates from disk
        for u in users:
            npy_path = out_dir / f"{u}.npy"
            if not npy_path.exists():
                raise SystemExit(f"--skip-enroll set but templates not found for {u}: {npy_path}")
            T = np.load(npy_path)
            if T.ndim != 2 or T.shape[1] not in (128, 512):
                raise SystemExit(f"Template shape invalid for {u}: {T.shape}")
            templates[u] = T.astype(np.float32)
            print(f"Loaded templates: {u}.npy with shape {templates[u].shape}")
    else:
        for u in users:
            embeds = []
            print(f"[Enrollment] {u}: {len(splits[u].enroll)} images")
            for rec in tqdm(splits[u].enroll, desc=f"Embedding {u}"):
                e, _ = face_embed_from_path(rec.path, model=args.model)
                if e is not None:
                    embeds.append(e)
            if len(embeds) < 3:
                raise SystemExit(f"Not enough usable enroll embeddings for {u} (got {len(embeds)}).")
            # Diagnostics: how many survive dedup? and what K will we use?
            try:
                E_tmp = np.stack([l2norm(e) for e in embeds], axis=0)
                E_dedup = dedup_embeddings(E_tmp, near_cos=0.995)
                if args.verbose:
                    print(f"[Enrollment] {u}: usable={len(embeds)} after_dedup={E_dedup.shape[0]}")
            except Exception:
                pass
            T = build_templates(embeds, k_max=args.k, min_per_cluster=args.min_per_cluster)
            if args.verbose:
                print(f"[Enrollment] {u}: templates K={T.shape[0]}")
            templates[u] = T.astype(np.float32)
            np.save(out_dir / f"{u}.npy", templates[u])
            print(f"Saved templates: {u}.npy with shape {templates[u].shape}")

    # Online verification (evaluation)
    test_images = splits[users[0]].test + splits[users[1]].test
    metrics, timings = simulate_verification(test_images, templates, target_far=args.target_far, model=args.model)

    print("\n================= RESULTS =================")
    print(f"Users: {users}")
    print(json.dumps(metrics, indent=2))
    print("\nTimings (ms):")
    for k, v in timings.items():
        print(f"  {k}: {v:.2f}")

    # Provide a recommended threshold
    th = metrics['th_at_target_far'] if not np.isnan(metrics['th_at_target_far']) else metrics['th_best_f1']
    print(f"\nRecommended threshold: {th:.4f}  (cosine similarity)")
    print("==========================================\n")


if __name__ == '__main__':

    main()