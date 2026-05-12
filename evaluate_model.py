"""
Model Performance Evaluator
----------------------------
Usage:
  python evaluate_model.py                        # checkpoint metrics + speed
  python evaluate_model.py --data path/data.yaml  # also runs live validation
  python evaluate_model.py --imgsz 320            # smaller size for speed test
"""

import os
import sys
import argparse
import io
import csv
import time
import numpy as np
import torch
from ultralytics import YOLO

MODEL_PATH    = os.path.join(os.path.dirname(__file__), "model", "best_fixed.pt")
WARMUP_RUNS   = 5
BENCHMARK_RUNS = 50
WIDTH         = 66


# ─────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────
def bar(char="─"):
    print(char * WIDTH)

def header(title):
    bar("═")
    pad = (WIDTH - len(title) - 2) // 2
    print(" " * pad + f" {title} " + " " * pad)
    bar("═")

def section(title):
    print()
    bar()
    print(f"  {title}")
    bar()

def row(label, value, width=24):
    print(f"  {label:<{width}}{value}")

def pct(v):
    return f"{float(v):.4f}   ({float(v)*100:.2f}%)"


# ─────────────────────────────────────────────────────────────────
# Checkpoint metric extraction
# ─────────────────────────────────────────────────────────────────
METRIC_KEYS = [
    ("metrics/precision(B)", "Precision"),
    ("metrics/recall(B)",    "Recall"),
    ("metrics/mAP50(B)",     "mAP@50"),
    ("metrics/mAP50-95(B)",  "mAP@50-95"),
]
LOSS_KEYS = [
    ("val/box_loss", "Val Box Loss"),
    ("val/cls_loss", "Val Cls Loss"),
    ("val/dfl_loss", "Val DFL Loss"),
]


def extract_ckpt_metrics(ckpt: dict) -> dict:
    """Return best-epoch metrics from the checkpoint if they exist."""
    # ultralytics >= 8.1 stores train_metrics as a flat dict
    m = ckpt.get("train_metrics") or {}
    if m:
        return {k.strip(): float(v) for k, v in m.items() if v is not None}

    # older versions store a CSV string under train_results
    csv_text = ckpt.get("train_results", "")
    if csv_text:
        reader = csv.DictReader(io.StringIO(csv_text))
        rows = list(reader)
        if rows:
            last = {k.strip(): v.strip() for k, v in rows[-1].items()}
            try:
                return {k: float(v) for k, v in last.items() if v}
            except ValueError:
                pass

    return {}


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Model Performance Evaluator")
    parser.add_argument("--data",   default=None,
                        help="Path to dataset YAML for live validation (optional)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Inference image size (default 640)")
    parser.add_argument("--conf",  type=float, default=0.25)
    parser.add_argument("--iou",   type=float, default=0.45)
    args = parser.parse_args()

    header("Traffic Violation Detection — Model Evaluation")

    # ── Load ──────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"\n  ERROR: model not found at {MODEL_PATH}")
        sys.exit(1)

    print(f"\n  Model  : {MODEL_PATH}")
    model  = YOLO(MODEL_PATH)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"  Device : {device}")

    # ── 1. Architecture ───────────────────────────────────────────
    section("1. Architecture")

    class_names = list(model.names.values())
    row("Classes", f"{len(class_names)}  →  {', '.join(class_names)}")

    try:
        info = model.info(verbose=False)
        if isinstance(info, (list, tuple)) and len(info) >= 4:
            layers, params, grads, gflops = info[:4]
            row("Layers",     f"{int(layers):,}")
            row("Parameters", f"{int(params):,}")
            row("GFLOPs",     f"{float(gflops):.2f}")
    except Exception:
        pass  # some ultralytics builds return nothing

    ckpt = getattr(model, "ckpt", {}) or {}
    epoch = ckpt.get("epoch")
    best  = ckpt.get("best_fitness")
    if epoch is not None:
        row("Trained epochs", str(int(epoch) + 1))
    if best is not None:
        row("Best fitness",   f"{float(best):.4f}")

    # ── 2. Stored Training Metrics ────────────────────────────────
    section("2. Stored Training / Validation Metrics")

    metrics = extract_ckpt_metrics(ckpt)
    if metrics:
        print()
        found_any = False
        for key, label in METRIC_KEYS:
            if key in metrics:
                row(label, pct(metrics[key]))
                found_any = True
        print()
        for key, label in LOSS_KEYS:
            if key in metrics:
                row(label, f"{metrics[key]:.6f}")
                found_any = True
        if not found_any:
            print("  (No recognizable metric keys in checkpoint.)")
    else:
        print()
        print("  No stored metrics found in this checkpoint.")
        print("  Supply --data <yaml> to run live validation instead.")

    # ── 3. Confusion / per-class from checkpoint ──────────────────
    # ultralytics stores per-class AP in train_results CSV columns like
    # "metrics/mAP50(B)/ClassName" (newer versions only)
    per_class = {}
    for key, val in metrics.items():
        # e.g. "metrics/mAP50(B)/Bike"
        if key.startswith("metrics/") and key.count("/") == 2:
            _, metric_type, cls_name = key.split("/")
            metric_type = metric_type.rstrip("(B)")
            per_class.setdefault(cls_name, {})[metric_type] = val

    if per_class:
        section("3. Per-Class Metrics (from checkpoint)")
        col = 16
        print(f"\n  {'Class':<{col}} {'Precision':>11} {'Recall':>9} {'mAP50':>9} {'mAP50-95':>11}")
        bar()
        for cls_name, m in per_class.items():
            p    = m.get("precision", float("nan"))
            r    = m.get("recall",    float("nan"))
            a50  = m.get("mAP50",     float("nan"))
            a95  = m.get("mAP50-95",  float("nan"))
            print(f"  {cls_name:<{col}} {p:>11.4f} {r:>9.4f} {a50:>9.4f} {a95:>11.4f}")

    # ── 4. Inference Speed ────────────────────────────────────────
    sec_num = 4 if per_class else 3
    section(f"{sec_num}. Inference Speed  (image {args.imgsz}×{args.imgsz})")

    dummy = np.random.randint(0, 255, (args.imgsz, args.imgsz, 3), dtype=np.uint8)

    print(f"\n  Warming up ({WARMUP_RUNS} runs)…")
    for _ in range(WARMUP_RUNS):
        model.predict(dummy, verbose=False, device=device,
                      conf=args.conf, iou=args.iou)

    print(f"  Benchmarking ({BENCHMARK_RUNS} runs)…")
    times = []
    for _ in range(BENCHMARK_RUNS):
        t0 = time.perf_counter()
        model.predict(dummy, verbose=False, device=device,
                      conf=args.conf, iou=args.iou)
        times.append((time.perf_counter() - t0) * 1000)

    t = np.array(times)
    print()
    row("Mean latency",  f"{t.mean():.2f} ms")
    row("Median latency",f"{np.median(t):.2f} ms")
    row("Min / Max",     f"{t.min():.2f} ms  /  {t.max():.2f} ms")
    row("Std dev",       f"{t.std():.2f} ms")
    row("Throughput",    f"{1000/t.mean():.1f} FPS")

    # ── 5. Live Validation (optional) ────────────────────────────
    if args.data:
        sec_num += 1
        section(f"{sec_num}. Live Validation  →  {args.data}")

        if not os.path.exists(args.data):
            print(f"\n  ERROR: data file not found: {args.data}")
        else:
            print("\n  Running model.val() — this may take a few minutes…\n")
            val = model.val(
                data=args.data,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=device,
                verbose=False,
            )

            # Overall
            b = val.box
            mp    = float(b.mp)    if b.mp    is not None else float("nan")
            mr    = float(b.mr)    if b.mr    is not None else float("nan")
            map50 = float(b.map50) if b.map50 is not None else float("nan")
            mapv  = float(b.map)   if b.map   is not None else float("nan")

            print()
            row("Overall Precision",  pct(mp))
            row("Overall Recall",     pct(mr))
            row("Overall mAP@50",     pct(map50))
            row("Overall mAP@50-95",  pct(mapv))

            # Per-class table
            names = val.names
            if names and b.ap50 is not None:
                print()
                col = 16
                print(f"  {'Class':<{col}} {'Precision':>11} {'Recall':>9} "
                      f"{'mAP50':>9} {'mAP50-95':>11}")
                bar()
                for i, name in names.items():
                    try:
                        p_   = float(b.p[i])    if b.p    is not None else float("nan")
                        r_   = float(b.r[i])    if b.r    is not None else float("nan")
                        a50_ = float(b.ap50[i]) if b.ap50 is not None else float("nan")
                        a95_ = float(b.ap[i])   if b.ap   is not None else float("nan")
                        print(f"  {name:<{col}} {p_:>11.4f} {r_:>9.4f} "
                              f"{a50_:>9.4f} {a95_:>11.4f}")
                    except (IndexError, TypeError):
                        pass
                bar()
                print(f"  {'ALL (mean)':<{col}} {mp:>11.4f} {mr:>9.4f} "
                      f"{map50:>9.4f} {mapv:>11.4f}")

    # ── Done ──────────────────────────────────────────────────────
    print()
    bar("═")
    print("  Evaluation complete.")
    bar("═")
    print()


if __name__ == "__main__":
    main()
