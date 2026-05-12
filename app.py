"""
Traffic Violation Detection - YOLOv8 Flask App
Run: python app.py
"""

import os
import uuid
import threading
import subprocess
from collections import Counter
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
MODEL_PATH    = os.path.join(BASE_DIR, "model", "best_fixed.pt")

ALLOWED_EXT   = {"mp4", "avi", "mov", "mkv"}
CONFIDENCE    = 0.25
IOU           = 0.45
FRAME_SKIP    = 1   # 1 = every frame, 2 = every other frame (faster but less accurate)

TRIPLE_RIDING_MIN    = 3      # riders on one bike to trigger violation
MAX_RIDER_BIKE_RATIO = 0.30   # rider assigned to nearest bike only within 30% of frame diagonal

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

# ─────────────────────────────────────────────
# LOAD YOLO MODEL
# ─────────────────────────────────────────────
model = None

# Violation classes shown in red, safe classes in green
VIOLATION_CLASSES = {"Helmet_Off", "No_helmet"}
ALL_VIOLATION_CLASSES = VIOLATION_CLASSES | {"Triple_Riding"}

# Model class name sets for bike/rider (lowercase for case-insensitive match)
BIKE_CLASS_NAMES  = {"bike"}
RIDER_CLASS_NAMES = {"rider"}

COLOR_MAP = {}  # populated after model loads

def load_model():
    global model, COLOR_MAP
    if not os.path.exists(MODEL_PATH):
        print(f"[WARN] Model not found at {MODEL_PATH} — detections disabled.")
        return
    model = YOLO(MODEL_PATH)
    for cls_id, cls_name in model.names.items():
        if cls_name in VIOLATION_CLASSES:
            COLOR_MAP[cls_id] = (0, 0, 255)    # RED  — violation
        else:
            COLOR_MAP[cls_id] = (0, 200, 80)   # GREEN — safe
    print(f"[INFO] YOLOv8 loaded | classes: {list(model.names.values())}")

load_model()

# ─────────────────────────────────────────────
# JOB STORE
# ─────────────────────────────────────────────
jobs: dict[str, dict] = {}

# ─────────────────────────────────────────────
# TRIPLE-RIDING HELPERS
# ─────────────────────────────────────────────
def _intersection_area(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def assign_riders_to_bikes(bike_boxes, rider_boxes, frame_diag):
    """
    Returns {bike_idx: [rider_idx, ...]} using a two-tier strategy:
      1. Overlap-first: if a rider overlaps any bike, it belongs to the one
         whose x-center is closest (handles side-by-side bikes correctly —
         a rider's horizontal midpoint is naturally nearer its own bike).
      2. Fallback: no overlap → nearest bike by Euclidean distance, capped at
         MAX_RIDER_BIKE_RATIO of the frame diagonal to ignore distant bikes.
    """
    assignments = {i: [] for i in range(len(bike_boxes))}
    max_dist = frame_diag * MAX_RIDER_BIKE_RATIO

    for ri, rbox in enumerate(rider_boxes):
        r_cx = (rbox[0] + rbox[2]) / 2
        r_cy = (rbox[1] + rbox[3]) / 2

        overlapping = []
        for bi, bbox in enumerate(bike_boxes):
            if _intersection_area(rbox, bbox) > 0:
                b_cx = (bbox[0] + bbox[2]) / 2
                overlapping.append((bi, abs(b_cx - r_cx)))

        if overlapping:
            # Pick the bike whose x-center is horizontally closest to the rider
            best_bike = min(overlapping, key=lambda t: t[1])[0]
        else:
            best_bike = None
            min_dist  = float("inf")
            for bi, bbox in enumerate(bike_boxes):
                b_cx = (bbox[0] + bbox[2]) / 2
                b_cy = (bbox[1] + bbox[3]) / 2
                dist = ((r_cx - b_cx) ** 2 + (r_cy - b_cy) ** 2) ** 0.5
                if dist < min_dist and dist < max_dist:
                    min_dist  = dist
                    best_bike = bi

        if best_bike is not None:
            assignments[best_bike].append(ri)

    return assignments


def _draw_box(img, box, color, label):
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
def predict_frame(frame_bgr):
    results = model.predict(
        source=frame_bgr,
        conf=CONFIDENCE,
        iou=IOU,
        verbose=False,
        device=0 if __import__('torch').cuda.is_available() else 'cpu'
    )[0]

    annotated  = frame_bgr.copy()
    detections = []

    # Bucket raw detections by role
    bike_dets  = []
    rider_dets = []
    other_dets = []

    for box in results.boxes:
        cls_id   = int(box.cls[0])
        score    = float(box.conf[0])
        cls_name = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        det = {"cls_id": cls_id, "cls_name": cls_name,
               "score": score, "box": [x1, y1, x2, y2]}
        detections.append({"class": cls_name, "score": round(score, 3),
                            "box": [x1, y1, x2, y2]})
        name_lower = cls_name.lower()
        if name_lower in BIKE_CLASS_NAMES:
            bike_dets.append(det)
        elif name_lower in RIDER_CLASS_NAMES:
            rider_dets.append(det)
        else:
            other_dets.append(det)

    # ── Triple-riding check ──────────────────────────────────────────────────
    fh, fw = frame_bgr.shape[:2]
    frame_diag  = (fw ** 2 + fh ** 2) ** 0.5
    bike_boxes  = [d["box"] for d in bike_dets]
    rider_boxes = [d["box"] for d in rider_dets]
    assignments = assign_riders_to_bikes(bike_boxes, rider_boxes, frame_diag)
    triple_bikes = {bi for bi, riders in assignments.items()
                    if len(riders) >= TRIPLE_RIDING_MIN}

    # ── Draw bikes ───────────────────────────────────────────────────────────
    for bi, d in enumerate(bike_dets):
        if bi in triple_bikes:
            n     = len(assignments[bi])
            color = (0, 0, 255)   # RED — triple-riding violation
            label = f"Triple Riding ({n} riders) {d['score']:.2f}"
            detections.append({"class": "Triple_Riding",
                                "score": round(d["score"], 3), "box": d["box"]})
        else:
            color = COLOR_MAP.get(d["cls_id"], (0, 200, 80))
            label = f"{d['cls_name']} {d['score']:.2f}"
        _draw_box(annotated, d["box"], color, label)

    # ── Draw riders ──────────────────────────────────────────────────────────
    for d in rider_dets:
        color = COLOR_MAP.get(d["cls_id"], (0, 200, 80))
        _draw_box(annotated, d["box"], color, f"{d['cls_name']} {d['score']:.2f}")

    # ── Draw everything else (helmets, no_helmet, …) ────────────────────────
    for d in other_dets:
        color = COLOR_MAP.get(d["cls_id"], (0, 200, 80))
        _draw_box(annotated, d["box"], color, f"{d['cls_name']} {d['score']:.2f}")

    return annotated, detections


def process_video(job_id: str, input_path: str):
    raw_out   = os.path.join(OUTPUT_FOLDER, f"{job_id}_raw.mp4")
    final_out = os.path.join(OUTPUT_FOLDER, f"{job_id}_out.mp4")

    cap          = cv2.VideoCapture(input_path)
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25
    w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    jobs[job_id].update({"total": total_frames, "status": "processing"})

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(raw_out, fourcc, fps, (w, h))

    frame_idx      = 0
    all_detections = []
    prev_annotated = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if model and frame_idx % FRAME_SKIP == 0:
            annotated, dets = predict_frame(frame)
            all_detections.extend(dets)
            prev_annotated = annotated
        else:
            annotated = prev_annotated if prev_annotated is not None else frame

        writer.write(annotated)
        frame_idx += 1
        jobs[job_id]["progress"] = frame_idx

    cap.release()
    writer.release()

    # Re-encode to H.264 for browser playback
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", raw_out,
             "-vcodec", "libx264", "-pix_fmt", "yuv420p", final_out],
            check=True, capture_output=True
        )
        os.remove(raw_out)
        out_file = final_out
    except Exception:
        out_file = raw_out

    counts = dict(Counter(d["class"] for d in all_detections))
    violation_count = sum(v for k, v in counts.items() if k in ALL_VIOLATION_CLASSES)

    jobs[job_id].update({
        "status":          "done",
        "output":          os.path.basename(out_file),
        "detections":      counts,
        "total_det":       len(all_detections),
        "violation_count": violation_count,
    })


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files["video"]
    if not f.filename:
        return jsonify({"error": "No selected file"}), 400
    ext = f.filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Unsupported format: {ext}"}), 400

    job_id   = str(uuid.uuid4())[:8]
    filename = secure_filename(f"{job_id}_{f.filename}")
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    f.save(save_path)

    jobs[job_id] = {"status": "queued", "progress": 0, "total": 1}
    t = threading.Thread(target=process_video, args=(job_id, save_path), daemon=True)
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job"}), 404
    return jsonify(job)


@app.route("/outputs/<filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


if __name__ == "__main__":
    print("[INFO] Starting → http://127.0.0.1:5001")
    app.run(debug=True, port=5001)