"""
Annotation Visualizer + Quality Report
=======================================
Uses Supervision to:
  1. Draw bounding boxes on annotated images (saves to visualized/)
  2. Print a per-class stats table
  3. Save a summary mosaic (grid of annotated images) → annotation_preview.jpg

Run:
    .venv/bin/python3 visualize_annotations.py
"""

import glob
import os
import random

import cv2
import numpy as np
import supervision as sv

# ── Config ──────────────────────────────────────────────────────────────────
IMAGE_DIR  = "test"
LABEL_DIR  = "test_labels"
OUTPUT_DIR = "visualized"
MOSAIC_PATH = "annotation_preview.jpg"

CLASSES = ["car", "bus", "traffic_light", "person","tree","buildings"]

# Distinct colours per class  (BGR)
COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#4363D8"])

# How many images to include in the preview mosaic (shown to supervisor)
MOSAIC_COUNT = 9
MOSAIC_COLS  = 3
THUMB_SIZE   = (640, 360)   # each tile in the mosaic

# ── Helpers ──────────────────────────────────────────────────────────────────

def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return [x1, y1, x2, y2]


def load_yolo_detections(label_path, img_w, img_h):
    """Parse a YOLO .txt file and return a sv.Detections object."""
    xyxy_list, class_ids = [], []
    if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
        with open(label_path) as f:
            for line in f.read().splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
                xyxy_list.append(yolo_to_xyxy(cx, cy, w, h, img_w, img_h))
                class_ids.append(cls_id)
    if xyxy_list:
        return sv.Detections(
            xyxy=np.array(xyxy_list, dtype=np.float32),
            class_id=np.array(class_ids, dtype=int),
        )
    return sv.Detections.empty()


def draw_annotated(image, detections):
    """Draw bounding boxes + labels on an image using Supervision."""
    box_ann   = sv.BoundingBoxAnnotator(color=COLORS, thickness=2)
    label_ann = sv.LabelAnnotator(
        color=COLORS,
        text_scale=0.5,
        text_thickness=1,
        text_padding=4,
    )
    labels = [
        f"{CLASSES[cid]} #{i}"
        for i, cid in enumerate(detections.class_id)
    ] if len(detections) > 0 else []

    frame = box_ann.annotate(scene=image.copy(), detections=detections)
    frame = label_ann.annotate(scene=frame, detections=detections, labels=labels)
    return frame


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
    print(f"Found {len(image_paths)} images in '{IMAGE_DIR}/'")
    print(f"Annotated frames will be saved to '{OUTPUT_DIR}/'\n")

    # ── 1. Per-image stats ───────────────────────────────────────────────────
    class_counts   = {c: 0 for c in CLASSES}
    images_with    = {c: 0 for c in CLASSES}
    total_boxes    = 0
    empty_images   = 0
    annotated_imgs = []          # (annotated_frame, stem) for mosaic

    for img_path in image_paths:
        stem  = os.path.splitext(os.path.basename(img_path))[0]
        label = os.path.join(LABEL_DIR, stem + ".txt")

        image = cv2.imread(img_path)
        h, w  = image.shape[:2]

        dets = load_yolo_detections(label, w, h)

        if len(dets) == 0:
            empty_images += 1
        else:
            for cid in dets.class_id:
                class_counts[CLASSES[cid]] += 1
            for cls in CLASSES:
                cid = CLASSES.index(cls)
                if cid in dets.class_id:
                    images_with[cls] += 1
            total_boxes += len(dets)

        frame = draw_annotated(image, dets)
        out_path = os.path.join(OUTPUT_DIR, stem + ".jpg")
        cv2.imwrite(out_path, frame)
        annotated_imgs.append((frame, stem))

    # ── 2. Stats table ───────────────────────────────────────────────────────
    n_imgs = len(image_paths)
    print("=" * 58)
    print("  ANNOTATION QUALITY REPORT")
    print("=" * 58)
    print(f"  Dataset folder   : {os.path.abspath(IMAGE_DIR)}")
    print(f"  Total images     : {n_imgs}")
    print(f"  Images with boxes: {n_imgs - empty_images}  "
          f"({(n_imgs - empty_images) / n_imgs * 100:.1f}%)")
    print(f"  Empty images     : {empty_images}")
    print(f"  Total boxes      : {total_boxes}")
    print(f"  Avg boxes/image  : {total_boxes / max(n_imgs - empty_images, 1):.1f}")
    print()
    print(f"  {'Class':<16} {'Boxes':>6}  {'Images':>7}  {'Coverage':>9}")
    print(f"  {'-'*16} {'-'*6}  {'-'*7}  {'-'*9}")
    for cls in CLASSES:
        b = class_counts[cls]
        i = images_with[cls]
        pct = i / n_imgs * 100
        print(f"  {cls:<16} {b:>6}  {i:>7}  {pct:>8.1f}%")
    print("=" * 58)

    # ── 3. Mosaic preview  ───────────────────────────────────────────────────
    # Pick images that have detections for a richer visual
    rich = [(f, s) for f, s in annotated_imgs
            if os.path.getsize(os.path.join(LABEL_DIR, s + ".txt")) > 0]
    sample = random.sample(rich, min(MOSAIC_COUNT, len(rich)))

    tiles = []
    for frame, stem in sample:
        thumb = cv2.resize(frame, THUMB_SIZE)

        # Burn a small filename label onto each tile
        cv2.putText(thumb, stem[:28], (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(thumb, stem[:28], (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(thumb)

    # Pad to full grid
    blank = np.zeros((THUMB_SIZE[1], THUMB_SIZE[0], 3), dtype=np.uint8)
    while len(tiles) % MOSAIC_COLS != 0:
        tiles.append(blank)

    rows = [np.hstack(tiles[i:i + MOSAIC_COLS])
            for i in range(0, len(tiles), MOSAIC_COLS)]
    mosaic = np.vstack(rows)

    # Title bar
    bar_h = 60
    bar   = np.full((bar_h, mosaic.shape[1], 3), 30, dtype=np.uint8)
    title = (f"GroundingDINO Auto-Annotation  |  {n_imgs} images  |  "
             f"{total_boxes} boxes  |  {len(CLASSES)} classes")
    cv2.putText(bar, title, (16, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 220, 80), 2, cv2.LINE_AA)
    mosaic = np.vstack([bar, mosaic])

    cv2.imwrite(MOSAIC_PATH, mosaic)
    print(f"\n  Mosaic preview saved → {os.path.abspath(MOSAIC_PATH)}")
    print(f"  Individual frames    → {os.path.abspath(OUTPUT_DIR)}/")
    print()
    print("  What to show your supervisor:")
    print("  ┌─────────────────────────────────────────────────────┐")
    print(f"  │  1. annotation_preview.jpg  ← mosaic of 9 images   │")
    print(f"  │  2. visualized/             ← all 50 annotated imgs │")
    print(f"  │  3. test_labels/            ← YOLO .txt files       │")
    print(f"  │  4. The stats table above                           │")
    print("  └─────────────────────────────────────────────────────┘")


if __name__ == "__main__":
    main()
