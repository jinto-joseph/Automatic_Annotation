import glob
import os
import time

import cv2
from tqdm import tqdm

from autodistill.detection import CaptionOntology
from autodistill_grounding_dino import GroundingDINO


# --------------------------------------------------------
# Configuration
# --------------------------------------------------------

DATASET_ROOT = "/data/IDD_FGVD"
OUTPUT_ROOT = "/workspace/labels"

ONTOLOGY_MAP = {
    "car": "car",
    "bus": "bus",
    "traffic light": "traffic_light",
    "person": "person",
    "tree": "tree",
    "building": "building"
}

SPLITS = ["train", "val", "test"]

BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

APPLY_NMS = True
NMS_IOU_THRESH = 0.5


# --------------------------------------------------------
# Helper function
# --------------------------------------------------------

def write_yolo_label(label_path, detections, img_w, img_h):

    os.makedirs(os.path.dirname(label_path), exist_ok=True)

    lines = []

    if detections is not None and len(detections) > 0:
        for xyxy, cls_id in zip(detections.xyxy, detections.class_id):

            x1, y1, x2, y2 = xyxy

            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h

            if w > 0 and h > 0:
                lines.append(f"{int(cls_id)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    with open(label_path, "w") as f:
        f.write("\n".join(lines))

    return len(lines)


# --------------------------------------------------------
# Main pipeline
# --------------------------------------------------------

def main():

    print("\nAutomatic Annotation Pipeline (GroundingDINO)")
    print("Dataset:", DATASET_ROOT)
    print("Output :", OUTPUT_ROOT)

    ontology = CaptionOntology(ONTOLOGY_MAP)

    print("\nLoading GroundingDINO model...")
    model = GroundingDINO(
        ontology=ontology,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )

    total_images = 0
    total_labels = 0

    start = time.time()

    for split in SPLITS:

        img_dir = os.path.join(DATASET_ROOT, split, "images")
        label_dir = os.path.join(OUTPUT_ROOT, split)

        os.makedirs(label_dir, exist_ok=True)

        image_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

        print(f"\n[{split}] images:", len(image_paths))

        pbar = tqdm(image_paths)

        for img_path in pbar:

            name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(label_dir, name + ".txt")

            if os.path.exists(label_path):
                continue

            image = cv2.imread(img_path)

            if image is None:
                continue

            h, w = image.shape[:2]

            detections = model.predict(image)

            if APPLY_NMS and detections is not None and len(detections) > 0:
                detections = detections.with_nms(
                    threshold=NMS_IOU_THRESH,
                    class_agnostic=True
                )

            n = write_yolo_label(label_path, detections, w, h)

            total_labels += n
            total_images += 1

    end = time.time()

    print("\nAnnotation complete")
    print("Images processed:", total_images)
    print("Annotations:", total_labels)
    print("Time:", round((end - start) / 60, 2), "minutes")


# --------------------------------------------------------

if __name__ == "__main__":
    main()