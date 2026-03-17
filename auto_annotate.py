import argparse
import glob
import os
import sys
import time

import cv2
from tqdm import tqdm

from autodistill.detection import CaptionOntology
from autodistill_grounding_dino import GroundingDINO


# --------------------------------------------------------
# Configuration
# --------------------------------------------------------

DEFAULT_DATASET_ROOT = "/data/IDD_FGVD"
DEFAULT_OUTPUT_ROOT = "/workspace/labels"

ONTOLOGY_MAP = {
    "car": "car",
    "bus": "bus",
    "truck": "truck",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
    "autorickshaw": "autorickshaw",
    "person": "person",
    "traffic light": "traffic_light",
    "traffic sign": "traffic_sign",
    "road barrier": "barrier"
}

DEFAULT_SPLITS = ["train", "val", "test"]

BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

APPLY_NMS = True
NMS_IOU_THRESH = 0.5
SAVE_CONFIDENCE = True

SUPPORTED_MODELS = [
    "groundingdino",
    "groundedsam",
    "yoloworld",
    "owlvit",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Auto-annotate dataset using open-vocabulary detection models.")
    parser.add_argument("--model", default="groundingdino", choices=SUPPORTED_MODELS)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-confidence", action="store_true", default=SAVE_CONFIDENCE)
    parser.add_argument("--box-threshold", type=float, default=BOX_THRESHOLD)
    parser.add_argument("--text-threshold", type=float, default=TEXT_THRESHOLD)
    parser.add_argument("--nms-iou", type=float, default=NMS_IOU_THRESH)
    parser.add_argument("--disable-nms", action="store_true")
    return parser.parse_args()


def build_model(model_name, ontology, box_threshold, text_threshold):
    def _init_model(model_cls):
        # Backends expose slightly different constructor signatures.
        try:
            return model_cls(
                ontology=ontology,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
        except TypeError:
            try:
                return model_cls(
                    ontology=ontology,
                    box_threshold=box_threshold,
                )
            except TypeError:
                return model_cls(ontology=ontology)

    if model_name == "groundingdino":
        return _init_model(GroundingDINO)

    if model_name == "groundedsam":
        try:
            from autodistill_grounded_sam import GroundedSAM
        except ImportError as exc:
            raise ImportError(
                "Missing backend package for groundedsam. Install with: pip install autodistill-grounded-sam"
            ) from exc
        return _init_model(GroundedSAM)

    if model_name == "yoloworld":
        try:
            from autodistill_yolo_world import YOLOWorldModel
        except ImportError as exc:
            raise ImportError(
                "Missing backend package for yoloworld. Install with: pip install autodistill-yolo-world"
            ) from exc
        return _init_model(YOLOWorldModel)

    if model_name == "owlvit":
        try:
            from autodistill_owl_vit import OWLViT
        except ImportError as exc:
            raise ImportError(
                "Missing backend package for owlvit. Install with: pip install autodistill-owl-vit"
            ) from exc
        return _init_model(OWLViT)

    raise ValueError(f"Unsupported model: {model_name}")


# --------------------------------------------------------
# Helper function
# --------------------------------------------------------

def write_yolo_label(label_path, detections, img_w, img_h, save_confidence=False):

    os.makedirs(os.path.dirname(label_path), exist_ok=True)

    lines = []

    if detections is not None and len(detections) > 0:
        has_conf = hasattr(detections, "confidence") and detections.confidence is not None

        for idx, (xyxy, cls_id) in enumerate(zip(detections.xyxy, detections.class_id)):

            x1, y1, x2, y2 = xyxy

            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h

            if w > 0 and h > 0:
                if save_confidence and has_conf and idx < len(detections.confidence):
                    conf = float(detections.confidence[idx])
                    lines.append(
                        f"{int(cls_id)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.6f}"
                    )
                else:
                    lines.append(f"{int(cls_id)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    with open(label_path, "w") as f:
        f.write("\n".join(lines))

    return len(lines)


# --------------------------------------------------------
# Main pipeline
# --------------------------------------------------------

def main():
    args = parse_args()
    apply_nms = not args.disable_nms

    print("\nAutomatic Annotation Pipeline")
    print("Model  :", args.model)
    print("Dataset:", args.dataset_root)
    print("Output :", args.output_root)

    # ── GPU guard ────────────────────────────────────────────────────────────
    import torch
    if not torch.cuda.is_available():
        print("\nERROR: No GPU detected. Set --device cpu or fix CUDA setup.")
        sys.exit(1)
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem  = torch.cuda.get_device_properties(0).total_memory // (1024**2)
    print(f"GPU    : {gpu_name}  ({gpu_mem} MiB)")
    # ─────────────────────────────────────────────────────────────────────────

    ontology = CaptionOntology(ONTOLOGY_MAP)

    print("\nLoading model...")
    model = build_model(
        model_name=args.model,
        ontology=ontology,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    total_images = 0
    total_labels = 0

    start = time.time()

    for split in args.splits:

        img_dir = os.path.join(args.dataset_root, split, "images")
        label_dir = os.path.join(args.output_root, split)

        os.makedirs(label_dir, exist_ok=True)

        image_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

        print(f"\n[{split}] images:", len(image_paths))

        pbar = tqdm(image_paths)

        for img_path in pbar:

            name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(label_dir, name + ".txt")

            if not args.overwrite and os.path.exists(label_path):
                continue

            image = cv2.imread(img_path)

            if image is None:
                continue

            h, w = image.shape[:2]

            try:
                detections = model.predict(image)
            except NameError as exc:
                # Common GroundingDINO packaging conflict: wrong package provides `groundingdino`
                # without the required ops expected by autodistill-grounding-dino on GPU.
                if "_C" in str(exc):
                    print("\nERROR: GroundingDINO CUDA op `_C` is missing in this environment.")
                    print("This is usually caused by installing `groundingdino` from GitHub directly")
                    print("instead of `rf-groundingdino` used by autodistill-grounding-dino.")
                    print("\nFix inside the container:")
                    print("  pip uninstall -y groundingdino rf-groundingdino")
                    print("  pip install rf-groundingdino==0.1.2 --no-build-isolation")
                    print("  pip install autodistill-grounding-dino==0.1.4")
                    sys.exit(1)
                raise

            if apply_nms and detections is not None and len(detections) > 0:
                detections = detections.with_nms(
                    threshold=args.nms_iou,
                    class_agnostic=True
                )

            n = write_yolo_label(
                label_path,
                detections,
                w,
                h,
                save_confidence=args.save_confidence,
            )

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