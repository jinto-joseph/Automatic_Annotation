import argparse
import glob
import os
import random
import xml.etree.ElementTree as ET

try:
    import cv2
except ImportError:
    cv2 = None
    from PIL import Image, ImageDraw


GT_CLASS_ALIASES = {
    "car": "car",
    "bus": "bus",
    "mini-bus": "bus",
    "truck": "truck",
    "motorcycle": "motorcycle",
    "scooter": "motorcycle",
    "bicycle": "bicycle",
    "cycle": "bicycle",
    "autorickshaw": "autorickshaw",
    "person": "person",
    "traffic light": "traffic_light",
    "traffic_light": "traffic_light",
    "traffic sign": "traffic_sign",
    "traffic_sign": "traffic_sign",
    "road barrier": "barrier",
    "barrier": "barrier",
}


def load_yolo(label_path):
    items = []
    if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
        return items

    with open(label_path, "r", encoding="utf-8") as f:
        for raw in f:
            parts = raw.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:5])
            conf = float(parts[5]) if len(parts) >= 6 else None
            items.append((cls_id, cx, cy, w, h, conf))
    return items


def normalize_gt_class(raw_name):
    if not raw_name:
        return None
    prefix = raw_name.strip().split("_")[0].strip().lower()
    return GT_CLASS_ALIASES.get(prefix)


def load_voc_xml(label_path, class_to_id):
    items = []
    if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
        return items

    tree = ET.parse(label_path)
    root = tree.getroot()

    width = float(root.findtext("size/width", default="0"))
    height = float(root.findtext("size/height", default="0"))
    if width <= 0 or height <= 0:
        return items

    for obj in root.findall("object"):
        class_name = normalize_gt_class(obj.findtext("name", default=""))
        if class_name is None or class_name not in class_to_id:
            continue

        bbox = obj.find("bndbox")
        if bbox is None:
            continue

        xmin = float(bbox.findtext("xmin", default="0"))
        ymin = float(bbox.findtext("ymin", default="0"))
        xmax = float(bbox.findtext("xmax", default="0"))
        ymax = float(bbox.findtext("ymax", default="0"))

        cx = ((xmin + xmax) / 2.0) / width
        cy = ((ymin + ymax) / 2.0) / height
        bw = (xmax - xmin) / width
        bh = (ymax - ymin) / height

        if bw <= 0 or bh <= 0:
            continue

        items.append((class_to_id[class_name], cx, cy, bw, bh, None))

    return items


def to_xyxy(cx, cy, w, h, img_w, img_h):
    x1 = int((cx - (w / 2.0)) * img_w)
    y1 = int((cy - (h / 2.0)) * img_h)
    x2 = int((cx + (w / 2.0)) * img_w)
    y2 = int((cy + (h / 2.0)) * img_h)
    return x1, y1, x2, y2


def load_image(path):
    if cv2 is not None:
        return cv2.imread(path)
    return Image.open(path).convert("RGB")


def save_image(path, image):
    if cv2 is not None:
        cv2.imwrite(path, image)
        return
    image.save(path)


def get_image_size(image):
    if cv2 is not None:
        h, w = image.shape[:2]
        return w, h
    return image.size


def draw_text(image, position, text, color):
    if cv2 is not None:
        cv2.putText(
            image,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
        return

    draw = ImageDraw.Draw(image)
    draw.text(position, text, fill=color)


def draw_boxes(image, yolo_rows, class_names, color, prefix, show_conf=False):
    w, h = get_image_size(image)
    for cls_id, cx, cy, bw, bh, conf in yolo_rows:
        x1, y1, x2, y2 = to_xyxy(cx, cy, bw, bh, w, h)
        if cv2 is not None:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        else:
            draw = ImageDraw.Draw(image)
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)

        name = class_names[cls_id] if 0 <= cls_id < len(class_names) else f"cls_{cls_id}"
        label = f"{prefix}:{name}"
        if show_conf and conf is not None:
            label += f" {conf:.2f}"

        draw_text(image, (max(0, x1), max(16, y1 - 6)), label, color)


def main():
    parser = argparse.ArgumentParser(description="Visualize GT vs Predictions.")
    parser.add_argument("--images-dir", default="/data/IDD_FGVD/test/images")
    parser.add_argument("--gt-dir", default="/data/IDD_FGVD/test/annos")
    parser.add_argument("--pred-dir", default="/workspace/labels/test")
    parser.add_argument("--out-dir", default="/workspace/results/visualizations/groundingdino_test")
    parser.add_argument("--sample", type=int, default=50)
    parser.add_argument("--ground-truth-format", default="auto", choices=["auto", "yolo", "vocxml"])
    parser.add_argument(
        "--classes",
        nargs="+",
        default=[
            "car",
            "bus",
            "truck",
            "motorcycle",
            "bicycle",
            "autorickshaw",
            "person",
            "traffic_light",
            "traffic_sign",
            "barrier",
        ],
    )
    args = parser.parse_args()
    class_to_id = {name: idx for idx, name in enumerate(args.classes)}

    os.makedirs(args.out_dir, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(args.images_dir, "*.jpg")))
    if not image_paths:
        image_paths = sorted(glob.glob(os.path.join(args.images_dir, "*.png")))

    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.images_dir}")

    gt_format = args.ground_truth_format
    if gt_format == "auto":
        gt_format = "vocxml" if glob.glob(os.path.join(args.gt_dir, "*.xml")) else "yolo"

    sample_count = min(args.sample, len(image_paths))
    chosen = random.sample(image_paths, sample_count)

    for img_path in chosen:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        pred_path = os.path.join(args.pred_dir, stem + ".txt")

        image = load_image(img_path)
        if image is None:
            continue

        if gt_format == "vocxml":
            gt_path = os.path.join(args.gt_dir, stem + ".xml")
            gt_rows = load_voc_xml(gt_path, class_to_id)
        else:
            gt_path = os.path.join(args.gt_dir, stem + ".txt")
            gt_rows = load_yolo(gt_path)

        pred_rows = load_yolo(pred_path)

        # Green: GT, Red: predictions
        draw_boxes(image, gt_rows, args.classes, (0, 255, 0), "GT", show_conf=False)
        draw_boxes(image, pred_rows, args.classes, (0, 0, 255), "PRED", show_conf=True)

        draw_text(image, (10, 28), "Green=Ground Truth | Red=Prediction", (255, 255, 255))

        out_path = os.path.join(args.out_dir, stem + ".jpg")
        save_image(out_path, image)

    print(f"Saved {sample_count} visualizations to: {args.out_dir}")


if __name__ == "__main__":
    main()
