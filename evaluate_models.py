import argparse
import csv
import glob
import os
import xml.etree.ElementTree as ET
from collections import defaultdict


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


def load_yolo_boxes(label_path, has_conf=False):
    boxes = []
    if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
        return boxes

    with open(label_path, "r", encoding="utf-8") as f:
        for raw in f:
            parts = raw.strip().split()
            if len(parts) < 5:
                continue

            cls_id = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:5])
            conf = 1.0
            if has_conf and len(parts) >= 6:
                conf = float(parts[5])

            x1 = cx - (w / 2.0)
            y1 = cy - (h / 2.0)
            x2 = cx + (w / 2.0)
            y2 = cy + (h / 2.0)

            boxes.append(
                {
                    "class_id": cls_id,
                    "xyxy": (x1, y1, x2, y2),
                    "conf": conf,
                }
            )

    return boxes


def normalize_gt_class(raw_name):
    if not raw_name:
        return None

    prefix = raw_name.strip().split("_")[0].strip().lower()
    return GT_CLASS_ALIASES.get(prefix)


def load_voc_xml_boxes(label_path, class_to_id):
    boxes = []
    if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
        return boxes

    tree = ET.parse(label_path)
    root = tree.getroot()

    width = float(root.findtext("size/width", default="0"))
    height = float(root.findtext("size/height", default="0"))
    if width <= 0 or height <= 0:
        return boxes

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

        x1 = xmin / width
        y1 = ymin / height
        x2 = xmax / width
        y2 = ymax / height

        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append(
            {
                "class_id": class_to_id[class_name],
                "xyxy": (x1, y1, x2, y2),
                "conf": 1.0,
            }
        )

    return boxes


def detect_ground_truth_format(gt_dir):
    if glob.glob(os.path.join(gt_dir, "*.xml")):
        return "vocxml"
    if glob.glob(os.path.join(gt_dir, "*.txt")):
        return "yolo"
    raise FileNotFoundError(f"No supported ground-truth files found in: {gt_dir}")


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    denom = area_a + area_b - inter_area
    if denom <= 0.0:
        return 0.0

    return inter_area / denom


def compute_ap(recall_points, precision_points):
    # COCO/VOC-style interpolation using monotonic precision envelope.
    mrec = [0.0] + recall_points + [1.0]
    mpre = [0.0] + precision_points + [0.0]

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]

    return ap


def evaluate_model(model_name, pred_dir, gt_dir, image_dir, class_names, iou_threshold=0.5, gt_format="auto"):
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    if not image_paths:
        image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))

    image_ids = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    if gt_format == "auto":
        gt_format = detect_ground_truth_format(gt_dir)

    per_class_gt_count = defaultdict(int)
    per_class_preds = defaultdict(list)

    tp_ious = []

    for image_id in image_ids:
        pred_path = os.path.join(pred_dir, image_id + ".txt")

        if gt_format == "vocxml":
            gt_path = os.path.join(gt_dir, image_id + ".xml")
            gt_boxes = load_voc_xml_boxes(gt_path, class_to_id)
        else:
            gt_path = os.path.join(gt_dir, image_id + ".txt")
            gt_boxes = load_yolo_boxes(gt_path, has_conf=False)

        pred_boxes = load_yolo_boxes(pred_path, has_conf=True)

        gt_by_class = defaultdict(list)
        for g in gt_boxes:
            gt_by_class[g["class_id"]].append(g)
            per_class_gt_count[g["class_id"]] += 1

        pred_by_class = defaultdict(list)
        for p in pred_boxes:
            pred_by_class[p["class_id"]].append(p)

        class_ids = set(gt_by_class.keys()) | set(pred_by_class.keys())

        for cid in class_ids:
            gt_list = gt_by_class.get(cid, [])
            pred_list = pred_by_class.get(cid, [])

            matched_gt = [False] * len(gt_list)
            pred_list_sorted = sorted(pred_list, key=lambda x: x["conf"], reverse=True)

            for pred in pred_list_sorted:
                best_iou = 0.0
                best_idx = -1

                for idx, gt in enumerate(gt_list):
                    if matched_gt[idx]:
                        continue
                    iou = iou_xyxy(pred["xyxy"], gt["xyxy"])
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx

                is_tp = best_idx >= 0 and best_iou >= iou_threshold
                if is_tp:
                    matched_gt[best_idx] = True
                    tp_ious.append(best_iou)

                per_class_preds[cid].append(
                    {
                        "conf": pred["conf"],
                        "tp": 1 if is_tp else 0,
                    }
                )

    class_rows = []
    ap_values = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for cid in range(len(class_names)):
        preds = sorted(per_class_preds[cid], key=lambda x: x["conf"], reverse=True)
        gt_count = per_class_gt_count[cid]

        tp_running = 0
        fp_running = 0
        recall_points = []
        precision_points = []

        for pred in preds:
            if pred["tp"] == 1:
                tp_running += 1
            else:
                fp_running += 1

            precision = tp_running / max(tp_running + fp_running, 1)
            recall = tp_running / max(gt_count, 1)
            precision_points.append(precision)
            recall_points.append(recall)

        tp = tp_running
        fp = fp_running
        fn = max(gt_count - tp, 0)

        precision_final = tp / max(tp + fp, 1)
        recall_final = tp / max(gt_count, 1)
        f1 = 0.0
        if precision_final + recall_final > 0:
            f1 = 2 * precision_final * recall_final / (precision_final + recall_final)

        ap50 = compute_ap(recall_points, precision_points) if gt_count > 0 else 0.0
        if gt_count > 0:
            ap_values.append(ap50)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        class_rows.append(
            {
                "model": model_name,
                "class_id": cid,
                "class_name": class_names[cid],
                "gt": gt_count,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(precision_final, 6),
                "recall": round(recall_final, 6),
                "f1": round(f1, 6),
                "ap50": round(ap50, 6),
            }
        )

    overall_precision = total_tp / max(total_tp + total_fp, 1)
    overall_recall = total_tp / max(total_tp + total_fn, 1)
    overall_f1 = 0.0
    if overall_precision + overall_recall > 0:
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall)

    metrics = {
        "model": model_name,
        "images": len(image_ids),
        "gt_boxes": sum(per_class_gt_count.values()),
        "pred_boxes": sum(len(v) for v in per_class_preds.values()),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": round(overall_precision, 6),
        "recall": round(overall_recall, 6),
        "f1": round(overall_f1, 6),
        "mean_iou_tp": round(sum(tp_ious) / max(len(tp_ious), 1), 6),
        "map50": round(sum(ap_values) / max(len(ap_values), 1), 6),
    }

    return metrics, class_rows


def find_model_prediction_dirs(predictions_root):
    candidates = []
    for path in sorted(glob.glob(os.path.join(predictions_root, "*"))):
        if os.path.isdir(path):
            candidates.append(path)
    return candidates


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple detection models against YOLO ground truth.")
    parser.add_argument("--dataset-root", default="/data/IDD_FGVD")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--predictions-root", default="/workspace/results/predictions")
    parser.add_argument("--ground-truth-subdir", default="annos")
    parser.add_argument("--images-subdir", default="images")
    parser.add_argument("--results-root", default="/workspace/results")
    parser.add_argument("--iou", type=float, default=0.5)
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

    gt_dir = os.path.join(args.dataset_root, args.split, args.ground_truth_subdir)
    image_dir = os.path.join(args.dataset_root, args.split, args.images_subdir)

    model_dirs = find_model_prediction_dirs(args.predictions_root)
    if not model_dirs:
        raise FileNotFoundError(
            f"No model prediction directories found in: {args.predictions_root}. "
            "Expected structure: results/predictions/<model_name>/<split>/*.txt"
        )

    summary_rows = []
    class_rows_all = []

    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)
        pred_dir = os.path.join(model_dir, args.split)
        if not os.path.isdir(pred_dir):
            print(f"Skipping {model_name}: missing split folder {pred_dir}")
            continue

        metrics, class_rows = evaluate_model(
            model_name=model_name,
            pred_dir=pred_dir,
            gt_dir=gt_dir,
            image_dir=image_dir,
            class_names=args.classes,
            iou_threshold=args.iou,
            gt_format=args.ground_truth_format,
        )

        summary_rows.append(metrics)
        class_rows_all.extend(class_rows)

        print(
            f"[{model_name}] "
            f"P={metrics['precision']:.4f} "
            f"R={metrics['recall']:.4f} "
            f"F1={metrics['f1']:.4f} "
            f"mAP50={metrics['map50']:.4f} "
            f"IoU(TP)={metrics['mean_iou_tp']:.4f}"
        )

    if not summary_rows:
        raise RuntimeError("No valid model predictions found for evaluation.")

    summary_path = os.path.join(args.results_root, f"metrics_summary_{args.split}.csv")
    class_path = os.path.join(args.results_root, f"metrics_per_class_{args.split}.csv")

    write_csv(
        summary_path,
        summary_rows,
        [
            "model",
            "images",
            "gt_boxes",
            "pred_boxes",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "mean_iou_tp",
            "map50",
        ],
    )

    write_csv(
        class_path,
        class_rows_all,
        [
            "model",
            "class_id",
            "class_name",
            "gt",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "ap50",
        ],
    )

    print(f"\nSaved summary metrics: {summary_path}")
    print(f"Saved per-class metrics: {class_path}")


if __name__ == "__main__":
    main()
