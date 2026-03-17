#!/usr/bin/env bash
set -euo pipefail

# Run from inside Docker container at /workspace.
DATASET_ROOT="${DATASET_ROOT:-/data/IDD_FGVD}"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/results}"
PRED_ROOT="${PRED_ROOT:-$RESULTS_ROOT/predictions}"
SPLIT_EVAL="${SPLIT_EVAL:-test}"
# Set OVERWRITE=1 to force re-annotate all images; default is resume mode.
OVERWRITE="${OVERWRITE:-0}"

# Default model list. Override with: MODELS="groundingdino yoloworld"
MODELS="${MODELS:-groundingdino groundedsam yoloworld owlvit}"

backend_module() {
  case "$1" in
    groundingdino) echo "autodistill_grounding_dino" ;;
    groundedsam) echo "autodistill_grounded_sam" ;;
    yoloworld) echo "autodistill_yolo_world" ;;
    owlvit) echo "autodistill_owl_vit" ;;
    *) echo "" ;;
  esac
}

mkdir -p "$PRED_ROOT"

echo "[INFO] Dataset root : $DATASET_ROOT"
echo "[INFO] Results root : $RESULTS_ROOT"
echo "[INFO] Models       : $MODELS"

# ── Pre-flight: verify every requested backend is importable ──────────────────
echo ""
echo "[CHECK] Verifying backends..."
missing=0
for model in $MODELS; do
  module_name="$(backend_module "$model")"
  if [[ -z "$module_name" ]]; then
    echo "  [WARN]  $model  → no module mapping defined in backend_module()"
    continue
  fi
  if python - <<PY >/dev/null 2>&1
import importlib, sys
importlib.import_module("$module_name")
# Also verify auto_annotate can build the model (catches stub packages)
sys.path.insert(0, '/workspace')
import auto_annotate
from autodistill.detection import CaptionOntology
ontology = CaptionOntology({"car": "car"})
auto_annotate.build_model("$model", ontology, 0.35, 0.25)
PY
  then
    echo "  [OK]     $model  ($module_name)"
  else
    echo "  [MISSING] $model  ($module_name)  ← not installed!"
    missing=$((missing + 1))
  fi
done

if [[ $missing -gt 0 ]]; then
  echo ""
  echo "[ERROR] $missing backend(s) missing. Install them first, e.g.:"
  echo "  pip install autodistill-grounding-dino autodistill-grounded-sam \\"
  echo "              autodistill-yolo-world autodistill-owl-vit"
  exit 1
fi
echo "[CHECK] All backends OK. Starting sequential annotation run..."
echo ""
# ─────────────────────────────────────────────────────────────────────────────

for model in $MODELS; do
  out_dir="$PRED_ROOT/$model"
  log_path="$RESULTS_ROOT/${model}_annotate.log"
  module_name="$(backend_module "$model")"

  echo "[RUN] ── Model: $model ──────────────────────────────────────────────"
  echo "[RUN] Output dir: $out_dir"

  # Only wipe existing predictions when OVERWRITE=1 is explicitly requested.
  if [[ "$OVERWRITE" == "1" ]]; then
    echo "[RUN] OVERWRITE=1: removing existing predictions for $model"
    rm -rf "$out_dir"
  fi
  mkdir -p "$out_dir"

  OVERWRITE_FLAG=""
  [[ "$OVERWRITE" == "1" ]] && OVERWRITE_FLAG="--overwrite"

  PYTHONUNBUFFERED=1 python /workspace/auto_annotate.py \
    --model "$model" \
    --dataset-root "$DATASET_ROOT" \
    --output-root "$out_dir" \
    $OVERWRITE_FLAG \
    --save-confidence \
    > "$log_path" 2>&1

  echo "[DONE] $model annotations complete. Log: $log_path"
done

echo ""
echo "[EVAL] Running aggregate evaluation..."
python /workspace/evaluate_models.py \
  --dataset-root "$DATASET_ROOT" \
  --split "$SPLIT_EVAL" \
  --predictions-root "$PRED_ROOT" \
  --results-root "$RESULTS_ROOT"

echo "[DONE] Metrics written under: $RESULTS_ROOT"
