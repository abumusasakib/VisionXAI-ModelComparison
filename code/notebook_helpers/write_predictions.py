"""Helpers to standardize predictions.json from notebook globals or results files.

Intended usage inside a notebook (one cell):

from code.notebook_helpers.write_predictions import write_predictions_from_globals
write_predictions_from_globals(globals(), output_dir='/results', checkpoint_name='ckpt-4')

The function will search for common in-notebook variables (`preds`, `predictions`,
`captions`, `img_name_vector`, `image_paths`) and assemble a mapping image_id -> caption.
If none are found, it will try to read existing `/results/captions.json` or
`/results/predictions.json` and copy them into the standardized file.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict


PREFERRED_KEYS = [
    "preds",
    "predictions",
    "predictions_map",
    "preds_map",
    "generated_captions",
    "captions",
    "preds_json",
]


def is_mapping(obj: Any) -> bool:
    return isinstance(obj, dict) and all(isinstance(k, (str,)) for k in obj.keys())


def write_predictions_from_globals(g: Dict[str, Any], output_dir: str = "/results", checkpoint_name: str = None) -> str:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    mapping = None

    # 1) look for a mapping variable
    for k in PREFERRED_KEYS:
        if k in g and is_mapping(g[k]):
            mapping = g[k]
            break

    # 2) if not found, try to pair image list + captions list
    if mapping is None:
        img_keys = None
        for cand in ("img_name_vector", "image_paths", "test_image_paths", "img_paths"):
            if cand in g and isinstance(g[cand], (list, tuple)):
                img_keys = list(g[cand])
                break

        captions_list = None
        for cand in ("preds_list", "generated_captions_list", "predicted_captions", "test_captions", "captions_list"):
            if cand in g and isinstance(g[cand], (list, tuple)):
                captions_list = list(g[cand])
                break

        if img_keys is not None and captions_list is not None and len(img_keys) == len(captions_list):
            mapping = {str(img): str(cap) for img, cap in zip(img_keys, captions_list)}

    # 3) fallback: try to read existing results files
    if mapping is None:
        for fname in ("/results/predictions.json", "/results/captions.json", "/results/predictions_map.json"):
            p = Path(fname)
            if p.exists():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    if is_mapping(data):
                        mapping = data
                        break
                except Exception:
                    continue

    if mapping is None:
        # last resort: empty mapping
        mapping = {}

    # If checkpoint_name provided, write into a subfolder
    target_dir = outdir
    if checkpoint_name:
        target_dir = outdir / str(checkpoint_name)
        target_dir.mkdir(parents=True, exist_ok=True)

    out_path = target_dir / "predictions.json"
    out_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Wrote standardized predictions to: {out_path}")
    return str(out_path)
