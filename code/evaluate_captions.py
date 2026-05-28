"""Evaluation script for image captioning models.

Features:
- Load model checkpoints from a directory
- Deterministic decoding (greedy or beam) with fixed seed
- Select best checkpoint by BLEU on a validation set
- Compute BLEU and CIDEr (CIDEr optional, requires pycocoevalcap)
- Produce report JSON and example captions + attention figures (if attention provided by model)

Usage examples:
 python code/evaluate_captions.py \
   --model-module code.InceptionV3.caption_parsers:build_model \
   --checkpoints-dir data/InceptionV3/ \
   --val-annotations data/InceptionV3/test/BNATURE/caption/caption.txt \
   --test-annotations data/InceptionV3/test/BNATURE/caption/test.txt \
   --vocab data/EfficientNetB4/Vocab/20250625_042759/vocab_20250625_042759 \
   --decode greedy --seed 1234 --output-dir results/eval_inception

Notes:
- The script expects a model factory function that returns (model, tokenizer, device) when called with no args
- The model should implement `generate()` that accepts a batch of images (or precomputed features) and returns dicts with keys: `caption` (str) and optionally `attention` (numpy array or list)
"""
import sys
from pathlib import Path

# Adjust sys.path to prioritize project root and script directory
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

import argparse
import json
import os
import random
import logging
from pathlib import Path
from typing import List, Dict

import numpy as np

try:
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    _HAS_NLTK = True
except Exception:
    _HAS_NLTK = False


def _import_torch():
    try:
        import torch as _torch
        return _torch
    except Exception:
        return None

torch = _import_torch()

logger = logging.getLogger("evaluate_captions")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    _torch = _import_torch()
    if _torch is not None:
        try:
            _torch.manual_seed(seed)
            if _torch.cuda.is_available():
                _torch.cuda.manual_seed_all(seed)
        except Exception:
            pass


def load_annotations(path: str) -> Dict[str, List[str]]:
    """Load simple caption files where each line is: <image_id>\t<caption> or just caption per line.

    Returns mapping image_id -> list of references (strings)
    """
    refs = {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    raw_lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    # Heuristic: if lines look like filenames (e.g., '1.jpg') then the file is a list of image ids
    fname_like = 0
    for ln in raw_lines:
        if ln.lower().endswith(('.jpg', '.jpeg', '.png')):
            fname_like += 1

    if fname_like >= max(1, len(raw_lines) // 2):
        # treat as a file-list; try to find sibling caption.txt and build refs from there
        sibling = p.parent / 'caption.txt'
        if sibling.exists():
            # build map from caption.txt
            all_refs = {}
            for ln in sibling.read_text(encoding='utf-8').splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                if "\t" in ln:
                    imgname, cap = ln.split("\t", 1)
                elif "   " in ln:
                    imgname, cap = ln.split("   ", 1)
                else:
                    parts = ln.split(None, 1)
                    if len(parts) < 2:
                        continue
                    imgname, cap = parts[0], parts[1]
                all_refs.setdefault(imgname, []).append(cap)
            for img_id in raw_lines:
                if img_id in all_refs:
                    refs[img_id] = all_refs[img_id]
                else:
                    refs[img_id] = []
            return refs
        else:
            # fallback: treat each line as an image id with empty refs
            for ln in raw_lines:
                refs.setdefault(ln, [])
            return refs

    for i, line in enumerate(raw_lines):
        # support both tab-separated and whitespace-separated formats
        parts = line.split(None, 1)
        if len(parts) == 2:
            img_id, cap = parts
        else:
            img_id = str(i)
            cap = parts[0]
        refs.setdefault(img_id, []).append(cap)
    return refs


def prepare_references_for_nltk(refs: Dict[str, List[str]]):
    """Return list of list of tokenized references in order of keys."""
    ids = list(refs.keys())
    references = []
    for k in ids:
        refs_k = [r.split() for r in refs[k]]
        references.append(refs_k)
    return ids, references


def compute_bleu(references: List[List[List[str]]], hypotheses: List[List[str]]):
    if not _HAS_NLTK:
        raise RuntimeError("nltk not available: install with `pip install nltk` to compute BLEU")
    smoothie = SmoothingFunction().method4
    # corpus_bleu expects list of list of reference tokens and list of hypothesis tokens
    score = corpus_bleu(references, hypotheses, smoothing_function=smoothie)
    return score


def try_import_cider():
    try:
        from pycocoevalcap.cider.cider import Cider

        return Cider()
    except Exception:
        return None


def evaluate_predictions(references_map: Dict[str, List[str]], preds_map: Dict[str, str]):
    ids, references = prepare_references_for_nltk(references_map)
    hypotheses = [preds_map.get(i, "").split() for i in ids]
    
    if not _HAS_NLTK:
        raise RuntimeError("nltk not available: install with `pip install nltk` to compute BLEU")
    smoothie = SmoothingFunction().method4
    
    bleu1 = corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0), smoothing_function=smoothie)
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.333, 0.333, 0.333, 0), smoothing_function=smoothie)
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

    cider_obj = try_import_cider()
    cider_score = None
    if cider_obj is not None:
        # pycocoevalcap expects dicts with id -> [str]
        gts = {i: references_map[i] for i in ids}
        res = {i: [preds_map.get(i, "")] for i in ids}
        cider_score, _ = cider_obj.compute_score(gts, res)

    return {
        "BLEU-1": float(bleu1),
        "BLEU-2": float(bleu2),
        "BLEU-3": float(bleu3),
        "BLEU-4": float(bleu4),
        "CIDEr": (float(cider_score) if cider_score is not None else None)
    }


def select_best_checkpoint(checkpoints_dir: str, model_factory, val_annotations: str, decode: str, beam_size: int, seed: int, device: str, loaded_tokenizer=None, limit: int = None):
    """Evaluate all checkpoints in directory and return path of best one by BLEU on validation set."""
    ckpt_dir = Path(checkpoints_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(checkpoints_dir)
    # collect candidate checkpoint files (common patterns)
    candidates = list(ckpt_dir.glob("**/*.ckpt")) + list(ckpt_dir.glob("**/*.index"))
    # If no matches, treat directories with `checkpoint` files as candidates
    if not candidates:
        # fallback: treat each subdir as a checkpoint candidate
        candidates = [p for p in ckpt_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise RuntimeError("No checkpoints found in " + checkpoints_dir)

    logger.info("Found %d checkpoint candidates", len(candidates))
    if len(candidates) == 1:
        logger.info("Only one candidate checkpoint found: %s. Skipping validation evaluation.", candidates[0])
        return str(candidates[0])

    val_refs = load_annotations(val_annotations)
    if limit is not None:
        val_refs = {k: val_refs[k] for k in list(val_refs.keys())[:limit]}
    best_bleu = -1.0
    best_ckpt = None
    for c in candidates:
        logger.info("Evaluating candidate %s", str(c))
        # instantiate model via factory
        model, tokenizer, device_obj = model_factory()
        if loaded_tokenizer is not None:
            tokenizer = loaded_tokenizer
            if hasattr(model, 'tokenizer'):
                model.tokenizer = loaded_tokenizer
        model.to(device_obj)
        # expect model to provide a load_checkpoint(path) hook
        if hasattr(model, "load_checkpoint"):
            try:
                model.load_checkpoint(str(c))
            except Exception:
                logger.warning("Model.load_checkpoint failed for %s; trying torch.load state_dict", c)
                if torch is not None:
                    try:
                        sd = torch.load(str(c), map_location=device_obj)
                        model.load_state_dict(sd)
                    except Exception:
                        logger.exception("Failed loading checkpoint %s", c)
                        continue
                else:
                    logger.warning("Torch is not available to load PyTorch checkpoint %s", c)
                    continue
        else:
            # try generic torch load
            if torch is not None:
                try:
                    sd = torch.load(str(c), map_location=device_obj)
                    model.load_state_dict(sd)
                except Exception:
                    logger.warning("Could not load checkpoint %s into model", c)
                    continue
            else:
                logger.warning("Torch is not available to load PyTorch checkpoint %s", c)
                continue

        # deterministic decode on validation
        set_seed(seed)
        preds = {}
        for img_id in val_refs.keys():
            # model.generate should accept image id or return caption; this is a flexible hook
            try:
                out = model.generate(img_id=img_id, decode=decode, beam_size=beam_size, tokenizer=tokenizer)
            except TypeError:
                out = model.generate(img_id=img_id)
            if isinstance(out, dict):
                caption = out.get("caption", "")
            else:
                caption = str(out)
            preds[img_id] = caption

        scores = evaluate_predictions(val_refs, preds)
        logger.info("Candidate %s -> BLEU-4=%.4f", str(c), scores["BLEU-4"])
        if scores["BLEU-4"] > best_bleu:
            best_bleu = scores["BLEU-4"]
            best_ckpt = c

    if best_ckpt is None:
        raise RuntimeError("No valid checkpoint could be evaluated")
    logger.info("Selected best checkpoint %s (BLEU-4=%.4f)", str(best_ckpt), best_bleu)
    return str(best_ckpt)


def run_evaluation(checkpoint: str, model_factory, annotations: str, decode: str, beam_size: int, seed: int, output_dir: str, loaded_tokenizer=None, limit: int = None):
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    refs = load_annotations(annotations)
    if limit is not None:
        refs = {k: refs[k] for k in list(refs.keys())[:limit]}
    model, tokenizer, device_obj = model_factory()
    if loaded_tokenizer is not None:
        tokenizer = loaded_tokenizer
        if hasattr(model, 'tokenizer'):
            model.tokenizer = loaded_tokenizer
    model.to(device_obj)
    if hasattr(model, "load_checkpoint"):
        model.load_checkpoint(checkpoint)
    else:
        if torch is not None:
            try:
                sd = torch.load(checkpoint, map_location=device_obj)
                model.load_state_dict(sd)
            except Exception:
                logger.warning("Could not load checkpoint with conventional method")
        else:
            logger.warning("Could not load checkpoint: torch is not available")

    set_seed(seed)
    preds = {}
    attn_examples = []
    for img_id in refs.keys():
        try:
            out = model.generate(img_id=img_id, decode=decode, beam_size=beam_size, tokenizer=tokenizer)
        except TypeError:
            out = model.generate(img_id=img_id)
        if isinstance(out, dict):
            caption = out.get("caption", "")
            attention = out.get("attention")
        else:
            caption = str(out)
            attention = None
        preds[img_id] = caption
        if attention is not None and len(attn_examples) < 10:
            attn_examples.append({"id": img_id, "caption": caption, "attention": np.array(attention).tolist()})

    scores = evaluate_predictions(refs, preds)
    report = {"checkpoint": checkpoint, "scores": scores}
    (outdir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (outdir / "predictions.json").write_text(json.dumps(preds, indent=2, ensure_ascii=False), encoding="utf-8")
    (outdir / "attn_examples.json").write_text(json.dumps(attn_examples, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved report and predictions to %s", str(outdir))


def import_model_factory(spec: str):
    """Import a model factory from a module path like `module.sub:callable`"""
    if ":" not in spec:
        raise ValueError("--model-module must be MODULE:callable")
    modpath, fn = spec.split(":", 1)
    import importlib

    m = importlib.import_module(modpath)
    if not hasattr(m, fn):
        raise AttributeError(f"Module {modpath} has no attribute {fn}")
    return getattr(m, fn)


class ListTokenizerWrapper:
    def __init__(self, vocab_list):
        self.vocab_list = vocab_list
        self.word_index = {word: idx for idx, word in enumerate(vocab_list) if word}
        self.index_word = {idx: word for idx, word in enumerate(vocab_list)}

    def __len__(self):
        return len(self.vocab_list)


def main():
    p = argparse.ArgumentParser(description="Evaluate image captioning models deterministically and select best checkpoint by BLEU")
    p.add_argument("--model-module", required=True, help="Module path and factory callable, e.g. code.InceptionV3.caption_parsers:build_model")
    p.add_argument("--checkpoints-dir", required=True)
    p.add_argument("--val-annotations", required=True)
    p.add_argument("--test-annotations", required=True)
    p.add_argument("--vocab", help="Path to shared vocabulary (ensure same tokenizer/vocab for fair comparison)")
    p.add_argument("--decode", choices=["greedy", "beam"], default="greedy")
    p.add_argument("--beam-size", type=int, default=3)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default=None, help="Device to run on (e.g. cpu or cuda). If not set the script will try to use CUDA when available.")
    p.add_argument("--output-dir", default="results/eval")
    p.add_argument("--limit", type=int, default=None, help="Limit number of images evaluated (for quick testing/development)")
    args = p.parse_args()
    # Resolve default device lazily to avoid importing torch when printing --help
    if args.device is None:
        _torch = _import_torch()
        if _torch is not None and _torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"

    model_factory = import_model_factory(args.model_module)

    # Load custom tokenizer/vocab if specified
    loaded_tokenizer = None
    if args.vocab and os.path.exists(args.vocab):
        import pickle
        try:
            with open(args.vocab, "rb") as f:
                loaded_tokenizer = pickle.load(f)
            if isinstance(loaded_tokenizer, list):
                logger.info("Loaded vocabulary list from %s; wrapping in ListTokenizerWrapper", args.vocab)
                loaded_tokenizer = ListTokenizerWrapper(loaded_tokenizer)
            else:
                logger.info("Successfully loaded vocabulary/tokenizer from %s", args.vocab)
        except Exception as e:
            logger.warning("Failed to load tokenizer from %s: %s", args.vocab, e)

    # Select best checkpoint by BLEU on validation
    best = select_best_checkpoint(
        args.checkpoints_dir, model_factory, args.val_annotations,
        args.decode, args.beam_size, args.seed, args.device,
        loaded_tokenizer=loaded_tokenizer, limit=args.limit
    )

    # Evaluate on test set
    run_evaluation(
        best, model_factory, args.test_annotations,
        args.decode, args.beam_size, args.seed, args.output_dir,
        loaded_tokenizer=loaded_tokenizer, limit=args.limit
    )


if __name__ == "__main__":
    main()
