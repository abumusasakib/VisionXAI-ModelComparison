# code/list_mappings.py
from pathlib import Path
from caption_parsers import collect_all_caption_data
import os
import argparse

# How to use:
# Default (validates images on disk):
#   python list_mappings.py
# Disable validation (for debugging):
#   python list_mappings.py --no-validate
# Combine options (skip validation and write JSON):
#   python list_mappings.py --no-validate --write-json results/mappings.json

def main():
    parser = argparse.ArgumentParser(description="List a few caption mappings discovered under data/test")
    parser.add_argument("--base", help="Base test data directory (defaults to repo/data/test)", default=None)
    # validation enabled by default; keep --no-validate for debugging
    parser.add_argument("--no-validate", dest="validate", action="store_false",
                        help="Do not validate that image files exist on disk (useful for debugging)", default=True)
    parser.add_argument("--write-json", dest="write_json", type=str, default=None,
                        help="Optional path to write the consolidated mappings as JSON (e.g. results/mappings.json)")
    args = parser.parse_args()

    # Repository root: two levels up from this script (code/ -> repo root)
    repo_root = Path(__file__).resolve().parents[1]
    base_dir = args.base or str(repo_root.joinpath("data", "test"))

    print(f"🔍 Scanning directories in {base_dir}...")
    mappings = collect_all_caption_data(base_dir=base_dir, validate_images=args.validate)

    print(f"\n📦 Total consolidated caption mappings: {len(mappings)}")
    for i, (img, caps) in enumerate(mappings.items()):
        if i >= 5:
            break
        print(img, caps[:2])
    # Optionally write mappings to JSON for downstream inference
    if args.write_json:
        out_path = Path(args.write_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Serialize as a list of objects with repo-root-relative image paths for portability
            import json as _json
            items = []
            repo_root = Path(__file__).resolve().parents[1]
            for img, caps in mappings.items():
                try:
                    rel = os.path.relpath(img, start=str(repo_root))
                except Exception:
                    # fallback to original path if relpath fails
                    rel = img
                items.append({"image": rel, "captions": caps})

            with open(out_path, "w", encoding="utf-8") as fh:
                _json.dump(items, fh, ensure_ascii=False, indent=2)
            print(f"\n💾 Wrote consolidated mappings to {out_path}")
        except Exception as e:
            print(f"⚠️ Failed to write JSON to {out_path}: {e}")


if __name__ == "__main__":
    main()