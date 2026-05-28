import json
from pathlib import Path

nb_path = Path(__file__).resolve().parent / 'bangla_image_caption_testing.ipynb'
print('Patching', nb_path)
nb = json.loads(nb_path.read_text(encoding='utf-8'))
cell_id = '#VSC-a62d84da'
new_source = [
    "# Build flat caption list and matching image list",
    "from typing import List",
    "",
    "train_captions: List[str] = []",
    "img_name_vector: List[str] = []",
    "",
    "# Ensure we have an image path list to iterate. If the training pipeline set `train_image_paths`",
    "# earlier, use it; otherwise build from the caption_mapping we just created.",
    "if 'train_image_paths' not in globals():",
    "    train_image_paths = sorted(list(caption_mapping.keys()))",
    "",
    "# Apply TRAIN_LIMIT when it's an integer > 0 (keeps behavior compatible with earlier notebook cells)",
    "if 'TRAIN_LIMIT' in globals() and isinstance(TRAIN_LIMIT, int) and TRAIN_LIMIT > 0:",
    "    train_image_paths = train_image_paths[:TRAIN_LIMIT]",
    "",
    "for image_path in train_image_paths:",
    "    captions = caption_mapping.get(image_path, [])",
    "    if not captions:",
    "        continue",
    "    train_captions.extend([f\"<start> {cap} <end>\" for cap in captions])",
    "    img_name_vector.extend([image_path] * len(captions))",
]

found = False
for i, cell in enumerate(nb.get('cells', [])):
    # Notebook cell id field may be stored under 'id' or under metadata; print to help debugging
    cid = cell.get('id') or cell.get('metadata', {}).get('id')
    print(i, 'cell id ->', repr(cid))
    if cid == cell_id:
        cell['source'] = new_source
        found = True
        print('Replaced cell at index', i)
        break

    # Additionally try to match by searching for the old signature line
    src = cell.get('source') or []
    if any('for image_path in train_image_paths' in (s if isinstance(s, str) else '') for s in src):
        print('Found matching source at index', i, 'replacing...')
        cell['source'] = new_source
        found = True
        break

if not found:
    raise SystemExit('Cell id not found')

nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding='utf-8')
print('Patched successfully')
