import csv
import zipfile
import xml.etree.ElementTree as ET
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import json

# Attempt to import cElementTree for faster XML parsing, fall back to ElementTree
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


# ============================================================
# 🔹 Abstract Base Class
# ============================================================
class CaptionParser(ABC):
    """
    Abstract Base Class for caption parsers.

    Defines the common interface for extracting image-caption mappings
    from different file formats (e.g., XLSX, CSV, JSON, TXT).
    """

    @abstractmethod
    def extract(self, file_path: str, images_path: str, validate_images: bool) -> Dict[str, List[str]]:
        """
        Abstract method to extract image-caption mappings from a given file.

        Args:
            file_path (str): The path to the data file (e.g., .xlsx, .csv, .json, .txt).
            images_path (str): The base directory where image files are located.
            validate_images (bool): If True, checks if the image file exists on disk
                                    before including its captions in the output.

        Returns:
            Dict[str, List[str]]: A dictionary where keys are absolute image paths
                                  and values are lists of formatted captions.
        """
        pass


# ============================================================
# 🔹 JSON Caption Parser
# ============================================================
class JSONCaptionParser(CaptionParser):
    """
    Parser for JSON caption files.
    Expected format:
    [
        {"filename": "image1.jpg", "caption": ["caption1", "caption2"]},
        {"filename": "image2.jpg", "caption": ["caption3"]}
    ]
    """

    def extract(self, file_path: str, images_path: str = "", validate_images: bool = True) -> Dict[str, List[str]]:
        caption_mapping: Dict[str, List[str]] = {}
        print(f"\n➡️  Parsing JSON: {os.path.basename(file_path)}...")
        try:
            with open(file_path, encoding="utf8") as caption_file:
                caption_data = json.load(caption_file)
                if not isinstance(caption_data, list):
                    print(f"Warning: JSON file {file_path} does not contain a list at its root. Skipping.")
                    return caption_mapping

                for idx, item in enumerate(caption_data):
                    if idx % 1000 == 0:
                        print(f"\r  → Processing JSON item {idx}...", end="", flush=True)

                    if not isinstance(item, dict) or 'filename' not in item or 'caption' not in item:
                        continue

                    img_name_abs = os.path.join(images_path, item['filename'].strip())
                    captions_raw = item['caption']
                    if not isinstance(captions_raw, list):
                        captions_raw = [captions_raw]

                    formatted = [str(c).strip() + " " for c in captions_raw if c]
                    if not validate_images or Path(img_name_abs).exists():
                        if formatted:
                            caption_mapping[img_name_abs] = formatted

            print(f"\r  → Finished parsing {os.path.basename(file_path)}. Total valid entries: {len(caption_mapping)}.", flush=True)
        except json.JSONDecodeError as e:
            print(f"\nError: Invalid JSON format in {file_path}: {e}")
        except Exception as e:
            print(f"\nError reading JSON file {file_path}: {e}")

        return caption_mapping


# ============================================================
# 🔹 TXT Caption Parser
# ============================================================
class TXTCaptionParser(CaptionParser):
    """
    Parser for TXT caption files.
    Supports two common formats:
        1️⃣ Triple-space separated:
            image1.jpg   একটি ছেলে মাঠে খেলছে।
        2️⃣ Hash-separated:
            1.png #একটি ছেলে ব্রিজের রেলিংয়ে দাড়িয়ে আছে।
    """

    def extract(
        self,
        file_path: str,
        images_path: str = "",
        validate_images: bool = True
    ) -> Dict[str, List[str]]:
        caption_mapping: Dict[str, List[str]] = {}
        print(f"\n➡️  Parsing TXT: {os.path.basename(file_path)}...")

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                for line_num, line in enumerate(file, start=1):
                    line = line.strip()
                    if not line:
                        continue  # skip blank lines

                    # --- Detect delimiter automatically ---
                    if "   " in line:
                        parts = line.split("   ", 1)
                    elif " #" in line:
                        parts = line.split(" #", 1)
                    elif "#" in line:
                        parts = line.split("#", 1)
                    else:
                        continue  # skip malformed lines

                    if len(parts) < 2:
                        continue

                    img_name = parts[0].strip()
                    caption = parts[1].strip()

                    if not img_name or not caption:
                        continue

                    # Build absolute image path
                    img_path_abs = os.path.join(images_path, img_name)

                    # Validate existence if required
                    if not validate_images or Path(img_path_abs).exists():
                        caption_mapping.setdefault(img_path_abs, []).append(caption)

                    if line_num % 1000 == 0:
                        print(f"\r  → Processed {line_num} lines...", end="", flush=True)

            print(f"\r  → Finished parsing {os.path.basename(file_path)}. Total valid entries: {len(caption_mapping)}.", flush=True)

        except FileNotFoundError:
            print(f"❌ File not found: {file_path}")
        except Exception as e:
            print(f"⚠️ Error reading TXT file {file_path}: {e}")

        return caption_mapping

# ============================================================
# 🔹 Unified Caption Collector
# ============================================================
def collect_all_caption_data(base_dir: str, validate_images: bool = True) -> Dict[str, List[str]]:
    """
    Walks through a base directory to find and extract caption data from dataset files.
    Supports both JSON and TXT formats and handles BNATURE and BNLIT dataset structures.
    """
    all_captions: Dict[str, List[str]] = {}
    json_parser = JSONCaptionParser()
    txt_parser = TXTCaptionParser()

    print(f"🔍 Scanning directories in {base_dir}...")

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            lower_file = file.lower()
            file_path = os.path.join(root, file)
            captions: Dict[str, List[str]] = {}

            # ====================================================
            # 🔸 1. Handle BNATURE test captions
            #    - 'test.txt' here contains only image filenames (one per line).
            #    - Full captions live in 'caption/caption.txt'. Images are in 'Pictures/'.
            # ====================================================
            if lower_file == "test.txt":
                # Assume we were called with base_dir pointing to data/test
                # Derive BNATURE root and expected files
                # file_path is .../BNATURE/caption/test.txt (common layout)
                bnature_caption_dir = os.path.dirname(file_path)
                bnature_root = os.path.normpath(os.path.join(bnature_caption_dir, ".."))
                pictures_dir = os.path.join(bnature_root, "Pictures")
                captions_file = os.path.join(bnature_caption_dir, "caption.txt")

                # Read test image list (one filename per line)
                test_image_names: List[str] = []
                try:
                    with open(file_path, "r", encoding="utf-8") as tf:
                        for ln in tf:
                            ln = ln.strip()
                            if ln:
                                test_image_names.append(ln)
                except Exception:
                    test_image_names = []

                # Read the caption file and build a mapping by filename
                cap_map: Dict[str, List[str]] = {}
                if os.path.exists(captions_file):
                    try:
                        with open(captions_file, "r", encoding="utf-8") as cf:
                            for ln in cf:
                                ln = ln.strip()
                                if not ln:
                                    continue
                                # support multiple delimiters used in this dataset
                                if "   " in ln:
                                    imgname, cap = ln.split("   ", 1)
                                elif " #" in ln:
                                    imgname, cap = ln.split(" #", 1)
                                elif "#" in ln:
                                    imgname, cap = ln.split("#", 1)
                                else:
                                    parts = ln.split(None, 1)
                                    if len(parts) < 2:
                                        continue
                                    imgname, cap = parts[0], parts[1]
                                imgname = imgname.strip()
                                cap = cap.strip()
                                if imgname and cap:
                                    cap_map.setdefault(imgname, []).append(cap)
                    except Exception:
                        pass

                # Build captions only for images listed in test.txt and present in cap_map
                for imgname in test_image_names:
                    caps = cap_map.get(imgname, [])
                    if not caps:
                        continue
                    img_path = os.path.join(pictures_dir, imgname)
                    if (not validate_images) or os.path.exists(img_path):
                        captions[os.path.abspath(img_path)] = caps

            # ====================================================
            # 🔸 2. Handle BNLIT test captions
            #    - Use Test-Annotation... file to extract image->caption lines
            #    - Images live in the preprocessed/resized folder under the BNLIT directory
            # ====================================================
            # Only treat the specific test-annotation file for BNLIT as the source of test captions.
            # Restrict to filenames that clearly indicate the test annotations only.
            elif ("test-annotation" in lower_file and "bnlit" in lower_file) or lower_file.startswith(
                "test-annotation-bangla natural language image to text"):
                # Derive BNLIT root and resized images folder
                bnlit_root = os.path.dirname(file_path)
                # The resized images folder usually lives inside the BNLIT folder itself.
                resized_folder = "Bangla Natural Language Image to Text (BNLIT)-Preprocessing and Resizing Dataset-resized-500_375"
                images_dir = os.path.join(bnlit_root, resized_folder)
                # Fallback: if the expected resized folder doesn't exist, try the BNLIT folder directly
                # (the annotations file often sits alongside the resized images). If that also fails,
                # fall back to the parent directory as a last resort.
                if not os.path.isdir(images_dir):
                    if os.path.isdir(bnlit_root):
                        images_dir = bnlit_root
                    else:
                        images_dir = os.path.normpath(os.path.join(bnlit_root, ".."))

                # Parse annotation lines like '1.png #caption'
                try:
                    with open(file_path, "r", encoding="utf-8") as af:
                        for ln in af:
                            ln = ln.strip()
                            if not ln:
                                continue
                            if " #" in ln:
                                imgname, cap = ln.split(" #", 1)
                            elif "#" in ln:
                                imgname, cap = ln.split("#", 1)
                            else:
                                parts = ln.split(None, 1)
                                if len(parts) < 2:
                                    continue
                                imgname, cap = parts[0], parts[1]
                            imgname = imgname.strip()
                            cap = cap.strip()
                            img_path = os.path.join(images_dir, imgname)
                            if (not validate_images) or os.path.exists(img_path):
                                captions.setdefault(os.path.abspath(img_path), []).append(cap)
                except Exception:
                    pass

            # ====================================================
            # 🔸 3. Skip unrelated TXT/JSON files
            # ====================================================
            elif lower_file.endswith(".txt") or lower_file.endswith(".json"):
                # Ignore training/validation files and unrelated datasets
                if any(skip_word in lower_file for skip_word in ["train", "validation", "val", "caption"]):
                    continue

            # ====================================================
            # 🔸 Merge valid captions
            # ====================================================
            if captions:
                all_captions.update(captions)
                print(f"✅ Added {len(captions)} captions from {file}")

    print(f"\n📦 Total consolidated caption mappings: {len(all_captions)}")
    return all_captions

