#!/usr/bin/env python3
"""
scripts/download_bottle.py
==========================
Safe, selective downloader for the MVTec Anomaly Detection 'bottle' category.
Fetches exclusively the 157 MB 'bottle' category files from the verified
MVTec AD CC BY-NC-SA 4.0 mirror (foersben/mvtec-ad), avoiding the monolithic
5.0 GB full archive to conserve local disk space.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import sys
import urllib.request

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.config import DatasetConfig
from src.data.validator import MVTecValidator

HF_API_BASE = "https://huggingface.co/api/datasets/foersben/mvtec-ad/tree/main"
RAW_BASE_URL = "https://huggingface.co/datasets/foersben/mvtec-ad/resolve/main"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Industrial-AI-Inspector"


def fetch_remote_directory_tree(subpath: str) -> list[str]:
    """Recursively list all file paths under subpath using the Hugging Face API."""
    url = f"{HF_API_BASE}/{subpath}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            items = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to query remote directory '{subpath}': {exc}") from exc

    file_paths = []
    for item in items:
        item_path = item["path"]
        item_type = item.get("type", "file")
        if item_type == "directory":
            file_paths.extend(fetch_remote_directory_tree(item_path))
        else:
            file_paths.append(item_path)
    return file_paths


def download_single_file(remote_rel_path: str, target_file: Path) -> Path:
    """Download a single file from remote repository to target location."""
    if target_file.is_file() and target_file.stat().st_size > 0:
        return target_file

    target_file.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target_file.with_suffix(f"{target_file.suffix}.tmp")

    file_url = f"{RAW_BASE_URL}/{remote_rel_path}"
    req = urllib.request.Request(file_url, headers={"User-Agent": USER_AGENT})

    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp, open(temp_target, "wb") as f_out:
                shutil.copyfileobj(resp, f_out)
            temp_target.replace(target_file)
            return target_file
        except Exception as exc:
            if temp_target.exists():
                temp_target.unlink()
            if attempt == 3:
                raise RuntimeError(f"Failed downloading '{remote_rel_path}' after 3 attempts: {exc}") from exc
    return target_file


def download_bottle_category(dest_dir: Path, max_workers: int = 8) -> bool:
    """Download and verify the complete MVTec AD bottle category."""
    print("=" * 65)
    print(" MVTec AD Selective Downloader: 'bottle' Category")
    print("=" * 65)
    print(f"[*] Destination Directory: {dest_dir}")
    print(f"[*] License:               CC BY-NC-SA 4.0 (Non-commercial research)")
    print("-" * 65)

    # 1. Pre-flight disk space safety check
    usage = shutil.disk_usage(dest_dir.anchor or "C:\\")
    free_gb = usage.free / (1024 ** 3)
    print(f"[*] Available Disk Space on {dest_dir.anchor}: {free_gb:.2f} GB")
    if free_gb < 1.0:
        print("[!] Critical Error: Less than 1.0 GB disk space remaining.")
        return False

    # 2. Query directory index from verified mirror
    print("[*] Querying remote file manifest for 'bottle'...")
    try:
        all_files = fetch_remote_directory_tree("bottle")
        print(f"    [+] Discovered {len(all_files)} files in category (images + masks + docs).")
    except Exception as exc:
        print(f"[!] Manifest discovery failed: {exc}")
        return False

    # 3. Concurrent selective download
    print(f"[*] Downloading {len(all_files)} files using {max_workers} concurrent threads...")
    completed = 0
    total = len(all_files)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for rel_path in all_files:
            # Strip leading 'bottle/' to align with dest_dir
            sub_path = Path(rel_path).relative_to("bottle")
            target_path = dest_dir / sub_path
            futures[executor.submit(download_single_file, rel_path, target_path)] = rel_path

        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 50 == 0 or completed == total:
                pct = (completed / total) * 100
                print(f"    [{completed:03d}/{total:03d}] {pct:5.1f}% completed...")

    print("[+] All files downloaded successfully.")

    # 4. Verify downloaded dataset integrity
    print("-" * 65)
    print("[*] Running MVTecValidator on downloaded data...")
    validator = MVTecValidator(dest_dir, category_name="bottle")
    result = validator.validate(check_image_bytes=True)

    print(result.summary())
    print("-" * 65)

    if result.is_valid:
        print("[SUCCESS] Real MVTec AD 'bottle' category is verified and ready for evaluation!")
        return True
    else:
        print("[FAILURE] Validation found missing or corrupted components.")
        return False


def main() -> int:
    cfg = DatasetConfig(category="bottle")
    dest_dir = cfg.category_dir

    success = download_bottle_category(dest_dir)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
