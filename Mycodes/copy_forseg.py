#!/usr/bin/env python3
"""Copy pdsag/t2sag NIfTI files into nnUNet inference folder with renamed ids."""

import re
import shutil
from pathlib import Path

SRC_ROOT = Path("/root/autodl-tmp/DATASET/nnUNet_raw/Dataset002_OurOA/data_nii_3000")
DST_ROOT = Path(
    "/root/autodl-tmp/DATASET/nnUNet_raw/Dataset002_OurOA/data_nii_3000_forseg"
)

SEQ_MAP = {
    # "pdsag.nii.gz": "0000",
    # "t2sag.nii.gz": "0001",
    "t1sag.nii.gz": "0002",
}


def extract_m_number(folder_name: str) -> str:
    """Extract digits after 'M' in folder name."""
    match = re.search(r"M(\d+)", folder_name)
    if not match:
        raise ValueError(f"Folder name '{folder_name}' does not contain 'M' followed by digits")
    return match.group(1)


def copy_and_rename():
    if not SRC_ROOT.exists():
        raise FileNotFoundError(f"Source root not found: {SRC_ROOT}")

    DST_ROOT.mkdir(parents=True, exist_ok=True)

    patient_dirs = [p for p in SRC_ROOT.iterdir() if p.is_dir()]
    if not patient_dirs:
        print(f"No patient folders found in {SRC_ROOT}")
        return

    total = 0
    for patient_dir in sorted(patient_dirs):
        try:
            m_number = extract_m_number(patient_dir.name)
        except ValueError as exc:
            print(f"Skip: {exc}")
            continue

        for src_name, seq_code in SEQ_MAP.items():
            src_file = patient_dir / src_name
            if not src_file.exists():
                print(f"Missing: {src_file}")
                continue

            dst_name = f"OurOA_{m_number}_{seq_code}.nii.gz"
            dst_file = DST_ROOT / dst_name

            shutil.copy2(src_file, dst_file)
            total += 1
            print(f"Copied: {src_file} -> {dst_file}")

    print(f"Done. Total copied files: {total}")


if __name__ == "__main__":
    copy_and_rename()
