# -*- coding: utf-8 -*-

"""
Collect and preprocess ulcerative colitis endoscopy images.

This script:
1. Collects BMP images organized by patient and Mayo endoscopic subscore.
2. Removes images with duplicated output filenames.
3. Copies the original BMP images and converts them to JPEG.
4. Saves image metadata, patient IDs, and Mayo labels as a CSV file.
"""

import os
import shutil
from collections import Counter

import cv2
import pandas as pd


CLASS_NAMES = ["Mayo 0", "Mayo 1", "Mayo 2", "Mayo 3"]
LABEL_MAP = {
    "Mayo 0": 0,
    "Mayo 1": 1,
    "Mayo 2": 2,
    "Mayo 3": 3,
}


def collect_image_records(patient_root):
    """Collect image metadata from patient- and Mayo-class-based directories."""
    records = []

    patient_list = sorted(
        patient
        for patient in os.listdir(patient_root)
        if os.path.isdir(os.path.join(patient_root, patient))
    )

    for patient in patient_list:
        for mayo_label in CLASS_NAMES:
            mayo_dir = os.path.join(patient_root, patient, mayo_label)

            if not os.path.isdir(mayo_dir):
                raise FileNotFoundError(f"Expected directory not found: {mayo_dir}")

            for image_name in sorted(os.listdir(mayo_dir)):
                source_path = os.path.join(mayo_dir, image_name)

                if not os.path.isfile(source_path):
                    continue

                stem, extension = os.path.splitext(image_name)
                if extension.lower() != ".bmp":
                    continue

                jpg_name = f"{stem}.JPG"
                records.append(
                    {
                        "filename": jpg_name,
                        "patient": patient,
                        "mayo_label": mayo_label,
                        "source_path": source_path,
                        "original_name": image_name,
                    }
                )

    return records


def remove_duplicate_filenames(records):
    """Remove all records whose output filename appears more than once."""
    filename_counts = Counter(record["filename"] for record in records)
    duplicate_names = {
        filename for filename, count in filename_counts.items() if count > 1
    }

    unique_records = [
        record for record in records if record["filename"] not in duplicate_names
    ]

    return unique_records, duplicate_names


def save_images(records, original_output_dir, jpg_output_dir):
    """Copy original BMP images and convert them to JPEG."""
    os.makedirs(original_output_dir, exist_ok=True)
    os.makedirs(jpg_output_dir, exist_ok=True)

    for record in records:
        source_path = record["source_path"]
        original_name = record["original_name"]
        jpg_name = record["filename"]

        original_output_path = os.path.join(original_output_dir, original_name)
        jpg_output_path = os.path.join(jpg_output_dir, jpg_name)

        shutil.copyfile(source_path, original_output_path)

        image = cv2.imread(source_path)
        if image is None:
            raise ValueError(f"Failed to read image: {source_path}")

        success = cv2.imwrite(
            jpg_output_path,
            image,
            [cv2.IMWRITE_JPEG_QUALITY, 100],
        )
        if not success:
            raise IOError(f"Failed to save JPEG image: {jpg_output_path}")


def save_metadata(records, csv_path):
    """Save image filename, patient ID, and Mayo label to CSV."""
    data = pd.DataFrame(
        {
            "filename": [record["filename"] for record in records],
            "patient": [record["patient"] for record in records],
            "mayo_label": [LABEL_MAP[record["mayo_label"]] for record in records],
        }
    )

    data.to_csv(csv_path, index=False)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    image_dir = os.path.join(root_dir, "Images")

    patient_root = os.path.join(image_dir, "patient_based_classified_images")
    original_output_dir = os.path.join(image_dir, "all_public_UC_images_original_bmp")
    jpg_output_dir = os.path.join(image_dir, "all_public_UC_images")
    csv_path = os.path.join(image_dir, "all_public_UC_data.csv")

    if not os.path.isdir(patient_root):
        raise FileNotFoundError(f"Patient image directory not found: {patient_root}")

    records = collect_image_records(patient_root)
    records, duplicate_names = remove_duplicate_filenames(records)

    if duplicate_names:
        print(
            f"Excluded {len(duplicate_names)} duplicated output filename(s): "
            f"{sorted(duplicate_names)}"
        )

    save_images(records, original_output_dir, jpg_output_dir)
    save_metadata(records, csv_path)

    print(f"Saved {len(records)} images.")
    print(f"Saved metadata: {csv_path}")


if __name__ == "__main__":
    main()
