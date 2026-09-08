# -*- coding: utf-8 -*-

"""
Create fold-specific dataset CSV files for downstream training.

This script:
1. Reads the patient-level train / validation / test split CSV files.
2. Renames the metadata columns to match the downstream training pipeline.
3. Saves one dataset CSV for each fold and split.
"""

import argparse
import os

import pandas as pd


N_FOLDS = 5
SPLIT_NAMES = ["train", "valid", "test"]


def make_dataset(fold_num, split_name, image_dir, output_dir):
    """Create one fold-specific dataset CSV."""
    input_path = os.path.join(
        image_dir,
        f"{fold_num}_original_{split_name}_data.csv",
    )

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input split CSV not found: {input_path}")

    data = pd.read_csv(input_path)

    required = {"filename", "sequence_num", "mayo_num"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"{input_path} is missing required columns: {sorted(missing)}"
        )

    output_data = data[
        ["filename", "sequence_num", "mayo_num"]
    ].rename(
        columns={
            "sequence_num": "sequence",
            "mayo_num": "MayoLabel",
        }
    )

    output_path = os.path.join(
        output_dir,
        f"{fold_num}_{split_name}.csv",
    )
    output_data.to_csv(output_path, index=False)

    print(f"Saved: {output_path} ({len(output_data)} images)")


def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_dir = os.path.join(root_dir, "Images")
    default_input_dir = os.path.join(image_dir, "original_splits")

    parser = argparse.ArgumentParser(
        description="Create fold-specific dataset CSV files for downstream training."
    )
    parser.add_argument(
        "--input-dir",
        default=default_input_dir,
        help="Directory containing the original fold split CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: <Images>/dataset",
    )
    parser.add_argument(
        "--folds",
        default="1,2,3,4,5",
        help="Comma-separated fold numbers. Default: 1,2,3,4,5",
    )
    args = parser.parse_args()

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else os.path.join(image_dir, "dataset")
    )
    os.makedirs(output_dir, exist_ok=True)

    fold_list = [
        int(value.strip())
        for value in args.folds.split(",")
        if value.strip()
    ]

    for fold_num in fold_list:
        if fold_num < 1 or fold_num > N_FOLDS:
            raise ValueError(
                f"Fold number must be between 1 and {N_FOLDS}: {fold_num}"
            )

        for split_name in SPLIT_NAMES:
            make_dataset(
                fold_num=fold_num,
                split_name=split_name,
                image_dir=args.input_dir,
                output_dir=output_dir,
            )

    print("")
    print("Finished creating fold-specific dataset CSV files.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
