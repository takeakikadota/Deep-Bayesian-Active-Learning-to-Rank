# -*- coding: utf-8 -*-

"""
Create shuffled training dataset CSV files for train and validation splits.

This script:
1. Reads fold-specific train / validation CSV files from Images/dataset/.
2. Keeps the filename and MayoLabel columns.
3. Shuffles the rows using a fixed random seed.
4. Saves the shuffled CSV files to Images/training_dataset/.
"""

import argparse
import os

import numpy as np
import pandas as pd


DEFAULT_SEED = 20231205
N_FOLDS = 5
SPLIT_NAMES = ["train", "valid"]


def create_training_dataset(
    fold_num,
    split_name,
    input_dir,
    output_dir,
    seed=DEFAULT_SEED,
):
    """Create one shuffled training dataset CSV."""
    input_path = os.path.join(
        input_dir,
        f"{fold_num}_{split_name}.csv",
    )

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Input dataset CSV not found: {input_path}"
        )

    data = pd.read_csv(input_path)

    required = {"filename", "MayoLabel"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"{input_path} is missing required columns: {sorted(missing)}"
        )

    output_data = data[["filename", "MayoLabel"]].copy()

    rng = np.random.RandomState(seed)
    shuffled_indices = rng.permutation(len(output_data))
    output_data = output_data.iloc[shuffled_indices].reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        f"{fold_num}_{split_name}.csv",
    )
    output_data.to_csv(output_path, index=False)

    print(f"Saved: {output_path} ({len(output_data)} images)")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)

    default_input_dir = os.path.join(root_dir, "Images", "dataset")
    default_output_dir = os.path.join(root_dir, "Images", "training_dataset")

    parser = argparse.ArgumentParser(
        description="Create shuffled training dataset CSV files."
    )
    parser.add_argument(
        "--input-dir",
        default=default_input_dir,
        help="Directory containing fold-specific dataset CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default=default_output_dir,
        help="Directory in which shuffled training CSV files are saved.",
    )
    parser.add_argument(
        "--folds",
        default="1,2,3,4,5",
        help="Comma-separated fold numbers. Default: 1,2,3,4,5",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed used to shuffle rows. Default: {DEFAULT_SEED}",
    )
    args = parser.parse_args()

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
            create_training_dataset(
                fold_num=fold_num,
                split_name=split_name,
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                seed=args.seed,
            )

    print("")
    print("Finished creating shuffled training dataset CSV files.")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
