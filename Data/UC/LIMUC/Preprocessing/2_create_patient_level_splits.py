# -*- coding: utf-8 -*-

"""
Create patient-level 5-fold train, validation, and test splits.

This script:
1. Reads image metadata created by 1_prepare_uc_image_dataset.py.
2. Splits patients into five groups using a fixed random seed.
3. Creates train / validation / test sets without patient overlap.
4. Saves the split metadata as CSV files for each fold.
"""

import argparse
import os
import random

import pandas as pd


DEFAULT_SEED = 20191125
N_FOLDS = 5


def load_metadata(csv_path):
    """Read image metadata and validate the required columns."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

    data = pd.read_csv(csv_path)
    required = {"filename", "patient", "mayo_label"}
    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"{csv_path} is missing required columns: {sorted(missing)}"
        )

    return data


def make_patient_folds(patient_ids, seed=DEFAULT_SEED):
    """Create five deterministic patient groups using a fixed random seed."""
    patient_list = sorted(set(patient_ids))

    if len(patient_list) < N_FOLDS:
        raise ValueError(
            f"At least {N_FOLDS} patients are required, "
            f"but only {len(patient_list)} were found."
        )

    rng = random.Random(seed)
    rng.shuffle(patient_list)

    fold_size = len(patient_list) // N_FOLDS

    patient_folds = [
        patient_list[i * fold_size:(i + 1) * fold_size]
        for i in range(N_FOLDS - 1)
    ]
    patient_folds.append(patient_list[(N_FOLDS - 1) * fold_size:])

    return patient_folds


def split_fold(data, patient_folds, fold_index):
    """Create one train / validation / test split at the patient level."""
    test_patients = set(patient_folds[fold_index])
    valid_patients = set(patient_folds[(fold_index + 1) % N_FOLDS])
    train_patients = set().union(
        *[
            set(patient_folds[i])
            for i in range(N_FOLDS)
            if i not in {fold_index, (fold_index + 1) % N_FOLDS}
        ]
    )

    train_data = data[data["patient"].isin(train_patients)].copy()
    valid_data = data[data["patient"].isin(valid_patients)].copy()
    test_data = data[data["patient"].isin(test_patients)].copy()

    return train_data, valid_data, test_data


def format_output(data):
    """Rename metadata columns to match the downstream training pipeline."""
    return data[["filename", "patient", "mayo_label"]].rename(
        columns={
            "patient": "sequence_num",
            "mayo_label": "mayo_num",
        }
    )


def save_split(data, output_dir, fold_num, split_name):
    """Save one fold split as a CSV file."""
    output_path = os.path.join(
        output_dir,
        f"{fold_num}_original_{split_name}_data.csv",
    )
    format_output(data).to_csv(output_path, index=False)
    print(f"Saved {split_name}: {output_path} ({len(data)} images)")


def divide_data(csv_path, output_dir, seed=DEFAULT_SEED):
    """Create and save patient-level 5-fold train / valid / test datasets."""
    data = load_metadata(csv_path)
    patient_folds = make_patient_folds(data["patient"], seed=seed)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Patients: {data['patient'].nunique()}")
    print(f"Images: {len(data)}")
    print(f"Random seed: {seed}")

    for fold_index in range(N_FOLDS):
        fold_num = fold_index + 1
        train_data, valid_data, test_data = split_fold(
            data,
            patient_folds,
            fold_index,
        )

        print("")
        print(f"Fold {fold_num}")
        print(
            f"  patients: train={train_data['patient'].nunique()}, "
            f"valid={valid_data['patient'].nunique()}, "
            f"test={test_data['patient'].nunique()}"
        )

        save_split(train_data, output_dir, fold_num, "train")
        save_split(valid_data, output_dir, fold_num, "valid")
        save_split(test_data, output_dir, fold_num, "test")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    image_dir = os.path.join(root_dir, "Images")

    parser = argparse.ArgumentParser(
        description="Create patient-level 5-fold train / validation / test splits."
    )
    parser.add_argument(
        "--csv-path",
        default=os.path.join(image_dir, "all_public_UC_data.csv"),
        help="Input metadata CSV created by 1_prepare_uc_image_dataset.py.",
    )
    original_split_dir = os.path.join(image_dir, "original_splits")

    parser.add_argument(
        "--output-dir",
        default=original_split_dir,
        help="Directory in which the original fold split CSV files are saved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed used to shuffle patients. Default: {DEFAULT_SEED}",
    )
    args = parser.parse_args()

    divide_data(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
