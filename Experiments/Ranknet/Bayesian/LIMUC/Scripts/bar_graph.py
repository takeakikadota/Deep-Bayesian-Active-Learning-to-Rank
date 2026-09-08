import os
import sys
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker


# ============================================================
# Compare Mayo score distributions of selected samples
#
# 1. Read prediction uncertainty calculated by
#      4_calculate_uncertainty.py
#
# 2. Select the top 5% most uncertain images.
#
# 3. Randomly select the same number of images for comparison.
#
# 4. Compare the Mayo score distributions of the
#    random and uncertainty-based selections.
#
# Output:
# Results/
#   <result_date>/
#     fold_<fold>/
#       <datatype>_class_imbalance.pdf
# ============================================================


SELECT_RATE = 0.05
RANDOM_SEED = 0


def read_pred(root_dir, datatype):
    datafile = os.path.join(
        root_dir,
        f"{datatype}_mean_score_and_uncertainty.csv",
    )
    data = pd.read_csv(datafile)

    labels = data["label"].to_numpy()
    uncertainty = data["var_score"].to_numpy()

    return labels, uncertainty


def graph(result_path, datatype):
    # data_load
    labels, uncertainty = read_pred(result_path, datatype)

    labels = np.asarray(labels)
    uncertainty = np.asarray(uncertainty)

    # select_high_uncertainty
    data_number = len(labels)
    select_number = round(data_number * SELECT_RATE)

    # IMPORTANT:
    # Keep the original tuple sorting exactly as before so that
    # the selected samples and resulting figure do not change.
    data_lst = list(zip(uncertainty, labels))
    data_lst = sorted(data_lst, reverse=True)

    select_data_lst = data_lst[:select_number]
    _, select_labels = zip(*select_data_lst)

    mayo0_num = select_labels.count(0)
    mayo1_num = select_labels.count(1)
    mayo2_num = select_labels.count(2)
    mayo3_num = select_labels.count(3)

    # random_select
    # Keep sampling from the sorted list exactly as in the original code.
    random.seed(RANDOM_SEED)
    random_data_lst = random.sample(data_lst, select_number)
    _, random_labels = zip(*random_data_lst)

    random_mayo0_num = random_labels.count(0)
    random_mayo1_num = random_labels.count(1)
    random_mayo2_num = random_labels.count(2)
    random_mayo3_num = random_labels.count(3)

    plt.rcParams["font.size"] = 14

    height2 = [
        random_mayo0_num,
        random_mayo1_num,
        random_mayo2_num,
        random_mayo3_num,
    ]
    height3 = [
        mayo0_num,
        mayo1_num,
        mayo2_num,
        mayo3_num,
    ]

    left = np.arange(len(height2))
    xlabels = ["M0", "M1", "M2", "M3"]
    width = 0.25

    fig, ax = plt.subplots(figsize=(5, 4))

    b2 = ax.bar(
        left + width,
        height2,
        color="royalblue",
        width=width,
        align="center",
    )
    b3 = ax.bar(
        left + width + width,
        height3,
        color="crimson",
        width=width,
        align="center",
    )

    ax.legend(
        (b2[0], b3[0]),
        ("Random", "Proposed"),
        edgecolor="white",
        loc="upper right",
        ncol=1,
        fontsize=10,
    )

    ax.set_xlabel("Mayo score", fontsize=14)
    ax.set_ylabel("Number of images", fontsize=14)

    spines = 0.5
    ax.spines["top"].set_linewidth(spines)
    ax.spines["left"].set_linewidth(spines)
    ax.spines["bottom"].set_linewidth(spines)
    ax.spines["right"].set_linewidth(spines)

    ax.spines["bottom"].set_color("silver")
    ax.spines["top"].set_color("silver")
    ax.spines["left"].set_color("silver")
    ax.spines["right"].set_color("silver")

    ax.tick_params(width=0.5, length=0)

    ax.set_xticks(left + width * 3 / 2)
    ax.set_xticklabels(xlabels)

    # Keep the original fixed y-axis range.
    ax.set_ylim(0, 400)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(50))

    fig.subplots_adjust(left=0.19, bottom=0.3, top=0.9)
    ax.grid(axis="y", linewidth=0.5)

    output_path = os.path.join(
        result_path,
        f"{datatype}_class_imbalance.pdf",
    )
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    result_date = str(sys.argv[-1])

    fold_lst = [1, 2, 3, 4, 5]
    datatype_lst = ["train", "valid", "test"]

    for fold in fold_lst:
        for datatype in datatype_lst:
            result_path = os.path.join(
                "./../Results",
                result_date,
                f"fold_{fold}",
            )
            graph(result_path, datatype)
