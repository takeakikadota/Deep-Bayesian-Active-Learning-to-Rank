import argparse
import gc
import math
import os
import random
from datetime import datetime

import cv2
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.layers import Input
from keras import regularizers
from keras.layers.core import Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam

from bayesian_densenet import DenseNet169
from callbacks import makecallbacks


# ============================================================
# Bayesian RankNet training for cumulative Active Learning
#
# AL_0:
#   - reads initial 20% RBS pair CSVs
#   - starts from ImageNet-pretrained Bayesian DenseNet169
#
# AL_1 and later:
#   - reads cumulative pair CSVs created by
#       5_active_learning_make_pair.py
#   - supports UBS or RBS cumulative datasets
#   - ALWAYS loads the best weights from the fixed AL_0 result
#   - trains each AL iteration independently from the same AL_0 initialization
#
# Example:
#   AL_0:
#     python 6_train_bayesian_ranknet_AL.py LIMUC --iteration 0
#
#   AL_1 (UBS):
#     python 6_train_bayesian_ranknet_AL.py LIMUC \
#       --iteration 1 \
#       --selection UBS \
#       --initial-result-date 20260825_161805_AL_0
#
#   AL_2 (UBS):
#     python 6_train_bayesian_ranknet_AL.py LIMUC \
#       --iteration 2 \
#       --selection UBS \
#       --initial-result-date 20260825_161805_AL_0
# ============================================================


class BatchGenerator(keras.utils.Sequence):
    def __init__(
        self,
        x1_image,
        x2_image,
        rel_label,
        image_shape,
        batch_size,
        shuffle=True,
    ):
        self.x1 = x1_image
        self.x2 = x2_image
        self.rel = rel_label
        self.length = len(x1_image)
        self.indexes = np.arange(self.length)
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.batches_per_epoch = math.ceil(self.length / batch_size)
        self.shuffle = shuffle

    def __getitem__(self, index):
        indexes = self.indexes[
            index * self.batch_size:(index + 1) * self.batch_size
        ]

        x1_batch = []
        x2_batch = []
        rel_batch = []

        for i in indexes:
            x1_img = cv2.imread(self.x1[i], cv2.IMREAD_COLOR)
            x2_img = cv2.imread(self.x2[i], cv2.IMREAD_COLOR)

            if x1_img is None:
                raise FileNotFoundError(
                    f"Could not read x1 image: {self.x1[i]}"
                )
            if x2_img is None:
                raise FileNotFoundError(
                    f"Could not read x2 image: {self.x2[i]}"
                )

            x1_img = cv2.resize(
                x1_img,
                dsize=(self.image_shape[0], self.image_shape[1]),
            )
            x2_img = cv2.resize(
                x2_img,
                dsize=(self.image_shape[0], self.image_shape[1]),
            )

            x1_batch.append(x1_img)
            x2_batch.append(x2_img)
            rel_batch.append(self.rel[i])

        x1_batch = np.asarray(x1_batch, dtype="float32") / 255.0
        x2_batch = np.asarray(x2_batch, dtype="float32") / 255.0
        rel_batch = np.asarray(rel_batch, dtype="float32")

        return [x1_batch, x2_batch, rel_batch]

    def __len__(self):
        return self.batches_per_epoch

    def on_epoch_end(self):
        self.indexes = np.arange(self.length)
        if self.shuffle:
            np.random.shuffle(self.indexes)


def get_pair_csv(datatype):
    """
    Return the pair CSV for the requested AL iteration.

    AL_0:
      Add_dataset/<al_id>/AL_0/fold_<fold>/RBS/
        <fold>_<datatype>_pair_RBS.csv

    AL_1+:
      Add_dataset/<al_id>/AL_<iteration>/fold_<fold>/<selection>/
        <fold>_<datatype>_pair_<selection>_cumulative.csv
    """
    if iteration == 0:
        pair_dir = os.path.join(
            add_dataset_root,
            al_id,
            "AL_0",
            f"fold_{fold}",
            "RBS",
        )
        return os.path.join(
            pair_dir,
            f"{fold}_{datatype}_pair_RBS.csv",
        )

    pair_dir = os.path.join(
        add_dataset_root,
        al_id,
        f"AL_{iteration}",
        f"fold_{fold}",
        selection,
    )
    return os.path.join(
        pair_dir,
        f"{fold}_{datatype}_pair_{selection}_cumulative.csv",
    )


def read_pair_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Pair CSV not found: {csv_path}"
        )

    data = pd.read_csv(csv_path)

    required = {"x1_image", "x2_image", "relative_label"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"{csv_path} is missing columns: {sorted(missing)}"
        )

    x1 = data["x1_image"].astype(str).to_numpy()
    x2 = data["x2_image"].astype(str).to_numpy()
    rel = data["relative_label"].astype("float32").to_numpy()

    x1 = [os.path.join(image_path, s) for s in x1]
    x2 = [os.path.join(image_path, s) for s in x2]

    return x1, x2, rel


def prepare_dataset():
    train_csv = get_pair_csv("train")
    valid_csv = get_pair_csv("valid")

    x1_train, x2_train, rel_train = read_pair_csv(train_csv)
    x1_valid, x2_valid, rel_valid = read_pair_csv(valid_csv)

    print(f"  train pair csv : {train_csv}")
    print(f"  valid pair csv : {valid_csv}")
    print(f"  train pairs    : {len(x1_train)}")
    print(f"  valid pairs    : {len(x1_valid)}")

    return (
        x1_train,
        x2_train,
        rel_train,
        x1_valid,
        x2_valid,
        rel_valid,
    )


def pairwise_loss(x1_score, x2_score, rel_label, sigma=1):
    return K.mean(
        (1 - rel_label) * sigma * (x1_score - x2_score)
        + K.log(
            1 + K.exp(
                -sigma * (x1_score - x2_score)
            )
        )
    )


def ranknet():
    x1_inputs = Input(shape=image_shape, name="data_x1")
    x2_inputs = Input(shape=image_shape, name="data_x2")

    model_features = DenseNet169(
        include_top=True,
        weights="imagenet",
        dropout_rate=dropout_rate,
        weight_decay=weight_decay,
    )

    model_features = Model(
        inputs=model_features.input,
        outputs=model_features.get_layer("avg_pool").output,
    )

    x_dropout = Dropout(dropout_rate)(
        model_features.output,
        training=True,
    )

    x_fc = Dense(
        1,
        kernel_regularizer=regularizers.l2(weight_decay),
        name="fc",
    )(x_dropout)

    x_fc = Model(
        inputs=model_features.input,
        outputs=x_fc,
    )

    if gpu_count >= 2:
        with tf.device("/gpu:0"):
            x1_score = x_fc(x1_inputs)
        with tf.device("/gpu:1"):
            x2_score = x_fc(x2_inputs)
    else:
        with tf.device("/gpu:0"):
            x1_score = x_fc(x1_inputs)
            x2_score = x_fc(x2_inputs)

    with tf.device("/cpu:0"):
        rel_label = Input(shape=(1,), name="rel_label")
        model = Model(
            inputs=[x1_inputs, x2_inputs, rel_label],
            outputs=[x1_score, x2_score],
        )
        model.add_loss(
            pairwise_loss(
                x1_score,
                x2_score,
                rel_label,
            )
        )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=None,
    )

    return model


def find_al0_initial_weight():
    """
    Select the fixed AL_0 initialization weight.

    Because callbacks.py uses save_best_only=True, only checkpoints
    that improve validation loss are saved. Therefore, the checkpoint
    from the latest saved epoch is the final best AL_0 weight.
    """
    if initial_result_date is None:
        raise ValueError(
            "--initial-result-date is required for AL_1 to AL_6 and must point to the AL_0 result directory."
        )

    weight_dir = os.path.join(
        results_root,
        initial_result_date,
        f"fold_{fold}",
        "weights",
    )

    if not os.path.isdir(weight_dir):
        raise FileNotFoundError(
            f"AL_0 initialization weight directory not found: {weight_dir}"
        )

    weight_files = [
        os.path.join(weight_dir, name)
        for name in os.listdir(weight_dir)
        if name.endswith(".h5")
    ]

    if not weight_files:
        raise FileNotFoundError(
            f"No AL_0 .h5 weights found in: {weight_dir}"
        )

    weight_files.sort()
    best_path = weight_files[-1]

    print(
        f"  AL_0 initialization weight : {best_path}"
    )

    return best_path

def set_callbacks():
    os.makedirs(
        os.path.join(result_path, "weights"),
        exist_ok=True,
    )

    csv_name = os.path.join(
        result_path,
        "history.csv",
    )
    weight_name = os.path.join(
        result_path,
        "weights",
        "epoch{epoch:04d}-{val_loss:.4f}.h5",
    )
    tensorboard_path = os.path.join(
        result_path,
        "logdir",
    )

    return makecallbacks(
        weight_name=weight_name,
        tsv_name=csv_name,
        isEarlyStop=True,
        patience=es_patience,
        isTensorBoard=False,
        tensorboard_path=tensorboard_path,
    )


def save_condition(
    x1_train,
    x1_valid,
    loaded_weight,
):
    vdic = {
        "date": date,
        "method": method_type,
        "backbone": backbone_type,
        "dataset": dataset_name,
        "al_id": al_id,
        "iteration": iteration,
        "selection": (
            "RBS" if iteration == 0 else selection
        ),
        "initial_rate": 0.20,
        "add_rate_per_iteration": 0.05,
        "initial_result_date": initial_result_date,
        "loaded_initial_weight": loaded_weight,
        "train_num": len(x1_train),
        "valid_num": len(x1_valid),
        "image_shape": image_shape,
        "batch_size": batch_size,
        "nb_epochs": epoch_num,
        "es_patience": es_patience,
        "learning_rate": learning_rate,
        "dropout_rate": str(dropout_rate),
        "weight_decay": str(weight_decay),
        "gpu_count": gpu_count,
    }

    pd.DataFrame.from_dict(
        vdic,
        orient="index",
    ).T.to_csv(
        os.path.join(
            result_path,
            "experimental_condition.csv",
        ),
        index=False,
    )


def learning():
    (
        x1_train,
        x2_train,
        rel_train,
        x1_valid,
        x2_valid,
        rel_valid,
    ) = prepare_dataset()

    model = ranknet()

    loaded_weight = None

    # Cumulative AL training:
    # AL_1 to AL_6 always start from the fixed AL_0 model.
    if iteration >= 1:
        loaded_weight = find_al0_initial_weight()
        model.load_weights(loaded_weight)

    callbacks = set_callbacks()

    save_condition(
        x1_train,
        x1_valid,
        loaded_weight,
    )

    train_gen = BatchGenerator(
        x1_train,
        x2_train,
        rel_train,
        image_shape,
        batch_size,
        shuffle=True,
    )

    valid_gen = BatchGenerator(
        x1_valid,
        x2_valid,
        rel_valid,
        image_shape,
        batch_size,
        shuffle=False,
    )

    model.fit(
        x=train_gen,
        epochs=epoch_num,
        steps_per_epoch=train_gen.batches_per_epoch,
        verbose=1,
        callbacks=callbacks,
        validation_data=valid_gen,
        validation_steps=valid_gen.batches_per_epoch,
        shuffle=True,
    )

    del model
    K.clear_session()
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Train Bayesian RankNet with cumulative Active Learning "
            "(AL_0 to AL_6)."
        )
    )

    parser.add_argument(
        "dataset",
        nargs="?",
        default="LIMUC",
    )
    parser.add_argument(
        "--al-id",
        default="LIMUC_AL_001",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        required=True,
        help="AL iteration: 0, 1, ..., 6",
    )
    parser.add_argument(
        "--initial-result-date",
        default=None,
        help=(
            "Fixed AL_0 training result directory. "
            "Required for iteration >= 1."
        ),
    )
    parser.add_argument(
        "--selection",
        choices=["UBS", "RBS"],
        default="UBS",
        help="Sampling method for AL_1 and later iterations.",
    )
    parser.add_argument(
        "--folds",
        default="1,2,3,4,5",
    )
    parser.add_argument(
        "--add-dataset-root",
        default="./../Add_dataset",
    )
    parser.add_argument(
        "--results-root",
        default="./../Results",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default="1",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    if args.iteration < 0:
        raise ValueError("--iteration must be >= 0.")

    if args.iteration >= 1 and args.initial_result_date is None:
        raise ValueError(
            "--initial-result-date is required for AL_1 to AL_6."
        )

    if args.iteration >= 1 and "_AL_0" not in args.initial_result_date:
        raise ValueError(
            "--initial-result-date must point to an AL_0 result directory "
            "(for example: 20260825_161805_AL_0)."
        )

    dataset_name = args.dataset
    al_id = args.al_id
    iteration = args.iteration
    selection = args.selection
    initial_result_date = args.initial_result_date

    add_dataset_root = args.add_dataset_root
    results_root = args.results_root

    if dataset_name == "LIMUC":
        image_data_file = "all_public_UC_images"
    else:
        image_data_file = "scale_UC_0224"

    data_path = (
        "./../../../../../Data/UC/"
        f"{dataset_name}/Images"
    )
    image_path = os.path.join(
        data_path,
        image_data_file,
    )

    now = datetime.now()
    date = (
        now.strftime("%Y%m%d_%H%M%S")
        + f"_AL_{iteration}"
    )

    method_type = "RankNet"
    backbone_type = "DenseNet169"
    image_shape = (224, 224, 3)
    batch_size = 32
    epoch_num = args.epochs
    es_patience = args.patience
    learning_rate = args.learning_rate
    dropout_rate = 0.2
    weight_decay = 1e-4

    gpu_count = 1

    fold_lst = [
        int(x.strip())
        for x in args.folds.split(",")
        if x.strip()
    ]

    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ[
        "CUDA_VISIBLE_DEVICES"
    ] = args.cuda_visible_devices

    print("")
    print("======================================")
    print(" Cumulative Active Learning training")
    print("======================================")
    print(f"al_id                : {al_id}")
    print(f"iteration            : AL_{iteration}")
    print(
        f"selection            : "
        f"{'RBS' if iteration == 0 else selection}"
    )
    print(
        f"initial_result_date : "
        f"{initial_result_date}"
    )
    print(f"learning_rate        : {learning_rate}")
    print(f"epochs               : {epoch_num}")
    print(f"patience             : {es_patience}")
    print(f"new result_date      : {date}")

    for fold in fold_lst:
        print("")
        print(f"fold {fold}")

        seed = 220428
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        session_conf = tf.compat.v1.ConfigProto(
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1,
        )
        sess = tf.compat.v1.Session(
            graph=tf.compat.v1.get_default_graph(),
            config=session_conf,
        )
        tf.compat.v1.keras.backend.set_session(sess)

        result_path = os.path.join(
            results_root,
            date,
            f"fold_{fold}",
        )
        os.makedirs(result_path, exist_ok=True)

        learning()

    print("")
    print(f"Finished AL_{iteration} training.")
    print(f"result_date: {date}")
    print(
        "results: "
        f"{os.path.join(results_root, date)}"
    )
