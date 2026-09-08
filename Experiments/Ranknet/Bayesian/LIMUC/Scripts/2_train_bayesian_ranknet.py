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
# Train the initial Bayesian RankNet model (AL_0)
#
# 1. Read the train / valid pair datasets created by:
#      1_initial_learning_make_pair.py
#
# 2. Train Bayesian RankNet using the initial 20% dataset.
#
# 3. Keep dropout active in the ranking model for
#    Monte Carlo Dropout-based uncertainty estimation.
#
# 4. Save:
#      - model weights
#      - training history
#      - experimental conditions
#
# Input:
# Add_dataset/
#   <al_id>/
#     AL_0/
#       fold_<fold>/
#         RBS/
#           <fold>_train_pair_RBS.csv
#           <fold>_valid_pair_RBS.csv
#
# Output:
# Results/
#   <result_date>_AL_0/
#     fold_<fold>/
#       weights/
#       history.csv
#       experimental_condition.csv
#
# In this implementation, AL_0 is trained using the
# initial randomly sampled 20% dataset.
# ============================================================

# batchgenerator
class BatchGenerator(keras.utils.Sequence):
    def __init__(self, x1_image, x2_image, rel_label, image_shape, batch_size, shuffle=True):
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
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        x1_batch = []
        x2_batch = []
        rel_batch = []
        for i in indexes:
            x1_img = cv2.imread(self.x1[i], cv2.IMREAD_COLOR)
            x1_img = cv2.resize(x1_img, dsize=(self.image_shape[0], self.image_shape[1]))
            x1_batch.append(x1_img)
            x2_img = cv2.imread(self.x2[i], cv2.IMREAD_COLOR)
            x2_img = cv2.resize(x2_img, dsize=(self.image_shape[0], self.image_shape[1]))
            x2_batch.append(x2_img)
            rel_batch.append(self.rel[i])

        x1_batch = np.asarray(x1_batch)
        x1_batch = x1_batch.astype('float32') / 255.0
        x2_batch = np.asarray(x2_batch)
        x2_batch = x2_batch.astype('float32') / 255.0
        rel_batch = np.asarray(rel_batch)
        return [x1_batch, x2_batch, rel_batch]

    def __len__(self):
        return self.batches_per_epoch

    def on_epoch_end(self):
        self.indexes = np.arange(self.length)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

# train and valid datasets
def prepare_dataset():
    """
    Read the AL_0 pair datasets created by
    1_initial_learning_make_pair.py.

    Expected:
      ../Add_dataset/<al_id>/AL_0/fold_<fold>/RBS/
        <fold>_train_pair_RBS.csv
        <fold>_valid_pair_RBS.csv
    """
    al0_dir = os.path.join(
        add_dataset_root,
        al_id,
        "AL_0",
        f"fold_{fold}",
        selection
    )

    train_csv = os.path.join(
        al0_dir,
        f"{fold}_train_pair_{selection}.csv"
    )
    valid_csv = os.path.join(
        al0_dir,
        f"{fold}_valid_pair_{selection}.csv"
    )

    if not os.path.exists(train_csv):
        raise FileNotFoundError(
            f"AL_0 train pair CSV not found: {train_csv}\n"
            "Run 1_initial_learning_make_pair.py first."
        )
    if not os.path.exists(valid_csv):
        raise FileNotFoundError(
            f"AL_0 valid pair CSV not found: {valid_csv}\n"
            "Run 1_initial_learning_make_pair.py first."
        )

    data_train = pd.read_csv(train_csv)
    data_valid = pd.read_csv(valid_csv)

    required = {"x1_image", "x2_image", "relative_label"}
    missing_train = required - set(data_train.columns)
    missing_valid = required - set(data_valid.columns)

    if missing_train:
        raise ValueError(
            f"{train_csv} is missing columns: {sorted(missing_train)}"
        )
    if missing_valid:
        raise ValueError(
            f"{valid_csv} is missing columns: {sorted(missing_valid)}"
        )

    x1_train = np.array(data_train["x1_image"].astype(str))
    x2_train = np.array(data_train["x2_image"].astype(str))
    rel_label_train = np.array(
        data_train["relative_label"],
        dtype="float32"
    )

    x1_valid = np.array(data_valid["x1_image"].astype(str))
    x2_valid = np.array(data_valid["x2_image"].astype(str))
    rel_label_valid = np.array(
        data_valid["relative_label"],
        dtype="float32"
    )

    x1_train = [os.path.join(image_path, s) for s in x1_train]
    x2_train = [os.path.join(image_path, s) for s in x2_train]
    x1_valid = [os.path.join(image_path, s) for s in x1_valid]
    x2_valid = [os.path.join(image_path, s) for s in x2_valid]

    print(f"  train pair csv : {train_csv}")
    print(f"  valid pair csv : {valid_csv}")
    print(f"  train pairs    : {len(x1_train)}")
    print(f"  valid pairs    : {len(x1_valid)}")

    return x1_train, x2_train, rel_label_train,\
           x1_valid, x2_valid, rel_label_valid

# total_loss
def pairwise_loss(x1_score, x2_score, rel_label, sigma=1):
    loss_ce =  K.mean((1 - rel_label) * sigma * (x1_score - x2_score) + K.log(1 + K.exp(-sigma * (x1_score - x2_score))))

    return loss_ce 

# ranknet
def ranknet():
    x1_inputs = Input(shape=image_shape, name='data_x1')
    x2_inputs = Input(shape=image_shape, name='data_x2')

    model_features = DenseNet169(include_top=True, weights="imagenet", dropout_rate=dropout_rate, weight_decay=weight_decay
)
    model_features = Model(inputs=model_features.input, outputs=model_features.get_layer('avg_pool').output)
    x_dropout = Dropout(dropout_rate)(model_features.output, training=True)
    x_fc = Dense(1, kernel_regularizer=regularizers.l2(weight_decay), name='fc')(x_dropout)
    x_fc = Model(inputs=model_features.input, outputs=x_fc)

    if gpu_count >= 2:
        with tf.device('/gpu:0'):
            x1_score = x_fc(x1_inputs)
        with tf.device('/gpu:1'):
            x2_score = x_fc(x2_inputs)
    else:
        with tf.device('/gpu:0'):
            x1_score = x_fc(x1_inputs)
            x2_score = x_fc(x2_inputs)

    with tf.device('/cpu:0'):
        rel_label = Input(shape=(1, ), name='rel_label')
        model = Model(inputs=[x1_inputs, x2_inputs, rel_label], outputs=[x1_score, x2_score])
        model.add_loss(pairwise_loss(x1_score, x2_score, rel_label))

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=None)
    return model

# callbacks
def set_callbacks():
    os.makedirs(result_path + '/weights', exist_ok=True)
    csv_name = result_path + '/history.csv'
    weight_name = result_path + '/weights/epoch{epoch:04d}-{val_loss:.4f}.h5'
    tensorboard_path = result_path + '/logdir/'

    callbacks = makecallbacks(weight_name=weight_name,
                              tsv_name=csv_name,
                              isEarlyStop=True,
                              patience=es_patience,
                              isTensorBoard=False,
                              tensorboard_path=tensorboard_path)
    return callbacks

# save_experimental_condition
def save_condition(x1_train, x1_valid):
    vdic = {
        'date': date,
        'method': method_type,
        'backbone': backbone_type,
        'dataset': dataset_name,
        'al_id': al_id,
        'iteration': 0,
        'selection': selection,
        'initial_rate': 0.20,
        'train_num': len(x1_train),
        'valid_num': len(x1_valid),
        'image_shape': image_shape,
        'batch_size': batch_size,
        'nb_epochs': epoch_num,
        'es_patience': es_patience,
        'learning_rate': learning_rate,
        'dropout_rate': str(dropout_rate),
        'weight_decay': str(weight_decay),
        'gpu_count': gpu_count
    }
    vdic = pd.DataFrame.from_dict(vdic, orient='index').T
    vdic.to_csv(result_path + '/experimental_condition.csv', index=False)

# learning
def learning():
    # prepare train and valid datasets
    x1_train, x2_train, rel_label_train,\
    x1_valid, x2_valid, rel_label_valid = prepare_dataset()

    # model
    model = ranknet()

    # callbacks
    callbacks= set_callbacks()

    # save_experimental_condition
    save_condition(x1_train, x1_valid)

    # batch_generator
    train_batch_generator = BatchGenerator(x1_train,
                                            x2_train,
                                            rel_label_train,
                                            image_shape,
                                            batch_size,
                                            shuffle=True)

    valid_batch_generator = BatchGenerator(x1_valid,
                                            x2_valid,
                                            rel_label_valid,
                                            image_shape,
                                            batch_size,
                                            shuffle=False)

    # model_fit
    model.fit(x=train_batch_generator, 
                epochs=epoch_num,
                steps_per_epoch=train_batch_generator.batches_per_epoch,
                verbose=1,
                callbacks=callbacks,
                validation_data=valid_batch_generator,
                validation_steps=valid_batch_generator.batches_per_epoch,
                shuffle=True)

    # model_reset
    del model
    K.clear_session()
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the initial Bayesian RankNet on the AL_0 20% pair dataset."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="LIMUC",
        help="Dataset name. Default: LIMUC"
    )
    parser.add_argument(
        "--al-id",
        default="LIMUC_AL_001",
        help="Active Learning experiment ID. Default: LIMUC_AL_001"
    )
    parser.add_argument(
        "--selection",
        default="RBS",
        choices=["RBS"],
        help="AL_0 uses random sampling (RBS)."
    )
    parser.add_argument(
        "--folds",
        default="1,2,3,4,5",
        help="Comma-separated folds. Default: 1,2,3,4,5"
    )
    parser.add_argument(
        "--add-dataset-root",
        default="./../Add_dataset",
        help="Root directory containing AL datasets."
    )
    parser.add_argument(
        "--results-root",
        default="./../Results",
        help="Root directory for training results."
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default="1"
    )
    cli_args = parser.parse_args()

    dataset_name = cli_args.dataset
    al_id = cli_args.al_id
    selection = cli_args.selection
    add_dataset_root = cli_args.add_dataset_root

    # image_data_file
    if dataset_name == "LIMUC":
        image_data_file = "all_public_UC_images"
    else:
        image_data_file = "scale_UC_0224"

    # image path
    data_path = "./../../../../../Data/UC/{}/Images".format(dataset_name)
    image_path = os.path.join(data_path, image_data_file)

    # experimental_condition
    now = datetime.now()
    # Make the result directory clearly identifiable as the initial AL model.
    date = now.strftime("%Y%m%d_%H%M%S") + "_AL_0"

    method_type = "RankNet"
    backbone_type = "DenseNet169"
    image_shape = (224, 224, 3)
    batch_size = 32
    epoch_num = 300
    es_patience = 20
    learning_rate = 5e-5
    dropout_rate = 0.2
    weight_decay = 1e-4

    # gpu
    gpu_count = 1

    fold_lst = [
        int(x.strip())
        for x in cli_args.folds.split(",")
        if x.strip()
    ]

    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = cli_args.cuda_visible_devices

    for fold in fold_lst:
        print("")
        print(f"fold {fold}")

        seed = 220428
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        session_conf = tf.compat.v1.ConfigProto(
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1
        )
        sess = tf.compat.v1.Session(
            graph=tf.compat.v1.get_default_graph(),
            config=session_conf
        )
        tf.compat.v1.keras.backend.set_session(sess)

        # result_path
        result_path = os.path.join(
            cli_args.results_root,
            date,
            f"fold_{fold}"
        )

        # learning
        learning()

    print("")
    print("Finished AL_0 training.")
    print(f"result_date: {date}")
    print(f"results: {os.path.join(cli_args.results_root, date)}")
