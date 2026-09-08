import gc
import sys
import math
import os
import random

import cv2
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.layers import Input
from keras.layers.core import Dense, Dropout
from keras.models import Model
from keras import regularizers

from bayesian_densenet import DenseNet169

# ============================================================
# Predict ranking scores using Monte Carlo Dropout
#
# 1. Read the train / valid / test image datasets.
#
# 2. Load the trained Bayesian RankNet weights for each fold.
#
# 3. Perform repeated predictions with dropout kept active
#    to obtain Monte Carlo Dropout samples.
#
# 4. Save the prediction scores for all sampling runs.
#
# Input:
# Data/UC/<dataset>/Images/
#   dataset/
#     <fold>_train.csv
#     <fold>_valid.csv
#     <fold>_test.csv
#
# Results/
#   <result_date>/
#     fold_<fold>/
#       weights/
#
# Output:
# Results/
#   <result_date>/
#     fold_<fold>/
#       <datatype>_prediction.csv
#       <datatype>_prediction_condition.csv
#
# Each prediction CSV contains one score column for each
# Monte Carlo Dropout sampling run.
# ============================================================

# batchgenerator
class BatchGenerator(keras.utils.Sequence):
    def __init__(self, image, label, image_shape, batch_size, shuffle=True):
        self.x = image
        self.y = label
        self.length = len(image)
        self.indexes = np.arange(self.length)
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.batches_per_epoch = math.ceil(self.length / batch_size)
        self.shuffle = shuffle

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        x_batch = []
        y_batch = []
        for i in indexes:
            img = cv2.imread(self.x[i], cv2.IMREAD_COLOR)
            img = cv2.resize(img, dsize=(image_shape[0], image_shape[1]))
            x_batch.append(img)
            y_batch.append(self.y[i])

        x_batch = np.asarray(x_batch)
        x_batch = x_batch.astype('float32') / 255.0
        y_batch = np.asarray(y_batch)
        return [x_batch, y_batch]

    def __len__(self):
        return self.batches_per_epoch

    def on_epoch_end(self):
        self.indexes = np.arange(self.length)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

# prepare dataset
def prepare_dataset():
    data_train = pd.read_csv(data_path + '/dataset/{}_{}.csv'.format(fold, datatype))
    image_train = data_train['filename']
    label_train = data_train['MayoLabel']
  
    image_train = np.array(image_train)
    label_train = np.array(label_train)

    train = [image_path + s for s in image_train]

    return train, label_train, image_train

# RankNet
def ranknet():
    inputs = Input(shape=image_shape, name='data_x')
    labels = Input(shape=(1, ), name='label')

    model_features = DenseNet169(include_top=True, weights="imagenet", dropout_rate=dropout_rate, weight_decay=weight_decay)
    model_features = Model(inputs=model_features.input, outputs=model_features.get_layer('avg_pool').output)
    x_dropout = Dropout(dropout_rate)(model_features.output, training=True)
    x_fc = Dense(1, kernel_regularizer=regularizers.l2(weight_decay), name='fc')(x_dropout)
    x_fc = Model(inputs=model_features.input, outputs=x_fc)
    scores = x_fc(inputs)
    model = Model(inputs=[inputs, labels], outputs=[scores])
    model.compile()
    return model

# save_prediction_condition
def save_condition(train):
    vdic = {
        'date': result_date,
        'method': method_type,
        'backbone': backbone_type,
        'dataset': str(args[-2]),
        'train_num': len(train),
        'image_shape': image_shape,
        'batch_size': batch_size,
        'sampling_num': sampling_num,
        'gpu_count': gpu_count
    }
    vdic = pd.DataFrame.from_dict(vdic, orient='index').T
    vdic.to_csv(result_path + '/{}_prediction_condition.csv'.format(datatype), index=False)

# predict
def predict():
    # prepare dataset
    train, label_train, image_train = prepare_dataset()

    # model
    model = ranknet()

    # save_prediction_condition
    save_condition(train)

    # load_model
    weight_path = result_path + '/weights/'
    file_lst = os.listdir(weight_path)
    file_lst.sort()
    weight_path_lst = [weight_path + s for s in file_lst]
    model.load_weights(weight_path_lst[-1])

    # batch_generator
    train_batch_generator = BatchGenerator(train, label_train, image_shape, batch_size, shuffle=False)

    # model.predict
    score_lst = []
    for n in range(sampling_num):
        score = model.predict(train_batch_generator, verbose=1, batch_size=batch_size)
        score = np.array(score)
        score = score.reshape(np.array(image_train).shape)
        score_lst.append(score)

    # save_result
    df_dataset = pd.DataFrame()
    df_dataset['filename'] = image_train
    df_dataset['label'] = label_train
    for n in range(sampling_num):
        df_dataset['sampling_' + str(n)] = score_lst[n]
    csv_path = result_path + '/{}_prediction.csv'.format(datatype)
    df_dataset.to_csv(csv_path, index=False)

    # model_reset
    del model
    K.clear_session()
    gc.collect()


if __name__ == "__main__":
    args = sys.argv

    # image_data_file
    if args[-2] == 'LIMUC':
        image_data_file = 'all_public_UC_images'
    else:
        image_data_file = 'scale_UC_0224'

    # path
    data_path = './../../../../../Data/UC/{}/Images'.format(args[-2])
    image_path = data_path + '/{}/'.format(image_data_file)

    # prediction_condition
    result_date = str(args[-1])
    method_type = 'RankNet'
    backbone_type = 'DenseNet169'
    image_shape = (224, 224, 3)
    batch_size = 64
    epoch_num = None
    es_patience = None
    learning_rate = None
    dropout_rate = 0.2
    weight_decay = 1e-4
    sampling_num = 30

    # gpu
    gpu_count = 1

    fold_lst = [1,2,3,4,5]
    datatype_lst = ['train','valid','test']
    for fold in fold_lst:
        for datatype in datatype_lst:    
            # random_seed
            os.environ['PYTHONHASHSEED'] = '0'
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
            os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"

            seed = 220428
            tf.random.set_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
            sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
            tf.compat.v1.keras.backend.set_session(sess)

            # result_path
            result_path = './../Results/{}/fold_{}/'.format(result_date, fold)

            # predict
            predict()
