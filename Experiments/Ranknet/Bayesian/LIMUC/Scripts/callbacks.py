import numpy as np
from keras.callbacks import CSVLogger,  ModelCheckpoint, EarlyStopping, TensorBoard

# ============================================================
# Callback utilities for Bayesian RankNet training
#
# - Save training history
# - Save the best model weights based on validation loss
# - Apply early stopping
# - Optionally enable TensorBoard logging
# ============================================================

def makecallbacks(tensorboard_path,
                  weight_name=None,
                  tsv_name=None,
                  base_model=None,
                  isEarlyStop=False,
                  patience=10,
                  isTensorBoard=True):

    callbacks = []

    # loss save
    csv_cb = CSVLogger(tsv_name, separator='\t')
    callbacks.append(csv_cb)

    # weights save
    checkpoint_cb = ModelCheckpoint(
        weight_name, 
        monitor='val_loss',
        save_weights_only=True, 
        save_best_only=True,
        mode='auto',
        verbose=1)
    callbacks.append(checkpoint_cb)

    # early stopping
    es_cb = EarlyStopping(
        monitor='val_loss', 
        patience=patience, 
        verbose=1, 
        mode='auto')
    callbacks.append(es_cb)

    # tensorboard
    if isTensorBoard:
        tb_cb = TensorBoard(log_dir=tensorboard_path, histogram_freq=1)
        callbacks.append(tb_cb)

    return callbacks
