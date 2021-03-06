import tensorflow as tf
import time
import os
import matplotlib.pyplot as plt
import pandas as pd
from utils.common import get_timestamp

def create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES=10):
    LAYERS = [
          tf.keras.layers.Flatten(input_shape=[28,28], name="inputLayer"),
          tf.keras.layers.Dense(300, activation="relu", name="hiddenLayer1"),
          tf.keras.layers.Dense(100, activation="relu", name="hiddenLayer2"),
          tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="outputLayer")
    ]
    model_clf = tf.keras.models.Sequential(LAYERS)
    model_clf.summary()

    model_clf.compile(loss=LOSS_FUNCTION,
                      optimizer=OPTIMIZER,
                      metrics=METRICS)
    return model_clf

def save_model(model, model_name, model_dir):
    unique_filename = get_timestamp(model_name)
    path_to_model = os.path.join(model_dir, unique_filename)
    model.save(path_to_model)

def save_plot(history, plot_name, plot_dir):
    unique_plotname = get_timestamp(plot_name)
    path_to_plot = os.path.join(plot_dir, unique_plotname)
    pd.DataFrame(history.history).plot(figsize=(10, 7))
    plt.grid()
    plt.savefig(path_to_plot)