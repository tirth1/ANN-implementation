import os.path
from utils.common import read_config
from utils.data_mgmt import get_data
from utils.model import create_model, save_model, save_plot, get_log_path
import argparse
import tensorflow as tf

def training(config_path):
    config = read_config(config_path)
    validation_datasize = config["params"]["validation_datasize"]
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_data(validation_datasize)
    LOSS_FUNCTION = config["params"]["loss_function"]
    OPTIMIZER = config["params"]["optimizer"]
    METRICS = config["params"]["metrics"]
    NUM_CLASSES = config["params"]["num_classes"]
    LOG_DIR = get_log_path()
    model = create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES)

    # callback functions
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir = LOG_DIR)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    CKPT_path = 'model_ckpt.h5'
    checkpointing_cb = tf.keras.callbacks.ModelCheckpoint(CKPT_path, save_best_only=True)

    EPOCHS = config["params"]["epochs"]
    VALIDATION = (X_valid, y_valid)
    CALLBACKS = [tensorboard_cb, early_stopping_cb, checkpointing_cb]
    history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=VALIDATION, callbacks=CALLBACKS)

    artifacts_dir = config["artifacts"]["artifacts_dir"]
    model_name = config["artifacts"]["model_name"]
    model_dir = config["artifacts"]["model_dir"]
    model_dir_path = os.path.join(artifacts_dir, model_dir)
    os.makedirs(model_dir_path, exist_ok=True)
    save_model(model, model_name, model_dir_path)

    artifacts_dir = config["artifacts"]["artifacts_dir"]
    plot_name = config["artifacts"]["plot_name"]
    plot_dir = config["artifacts"]["plots"]
    plot_dir_path = os.path.join(artifacts_dir, plot_dir)
    os.makedirs(plot_dir_path, exist_ok=True)
    save_plot(history, plot_name, plot_dir_path)


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config.yaml")

    parsed_args = args.parse_args()
    training(config_path=parsed_args.config)