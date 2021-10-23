import os.path
from utils.common import read_config
from utils.data_mgmt import get_data
from utils.model import create_model, save_model, save_plot
from utils.logging import get_log_path
from utils.callbacks import get_callbacks
import argparse
import tensorflow as tf
import logging



def training(config_path):
    config = read_config(config_path)

    logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
    GENERAL_LOG_PATH = get_log_path(config)
    logging.basicConfig(filename=GENERAL_LOG_PATH, level=logging.INFO, format=logging_str, filemode="a")

    validation_datasize = config["params"]["validation_datasize"]
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_data(validation_datasize)
    LOSS_FUNCTION = config["params"]["loss_function"]
    OPTIMIZER = config["params"]["optimizer"]
    METRICS = config["params"]["metrics"]
    NUM_CLASSES = config["params"]["num_classes"]

    model = create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES)

    EPOCHS = config["params"]["epochs"]
    VALIDATION = (X_valid, y_valid)
    CALLBACKS = get_callbacks(config, X_train)
    try:
        logging.info(">>> Training Started >>>")
        history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=VALIDATION, callbacks=CALLBACKS)
        logging.info(">>> Training Completed >>>")
        logging.info(history.history)
    except Exception as e:
        logging.exception(e)

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