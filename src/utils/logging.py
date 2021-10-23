from utils.common import get_timestamp
import os

def get_log_path(config):
    log_dir = config["logs"]["logs_dir"]
    general_logs = config["logs"]["general_logs"]
    general_logs_path = os.path.join(log_dir, general_logs)
    os.makedirs(general_logs_path, exist_ok=True)

    unique_name = get_timestamp("log")
    log_path = os.path.join(general_logs_path, unique_name)
    print(f"saving logs at: {log_path}")

    return log_path