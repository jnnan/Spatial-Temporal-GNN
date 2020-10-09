from torch.utils.tensorboard import SummaryWriter
import time
import os


class ResultWriter:
    def __init__(self, log_on):
        str_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        self.model_folder = f"results/{str_time}/save_models"
        tensorboard_folder = f"results/{str_time}/runs"
        if log_on:
            os.makedirs(self.model_folder, exist_ok=True)
            os.makedirs(tensorboard_folder, exist_ok=True)
            self.writer = SummaryWriter(tensorboard_folder)
        self.log_on = log_on

    def add_scalar(self, path, scaler, step):
        if self.log_on:
            self.writer.add_scalar(path, scaler, step)

    def flush(self):
        if self.log_on:
            self.writer.flush()

    def close(self):
        if self.log_on:
            self.writer.close()
