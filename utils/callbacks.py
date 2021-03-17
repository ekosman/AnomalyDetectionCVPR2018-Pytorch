import logging
import os
import time
from abc import ABC, abstractmethod
import datetime
import matplotlib.pyplot as plt


class Callback(ABC):
    @abstractmethod
    def on_training_start(self, epochs):
        pass

    @abstractmethod
    def on_training_end(self, model):
        pass

    @abstractmethod
    def on_epoch_start(self, epoch_num, epoch_iterations):
        pass

    @abstractmethod
    def on_epoch_step(self, global_iteration, epoch_iteration, loss):
        pass

    @abstractmethod
    def on_epoch_end(self, loss):
        pass

    @abstractmethod
    def on_evaluation_start(self, val_iterations):
        pass

    @abstractmethod
    def on_evaluation_step(self, iteration, model_outputs, targets, loss):
        pass

    @abstractmethod
    def on_evaluation_end(self):
        pass

    @abstractmethod
    def on_training_iteration_end(self, train_loss, val_loss):
        pass


class DefaultModelCallback(Callback):
    """
    A callback that simply logs the loss for epochs during training and evaluation
    """
    def __init__(self, log_every=10, visualization_dir=None):
        """
        Args:
            log_every (iterations): logging intervals
        """
        super(DefaultModelCallback, self).__init__()
        self.visualization_dir = visualization_dir
        self.log_every = log_every
        self.epochs = 0
        self.epoch = 0
        self.epoch_iterations = 0
        self.val_iterations = 0
        self.start_time = 0
        self.train_losses = None
        self.val_loss = None

    def on_training_start(self, epochs):
        logging.info(f"Training for {epochs} epochs")
        self.epochs = epochs
        self.train_losses = []
        self.val_loss = []

    def on_training_end(self, model):
        if self.visualization_dir is not None:
            plt.figure()
            plt.xlabel('Epoch')
            plt.ylabel('Loss')

            plt.plot(range(1, self.epochs + 1), self.train_losses, label=f'Training loss')
            if len(self.val_loss) != 0:
                plt.plot(range(1, self.epochs + 1), self.val_loss, label=f'Validation loss')

            plt.savefig(os.path.join(self.visualization_dir, 'loss.png'))
            plt.close()

    def on_epoch_start(self, epoch_num, epoch_iterations):
        self.epoch = epoch_num
        self.epoch_iterations = epoch_iterations
        self.start_time = time.time()

    def on_epoch_step(self, global_iteration, epoch_iteration, loss):
        if epoch_iteration % self.log_every == 0:
            average_time = round((time.time() - self.start_time) / (epoch_iteration + 1), 3)

            loss_string = f"loss: {loss}"

            logging.info(
                f"Epoch {self.epoch}/{self.epochs}      Iteration {epoch_iteration}/{self.epoch_iterations}    {loss_string}    Time: {average_time} seconds/iteration")

    def on_epoch_end(self, loss):
        self.train_losses.append(loss)

    def on_evaluation_start(self, val_iterations):
        self.val_iterations = val_iterations

    def on_evaluation_step(self, iteration, model_outputs, targets, loss):
        if iteration % self.log_every == 0:
            logging.info(f"Iteration {iteration}/{self.val_iterations}")

    def on_evaluation_end(self):
        pass

    def on_training_iteration_end(self, train_loss, val_loss):
        train_loss_string = f"Train loss: {train_loss}"
        if val_loss:
            val_loss_string = f"Validation loss: {val_loss}"
            logging.info(f"""
============================================================================================================================
Epoch {self.epoch}/{self.epochs}     {train_loss_string}     {val_loss_string}        time: {datetime.timedelta(seconds=time.time() - self.start_time)}
============================================================================================================================
""")

        else:
            logging.info(f"""
============================================================================================================================
Epoch {self.epoch}/{self.epochs}     {train_loss_string}        time: {datetime.timedelta(seconds=time.time() - self.start_time)}
============================================================================================================================
""")


class TensorBoardCallback(Callback):
    """
    A callback that simply logs the loss for epochs during training and evaluation
    """
    def __init__(self, tb_writer):
        """
        Args:
            tb_writer: tensorboard logger instance
        """
        super(TensorBoardCallback, self).__init__()
        self.tb_writer = tb_writer
        self.epoch = 0

    def on_training_start(self, epochs):
        pass

    def on_training_end(self, model):
        pass

    def on_epoch_start(self, epoch_num, epoch_iterations):
        self.epoch = epoch_num

    def on_epoch_step(self, global_iteration, epoch_iteration, loss):
        self.tb_writer.add_scalars('Train loss (iterations)',
                                   {'Loss': loss},
                                   global_iteration)

    def on_epoch_end(self, loss):
        pass

    def on_evaluation_start(self, val_iterations):
        pass

    def on_evaluation_step(self, iteration, model_outputs, targets, loss):
        pass

    def on_evaluation_end(self):
        pass

    def on_training_iteration_end(self, train_loss, val_loss):
        if train_loss is not None:
            self.tb_writer.add_scalars("Epoch loss",
                                       {f"Loss (train)": train_loss},
                                       self.epoch)

        if val_loss is not None:
            self.tb_writer.add_scalars("Epoch loss",
                                       {f"Loss (validation)": val_loss},
                                       self.epoch)
