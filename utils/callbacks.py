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
    def __init__(self, loss_names, log_every=10, visualization_dir=None):
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
        self.loss_names = loss_names

    def on_training_start(self, epochs):
        logging.info(f"Training for {epochs} epochs")
        self.epochs = epochs
        self.train_losses = [[] for _ in range(len(self.loss_names))]
        self.val_loss = [[] for _ in range(len(self.loss_names))]

    def on_training_end(self, model):
        if self.visualization_dir is not None:
            plt.figure()
            plt.xlabel('Epoch')
            plt.ylabel('Loss')

            for losses, loss_name in zip(self.train_losses, self.loss_names):
                plt.plot(range(1, self.epochs + 1), losses, label=f'Training loss ({loss_name})')

            for losses, loss_name in zip(self.val_loss, self.loss_names):
                plt.plot(range(1, self.epochs + 1), losses, label=f'Validation loss ({loss_name})')

            plt.savefig(os.path.join(self.visualization_dir, 'loss.png'))
            plt.close()

    def on_epoch_start(self, epoch_num, epoch_iterations):
        self.epoch = epoch_num
        self.epoch_iterations = epoch_iterations
        self.start_time = time.time()

    def on_epoch_step(self, global_iteration, epoch_iteration, loss):
        if epoch_iteration % self.log_every == 0:
            average_time = round((time.time() - self.start_time) / (epoch_iteration + 1), 3)

            loss_string = "   ".join([f"{loss_name} loss: {loss_value}" for loss_name, loss_value in zip(self.loss_names, loss)])

            logging.info(
                f"Epoch {self.epoch}/{self.epochs}      Iteration {epoch_iteration}/{self.epoch_iterations}    {loss_string}    Time: {average_time} seconds/iteration")

    def on_epoch_end(self, loss):
        for loss_list, loss_value in zip(self.train_losses, loss):
            loss_list.append(loss_value)

    def on_evaluation_start(self, val_iterations):
        self.val_iterations = val_iterations

    def on_evaluation_step(self, iteration, model_outputs, targets, loss):
        if iteration % self.log_every == 0:
            logging.info(f"Iteration {iteration}/{self.val_iterations}")

    def on_evaluation_end(self):
        pass

    def on_training_iteration_end(self, train_loss, val_loss):
        train_loss_string = "   ".join(
            [f"{loss_name} train loss: {loss_value}" for loss_name, loss_value in zip(self.loss_names, train_loss)])
        if val_loss:
            val_loss_string = "   ".join(
                [f"{loss_name} validation loss: {loss_value}" for loss_name, loss_value in zip(self.loss_names, val_loss)])
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
    def __init__(self, tb_writer, loss_names):
        """
        Args:
            tb_writer: tensorboard logger instance
        """
        super(TensorBoardCallback, self).__init__()
        self.tb_writer = tb_writer
        self.epoch = 0
        self.loss_names = loss_names

    def on_training_start(self, epochs):
        pass

    def on_training_end(self, model):
        pass

    def on_epoch_start(self, epoch_num, epoch_iterations):
        self.epoch = epoch_num

    def on_epoch_step(self, global_iteration, epoch_iteration, loss):
        self.tb_writer.add_scalars('Train loss (iterations)',
                                   {loss_name: loss_value for loss_name, loss_value in zip(self.loss_names, loss)},
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
                                       {f"{loss_name} (train)": loss_value for loss_name, loss_value in zip(self.loss_names, train_loss)},
                                       self.epoch)

        if val_loss is not None:
            self.tb_writer.add_scalars("Epoch loss",
                                       {f"{loss_name} (validation)": loss_value for loss_name, loss_value in zip(self.loss_names, val_loss)},
                                       self.epoch)
