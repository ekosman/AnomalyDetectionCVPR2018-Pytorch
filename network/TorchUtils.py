"""Written by Eitan Kosman."""

import logging
import os
import time
from typing import List, Optional, Union

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from utils.callbacks import Callback
from utils.types import Device


def get_torch_device() -> Device:
    """
    Retrieves the device to run torch models, with preferability to GPU (denoted as cuda by torch)
    Returns: Device to run the models
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path: str) -> nn.Module:
    """Loads a Pytorch model.

    Args:
        model_path: path to the model to load

    Returns: a model loaded from the specified path
    """
    logging.info(f"Load the model from: {model_path}")
    model = torch.load(model_path, map_location="cpu")
    logging.info(model)
    return model


class TorchModel(nn.Module):
    """Wrapper class for a torch model to make it comfortable to train and load
    models."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.device = get_torch_device()
        self.iteration = 0
        self.model = model
        self.is_data_parallel = False
        self.callbacks = []

    def register_callback(self, callback_fn: Callback) -> None:
        """
        Register a callback to be called after each evaluation run
        Args:
            callback_fn: a callable that accepts 2 inputs (output, target)
                            - output is the model's output
                            - target is the values of the target variable
        """
        self.callbacks.append(callback_fn)

    def data_parallel(self):
        """Transfers the model to data parallel mode."""
        self.is_data_parallel = True
        if not isinstance(self.model, torch.nn.DataParallel):
            self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1])

        return self

    @classmethod
    def load_model(cls, model_path: str):
        """
        Loads a pickled model
        Args:
            model_path: path to the pickled model

        Returns: TorchModel class instance wrapping the provided model
        """
        return cls(load_model(model_path))

    def notify_callbacks(self, notification, *args, **kwargs) -> None:
        """Calls all callbacks registered with this class.

        Args:
            notification: The type of notification to be called.
        """
        for callback in self.callbacks:
            try:
                method = getattr(callback, notification)
                method(*args, **kwargs)
            except (AttributeError, TypeError) as e:
                logging.error(
                    f"callback {callback.__class__.__name__} doesn't fully implement the required interface {e}"  # pylint: disable=line-too-long
                )

    def fit(
        self,
        train_iter: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        eval_iter: Optional[DataLoader] = None,
        epochs: int = 10,
        network_model_path_base: Optional[str] = None,
        save_every: Optional[int] = None,
        evaluate_every: Optional[int] = None,
    ) -> None:
        """

        Args:
            train_iter: iterator for training
            criterion: loss function
            optimizer: optimizer for the algorithm
            eval_iter: iterator for evaluation
            epochs: amount of epochs
            network_model_path_base: where to save the models
            save_every: saving model checkpoints every specified amount of epochs
            evaluate_every: perform evaluation every specified amount of epochs.
                            If the evaluation is expensive, you probably want to
                            choose a high value for this
        """
        criterion = criterion.to(self.device)
        self.notify_callbacks("on_training_start", epochs)

        for epoch in range(epochs):
            train_loss = self.do_epoch(
                criterion=criterion,
                optimizer=optimizer,
                data_iter=train_iter,
                epoch=epoch,
            )

            if save_every and network_model_path_base and epoch % save_every == 0:
                logging.info(f"Save the model after epoch {epoch}")
                self.save(os.path.join(network_model_path_base, f"epoch_{epoch}.pt"))

            val_loss = None
            if eval_iter and evaluate_every and epoch % evaluate_every == 0:
                logging.info(f"Evaluating after epoch {epoch}")
                val_loss = self.evaluate(
                    criterion=criterion,
                    data_iter=eval_iter,
                )

            self.notify_callbacks("on_training_iteration_end", train_loss, val_loss)

        self.notify_callbacks("on_training_end", self.model)
        # Save the last model anyway...
        if network_model_path_base:
            self.save(os.path.join(network_model_path_base, f"epoch_{epoch + 1}.pt"))

    def evaluate(self, criterion: nn.Module, data_iter: DataLoader) -> float:
        """
        Evaluates the model
        Args:
            criterion: Loss function for calculating the evaluation
            data_iter: torch data iterator
        """
        self.eval()
        self.notify_callbacks("on_evaluation_start", len(data_iter))
        total_loss = 0

        with torch.no_grad():
            for iteration, (batch, targets) in enumerate(data_iter):
                batch = self.data_to_device(batch, self.device)
                targets = self.data_to_device(targets, self.device)

                outputs = self.model(batch)
                loss = criterion(outputs, targets)

                self.notify_callbacks(
                    "on_evaluation_step",
                    iteration,
                    outputs.detach().cpu(),
                    targets.detach().cpu(),
                    loss.item(),
                )

                total_loss += loss.item()

        loss = total_loss / len(data_iter)
        self.notify_callbacks("on_evaluation_end")
        return loss

    def do_epoch(
        self,
        criterion: nn.Module,
        optimizer: Optimizer,
        data_iter: DataLoader,
        epoch: int,
    ) -> float:
        """Perform a whole epoch.

        Args:
            criterion (nn.Module): Loss function to be used.
            optimizer (Optimizer): Optimizer to use for minimizing the loss function.
            data_iter (DataLoader): Loader for data samples used for training the model.
            epoch (int): The epoch number.

        Returns:
            float: Average training loss calculated during the epoch.
        """
        total_loss = 0
        total_time = 0.0
        self.train()
        self.notify_callbacks("on_epoch_start", epoch, len(data_iter))
        for iteration, (batch, targets) in enumerate(data_iter):
            self.iteration += 1
            start_time = time.time()
            batch = self.data_to_device(batch, self.device)
            targets = self.data_to_device(targets, self.device)

            outputs = self.model(batch)

            loss = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            end_time = time.time()

            total_time += end_time - start_time

            self.notify_callbacks(
                "on_epoch_step",
                self.iteration,
                iteration,
                loss.item(),
            )
            self.iteration += 1

        loss = total_loss / len(data_iter)

        self.notify_callbacks("on_epoch_end", loss)
        return loss

    def data_to_device(
        self, data: Union[Tensor, List[Tensor]], device: Device
    ) -> Union[Tensor, List[Tensor]]:
        """
        Transfers a tensor data to a device
        Args:
            data: torch tensor
            device: target device
        """
        if isinstance(data, list):
            data = [d.to(device) for d in data]
        elif isinstance(data, tuple):
            data = tuple([d.to(device) for d in data])
        else:
            data = data.to(device)

        return data

    def save(self, model_path: str) -> None:
        """Saves the model to the given path.

        If currently using data parallel, the method
        will save the original model and not the data parallel instance of it
        Args:
            model_path: target path to save the model to
        """
        if self.is_data_parallel:
            torch.save(self.model.module, model_path)
        else:
            torch.save(self.model, model_path)

    def get_model(self) -> nn.Module:
        if self.is_data_parallel:
            return self.model.module

        return self.model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
