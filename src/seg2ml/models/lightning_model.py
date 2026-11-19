from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, TypedDict, Union
import numpy as np
import pytorch_lightning as pl
import torch
from torch.optim import Optimizer
import math
from seg2ml.utils.helper import create_class_instance


class Output(TypedDict):
    loss: Optional[torch.FloatTensor]
    pred: Optional[np.ndarray]
    target: Optional[np.ndarray]


class OptimizerDict(TypedDict):
    optimizer: Optimizer
    lr_scheduler: Optional[Dict]


Loss = torch.FloatTensor
Predictions = torch.FloatTensor


class LightningModel(pl.LightningModule, ABC):
    def __init__(self, learning_rate: float = 0.1, learning_rate_scheduler_config: Optional[Dict] = None, optimizer_config: Optional[Dict] = None, weight_decay: int = 1.0e-05, parameters: Dict=None, seed: float = 0):
        super().__init__()

        self.learning_rate = learning_rate
        self.learning_rate_scheduler_config = learning_rate_scheduler_config
        self.weight_decay = weight_decay
        self.optimzer_config = optimizer_config
        self.seed = seed
        self._set_seed(self.seed)

    @staticmethod
    def _set_seed(seed):
        if seed is not None:
            torch.manual_seed(seed)
            # set seed for CUDA
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    @abstractmethod
    def forward(self, x: torch.FloatTensor, *args, **kwargs):
        """

        :param x:
        :param args:
        :param kwargs:
        :return:
        """
        ...

    @abstractmethod
    def loss(self, y_pred: torch.Tensor, y_target: torch.Tensor, *args, **kwargs) -> torch.FloatTensor:
        """

        :param y_pred:
        :param y_target:
        :param args:
        :param kwargs:
        :return:
        """
        ...

    @abstractmethod
    def training_step(
        self, batch: Tuple[torch.FloatTensor, torch.LongTensor], batch_idx: int, optimizer_idx: int, hiddens: Any
    ) -> Output:
        """

        Parameters
        ----------
        batch
        batch_idx
        optimizer_idx
        hiddens

        Returns
        -------

        """
        ...

    @abstractmethod
    def validation_step(
        self, batch: Tuple[torch.FloatTensor, torch.LongTensor], batch_idx: int, dataloader_idx: int
    ) -> Output:
        """

        Parameters
        ----------
        batch
        batch_idx
        dataloader_idx

        Returns
        -------

        """
        ...

    @abstractmethod
    def test_step(
        self, batch: Tuple[torch.FloatTensor, torch.LongTensor], batch_idx: int, dataloader_idx: int
    ) -> Output:
        """

        Parameters
        ----------
        batch
        batch_idx
        dataloader_idx

        Returns
        -------

        """
        ...

    @abstractmethod
    def predict_step(
        self, batch: Tuple[torch.FloatTensor, torch.LongTensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Output:
        """

        Parameters
        ----------
        batch
        batch_idx
        dataloader_idx

        Returns
        -------

        """
        ...

    def configure_optimizers(self) -> Union[Optimizer, OptimizerDict]:

        optimizer = create_class_instance(self.optimzer_config["module"], self.optimzer_config["name"], self.optimzer_config["args"], params=self.parameters())

        if self.learning_rate_scheduler_config is None:
            return optimizer
        learning_rate_scheduler: Dict = self.init_learning_rate_scheduler(
            self.learning_rate_scheduler_config, optimizer
        )
        return {"optimizer": optimizer, "lr_scheduler": learning_rate_scheduler}

    @staticmethod
    def init_learning_rate_scheduler(learning_rate_scheduler_config: Dict, optimizer: torch.optim.Optimizer) -> Dict:
        additional_kwargs = {}
        for key in ["interval", "frequency", "monitor", "strict"]:
            if key in learning_rate_scheduler_config["args"]:
                additional_kwargs[key] = learning_rate_scheduler_config["args"].pop(key)
        return {
            "scheduler": create_class_instance(
                learning_rate_scheduler_config["module"],
                learning_rate_scheduler_config["name"],
                learning_rate_scheduler_config["args"],
                optimizer=optimizer,
            ),
            **additional_kwargs,
        }
