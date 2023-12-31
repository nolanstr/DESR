import numpy as np
import torch


class TrainingData:
    def __init__(self, x, y):
        """
        Parameters
        ----------
        self : object [Argument]
        x : [Argument]
        y : [Argument]

        """
        self._set_training_data(x, y)

    def _set_training_data(self, x, y):
        """
        Parameters
        ----------
        self : object [Argument]
        x : [Argument]
        y : [Argument]

        """
        self.x, self.y = self._check_inputs(x, y)

    def _check_inputs(self, x, y):
        """
        Parameters
        ----------
        self : object [Argument]
        x : [Argument]
        y : [Argument]

        """
        if isinstance(x, np.ndarray):
            x = x.astype(float)
            x = torch.from_numpy(x)
            x.requires_grad = True
        if isinstance(y, np.ndarray):
            y = y.astype(float)
            y = torch.from_numpy(y)
        if y.dim() == 1:
            y = y.reshape((-1, 1))
        if x.shape[0] != y.shape[0]:
            x = x.reshape((y.shape[0], -1))
        if x.dim() == 1:
            x = x.reshape((-1, 1))
        return x, y
