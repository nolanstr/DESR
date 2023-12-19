import numpy as np
import torch
from scipy.stats import norm


class LaplaceApproximation:
    def __init__(self, optimizer):
        """
        Parameters
        ----------
        self : object [Argument]
        optimizer : [Argument]

        """
        self._optimizer = optimizer
        self._b = 1 / np.sqrt(optimizer._training_data.x.shape[0])

    def __call__(self, equation):
        """
        Parameters
        ----------
        self : object [Argument]
        equation : [Argument]

        """
        self._optimizer(equation)
        f, df_dx = equation.evaluate_equation_derivative_wrt_x(
            self._optimizer._training_data.x
        )
        f, df_dc = equation.evaluate_equation_derivative_wrt_c(
            self._optimizer._training_data.x
        )
        R = torch.squeeze(self._optimizer._training_data.y - f, dim=0)
        n, p = R.shape[0], equation.constants.shape[0]
        #var = torch.matmul(R.T, R) / (n - p)
        std = self._optimizer._training_data.y.std().item()
        log_likelihood = np.sum(
            np.log(norm(loc=0, scale=std).pdf(R.detach().numpy()))
        )
        normalized_marginal_log_like = (1 - self._b) * log_likelihood + (
            p / 2
        ) * np.log(self._b)
        return -normalized_marginal_log_like
