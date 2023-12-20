import numpy as np
import scipy as sp


def mean_absolute_error(equation_output, y):
    """
    Parameters
    ----------
    equation_output : [Argument]
    y : [Argument]

    """
    return np.mean(np.abs(equation_output - y))


METRIC_DICT = {"mae": mean_absolute_error}


class Optimizer:
    def __init__(self, training_data, metric="mae", bounds=[0, 1]):
        """
        Parameters
        ----------
        self : object [Argument]
        training_data : [Argument]
        metric :default: 'mae' [Argument]
        bounds :default: 0, 1 [Argument]

        """
        self.metric = metric
        self._metric_function = METRIC_DICT[metric]
        self._training_data = training_data
        self._bounds = bounds

    def __call__(self, equation):
        """
        Parameters
        ----------
        self : object [Argument]
        equation : [Argument]

        """
        params0 = np.random.uniform(
            low=self._bounds[0], high=self._bounds[1], size=equation.number_of_constants
        )
        res = sp.optimize.minimize(
            lambda x: self._fitness_function(equation, x),
            params0,
            method="Nelder-Mead",
            tol=1e-6,
        )
        params = res.x
        equation.set_constants(params)
        fitness = self._fitness_function(equation, params)
        equation.fitness = fitness

    def _fitness_function(self, equation, constants):
        """
        Parameters
        ----------
        self : object [Argument]
        equation : [Argument]
        constants : [Argument]

        """
        equation.set_constants(constants)
        equation_output = equation.evaluate_equation(self._training_data.x)
        return self._metric_function(
            equation_output.detach().numpy(), self._training_data.y.detach().numpy()
        )
