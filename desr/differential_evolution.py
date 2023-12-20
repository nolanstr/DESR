from . import Equation
import numpy as np
from tqdm import tqdm


class DifferentialEvolution:
    def __init__(self, chains, generator, bayesian_fitness, epsilon=0.05):
        """
        Parameters
        ----------
        self : object [Argument]
        chains : [Argument]
        generator : [Argument]
        bayesian_fitness : [Argument]
        epsilon :default: 0.05 [Argument]

        """
        self._chains = chains
        self._generator = generator
        self._bayesian_fitness = bayesian_fitness
        self._gammas = [2.38 / np.sqrt(generator._genotype_size), 1.0]
        self._epsilon = epsilon

        self.states = generator(chains)
        self._evaluate_initial_states()
        self._accepted_states = []

    def sample(self, iterations=100, return_states=False):
        """
        Parameters
        ----------
        self : object [Argument]
        iterations :default: 100 [Argument]
        return_states :default: False [Argument]

        """

        for i in tqdm(range(iterations), total=iterations):
            for j in range(self._chains):
                proposed_state = self.generate_proposal_state(j)
                selected_state = self.select_state(proposed_state, self.states[j])
                self._accepted_states.append(selected_state.copy())
                self.states[j] = selected_state
        self._set_unique_states()
        self._set_unique_accepted_states()
        if return_states:
            return self.states

    def select_state(self, proposed_state, current_state):
        """
        Parameters
        ----------
        self : object [Argument]
        proposed_state : [Argument]
        current_state : [Argument]

        """

        if np.isnan(proposed_state.fitness):
            return current_state
        ratio = np.exp(-proposed_state.fitness + current_state.fitness)
        alpha = min(1, ratio)
        if np.random.uniform() <= alpha:
            return proposed_state
        return current_state

    def generate_proposal_state(self, state_idx):
        """
        Parameters
        ----------
        self : object [Argument]
        state_idx : [Argument]

        """
        sample_idxs = np.random.choice(
            np.hstack((np.arange(state_idx), np.arange(state_idx, self._chains))),
            2,
            replace=False,
        )
        gamma = np.random.choice(self._gammas, p=[0.9, 0.1])

        current_genotype = self.states[state_idx].genotype.copy()
        gamma_genotype = self.states[sample_idxs[0]].genotype.copy()
        genotype2 = self.states[sample_idxs[1]].genotype.copy()
        mixed_idxs = np.random.choice(
            np.arange(self._generator._genotype_size),
            size=self._generator._genotype_size // 2,
            replace=False,
        )

        gamma_genotype[mixed_idxs, :] = genotype2[mixed_idxs, :]
        epsilon_genotype = self._generator.generate_genotype()

        gamma_replace = np.repeat(
            np.random.choice(
                [1, 0],
                size=(self._generator._genotype_size, 1),
                replace=True,
                p=[gamma, 1 - gamma],
            ),
            3,
            axis=1,
        )

        epsilon_replace = np.repeat(
            np.random.choice(
                [1, 0],
                size=(self._generator._genotype_size, 1),
                replace=True,
                p=[self._epsilon, 1 - self._epsilon],
            ),
            3,
            axis=1,
        )

        sampled_genotype = np.where(
            epsilon_replace,
            epsilon_genotype,
            np.where(gamma_replace, gamma_genotype, current_genotype),
        )
        equation = Equation(genotype=sampled_genotype)
        equation.fitness = self._bayesian_fitness(equation)

        return equation

    def _evaluate_initial_states(self):
        """
        Parameters
        ----------
        self : object [Argument]

        """
        for state in self.states:
            state.fitness = self._bayesian_fitness(state)

    def _set_unique_states(self):
        """
        Parameters
        ----------
        self : object [Argument]

        """

        def check_arrays(array1, array2):
            """
            Parameters
            ----------
            array1 : [Argument]
            array2 : [Argument]

            """
            return np.array_equal(array1, array2)

        unique_states = [self.states[0]]
        for state in self.states[1:]:
            if not np.any(
                [
                    check_arrays(
                        state._simplified_genotype, unique_state._simplified_genotype
                    )
                    for unique_state in unique_states
                ]
            ):
                unique_states.append(state)
        fits = [state.fitness for state in unique_states]
        idxs = np.flip(np.argsort(fits))
        self.unique_states = [unique_states[i] for i in idxs]

    def _set_unique_accepted_states(self):
        """
        Parameters
        ----------
        self : object [Argument]

        """

        def check_arrays(array1, array2):
            """
            Parameters
            ----------
            array1 : [Argument]
            array2 : [Argument]

            """
            return np.array_equal(array1, array2)

        unique_accepted_states = [self._accepted_states[0]]
        for state in self._accepted_states[1:]:
            if not np.any(
                [
                    check_arrays(
                        state._simplified_genotype, unique_state._simplified_genotype
                    )
                    for unique_state in unique_accepted_states
                ]
            ):
                unique_accepted_states.append(state)
        fits = [state.fitness for state in unique_accepted_states]
        idxs = np.flip(np.argsort(fits))
        self.unique_accepted_states = [unique_accepted_states[i] for i in idxs]
