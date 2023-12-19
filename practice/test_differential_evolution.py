import numpy as np

from desr import Generator, Equation, TrainingData, \
                 Optimizer, LaplaceApproximation, DifferentialEvolution

equation = Equation(expression="1.0 * X_0")
X = np.linspace(0, 3*np.pi/2, 100)
y = np.sin(X) + 2.0 + np.random.normal(loc=0, scale=0.2, size=X.shape)

training_data = TrainingData(x=X, y=y)
optimizer = Optimizer(training_data)
generator = Generator(genotype_size=16)
generator.add_operator("add")
generator.add_operator("sub")
generator.add_operator("mult")
generator.add_operator("log")
generator.add_operator("sin")

fitness = optimizer(equation)
laplace_approximation = LaplaceApproximation(optimizer)

differential_evolution = DifferentialEvolution(chains=20, generator=generator,
                            bayesian_fitness=laplace_approximation)

equation = Equation(expression="sin(X_0) + 2.0")
print(laplace_approximation(equation))
states = differential_evolution.sample(iterations=100)

import pdb;pdb.set_trace()
