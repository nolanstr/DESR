import numpy as np
import matplotlib.pyplot as plt
import sys;sys.path.append("../")
from desr import Generator, Equation, TrainingData, \
                 Optimizer, LaplaceApproximation, DifferentialEvolution
X = np.random.uniform(0, 4, 100)
X.sort()
y = np.sqrt(X) 
std_dev = y.std()/2
y_noisy = y + np.random.normal(loc=0, scale=std_dev, size=X.shape)
training_data = TrainingData(x=X, y=y_noisy)

plt.plot(X, y, label="Noiseless Data", color="k")
plt.scatter(X, y_noisy, label="Noisy Data", color="b")
plt.xlabel(r"$X$")
plt.ylabel(r"$y$")
plt.legend()
optimizer = Optimizer(training_data)
generator = Generator(genotype_size=16, X_dim=training_data.x.shape[1])
operators = ["add", "sub", "mult", "sqrt", "pow", "log", "sin", "pow"]
generator.add_operators(operators)

laplace_approximation = LaplaceApproximation(optimizer)
differential_evolution = DifferentialEvolution(chains=20, generator=generator,
                            bayesian_fitness=laplace_approximation)

equation = Equation(expression="sin(log(X_0)) + X_0 - 1.5")
print(f"Target Fitness: {laplace_approximation(equation)}")

differential_evolution.sample()
import pdb;pdb.set_trace()
