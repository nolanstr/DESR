import numpy as np

from desr import Generator, Equation, TrainingData, \
        Optimizer, LaplaceApproximation, DifferentialEvolution

X = np.linspace(0, np.pi, 100)
y = np.sin(X) + 2.0 + np.random.normal(loc=0, scale=0.2, size=X.shape)

training_data = TrainingData(x=X, y=y)
generator = Generator()
optimizer = Optimizer(training_data)
laplace_approximation = LaplaceApproximation(optimizer)
model1 = Equation(expression="sin(X_0) + 1.0")
model2 = Equation(expression="X_0**2 + 1.0")
model3 = Equation(expression="X_0 ^ 9")
print(model3)

print(str(model1))
model1.fitness = laplace_approximation(model1)
print(model1.fitness)
model2.fitness = laplace_approximation(model2)
print(model2.fitness)
model3.fitness = laplace_approximation(model2)
print(model3.fitness)

differential_evolution = DifferentialEvolution(10, generator, 
                                            laplace_approximation)
selected_state = differential_evolution.select_state(model1, model2)
import pdb;pdb.set_trace()
