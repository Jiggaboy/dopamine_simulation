import numpy as np
import matplotlib.pyplot as plt

import dopamine as DOP
from custom_class import Population
from params import StarterConfig, PerlinConfig, BaseConfig

Config = StarterConfig()
# Sets up a new population. Either loads the connectivity matrix or sets up a new one.
neural_population = Population(Config)

patch = DOP.circular_patch(grid=Config.rows, center=(3, 3))

plt.figure()
plt.imshow(patch.reshape(Config.rows, Config.rows))

plt.figure()
shift = neural_population.shift
plt.imshow(shift)

plt.show()