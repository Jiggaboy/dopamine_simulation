import numpy as np
import matplotlib.pyplot as plt

import dopamine as DOP
from custom_class import Population
from params import StarterConfig, PerlinConfig, BaseConfig

Config = PerlinConfig()
# Sets up a new population. Either loads the connectivity matrix or sets up a new one.
neural_population = Population(Config)

patch = DOP.circular_patch(grid=Config.rows, center=(3, 3))

plt.figure()
plt.imshow(patch.reshape(Config.rows, Config.rows))

plt.figure()
from connectivitymatrix import plot_colored_shift, plot_shift_arrows, calculate_direction
shift = neural_population.shift
plot_colored_shift(shift)
plot_shift_arrows(shift)

Fx, Fy = calculate_direction(shift)
np.save("Fx", Fx)
np.save("Fy", Fy)
quit()

def plot_quivers(rows, x=None, y=None):
    X, Y = np.meshgrid(np.arange(rows), np.arange(rows))
    if x is not None and y is not None:
        plt.quiver(X, Y, x, y, pivot='middle', )

    if x is not None:
        plt.quiver(X, Y, x, np.zeros(x.shape), pivot='middle', scale=10.)
    if y is not None:
        plt.quiver(X, Y, np.zeros(y.shape), y, pivot='middle', )


plt.figure("Fx")
plot_quivers(Config.rows, x=Fx)
plt.figure("Fy")
plot_quivers(Config.rows, y=Fy)


step = 5
radius = 5

Fx_avg = np.zeros(shape=(Config.rows, Config.rows))
Fy_avg = np.zeros(shape=Fx_avg.shape)
for i in range(0, Config.rows, step):
    for j in range(0, Config.rows, step):
        print("location", i, j)
        patch = DOP.circular_patch(Config.rows, center=(i, j), radius=radius)
        Fx_avg[i, j] = Fx[patch].mean()
        Fy_avg[i, j] = Fy[patch].mean()

plt.figure("Fx_mean")
plt.imshow(Fx_avg.reshape(Config.rows, Config.rows), origin="lower")
plot_quivers(Config.rows, x = Fx_avg)
plt.figure("Fy_mean")
plot_quivers(Config.rows, y = Fy_avg)


plt.show()