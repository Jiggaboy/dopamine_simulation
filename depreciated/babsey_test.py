# -*- coding: utf-8 -*-
#
# lcrn_network.py
#
# Copyright 2017 Arvind Kumar, Sebastian Spreizer
# The MIT License

import numpy as np

__all__ = [
    'lcrn_gauss_targets',
    'lcrn_gamma_targets',
    'plot_targets',
]


def lcrn_gauss_targets(s_id, srow, scol, trow, tcol, ncon, con_std, selfconnection=True):
    grid_scale = float(trow) / float(srow)
    s_x = np.remainder(s_id, scol)  # column id
    s_y = int(s_id) // int(scol)  # row id
    s_x1 = int(s_x * grid_scale)  # column id in the new grid
    s_y1 = int(s_y * grid_scale)  # row_id in the new grid

    # pick up ncol values for phi and radius
    phi = np.random.uniform(low=-np.pi, high=np.pi, size=ncon)
    radius = con_std * np.random.randn(ncon)
    if selfconnection == False:
        radius[radius>0] = radius[radius>0] + 1.
        radius[radius<0] = radius[radius<0] - 1.
    t_x = np.remainder(radius * np.cos(phi) + s_x1, tcol)
    t_y = np.remainder(radius * np.sin(phi) + s_y1, trow)
    return t_x, t_y, radius
    target_ids = np.remainder(
        np.round(t_y) * tcol + np.round(t_x), trow * tcol)
    target = np.array(target_ids).astype('int')
    delays = np.abs(radius) / tcol
    return target, delays




import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Our 2-dimensional distribution will be over variables X and Y
N = 25
lim = 2.5
x = np.linspace(-lim, lim, N)
y = np.linspace(-lim, lim, N)
X, Y = np.meshgrid(x, y)
# Mean vector and covariance matrix
mu = np.array([0., 0.])
Sigma = np.array([[ 1. , 0], [0,  1.]])

######################################## CUSTOM ################################################################
def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos."""

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return  1 / np.sqrt(np.pi * 2) * np.exp(-fac / 2) / N


# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

x = np.linspace(-lim, lim, N+1)
y = np.linspace(-lim, lim, N+1)




# V: scipy multivariate gaussian
from scipy.stats import multivariate_normal, norm
rv = multivariate_normal(mu, Sigma)
# rv = multivariate_normal()
# V = rv.pdf(pos)

test = rv.rvs(1000000)
plt.hist(np.abs(test), density=True, bins=30)
quit()

def sample_pdf_and_hist_2D_radius_distribution(rv):
    V_sample = rv.rvs(10000)
    V_sample[:, 0] += 20
    V_sample_r = np.linalg.norm(V_sample, axis=1)

    plt.figure("2D distribution of radii")
    x_1 = np.arange(-3, 3, .1)
    plt.plot(x_1, norm.pdf(x_1, 0, 1))
    plt.hist(np.random.normal(0, 1, size=10000), density=True, bins=x_1)
    plt.hist(V_sample_r-20, density=True, bins=x_1)

# sample_pdf_and_hist_2D_radius_distribution(rv)
# The radius is distributed in 2D such as a Gaussian in 1D

############################# T: my function ########################################################
from lib import lcrn_network as lcrn
lcrn_targets, radius, phi = lcrn.get_off_grid_target_positions([0, 0], 1, 1000000, True)

# plt.figure("Dist. of my func")
# x_1 = np.arange(0, 3, .1)
# plt.hist(lcrn_targets[0], density=True, bins=x_1, label="X")
# plt.hist(lcrn_targets[1], density=True, bins=x_1, label="Y")
# plt.hist(lcrn_targets.ravel(), density=True, bins=x_1, label="XY")
# plt.hist(np.random.normal(0, 1, size=10000), density=True, bins=x_1, label="norm")
# radius_tmp = np.linalg.norm(lcrn_targets, axis=0)
# plt.hist(radius_tmp, density=True, bins=x_1, label="radius")
# plt.hist(radius, density=True, bins=x_1, label="true radius")
# plt.plot(x_1, 2 * norm.pdf(x_1, 0, 1))
# plt.legend()

# plt.figure()
# plt.scatter(*lcrn_targets)
# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
# ax.scatter(phi, np.abs(radius))

# The std of target_x is lower than std of radius
T, x_edges, y_edges = np.histogram2d(*lcrn_targets, bins=(x, y), density=True)


############################ Z: Multivariate from the function from the inet ############################
# The distribution on the variables X, Y packed into pos.
Z = multivariate_gaussian(pos, mu, Sigma)

############################ H: babsey implementation ########################################################
*targets, radius = lcrn_gauss_targets(1420, 70, 70, 70, 70, 5000000, 1)
targets = np.asarray(targets)
babsey_targets = targets - 20
H, x_edges, y_edges = np.histogram2d(*babsey_targets, bins=(x, y), density=True)

V_sample = rv.rvs(1000000)
V_hist, x_edges, y_edges = np.histogram2d(*V_sample.T, bins=(x, y), density=True)

def plot(Z):
    ax2 = fig.add_subplot(projection='3d')
    con = ax2.contourf(X, Y, Z, zdir='z', offset=0, cmap=cm.viridis)
    plt.colorbar(con)
    ax2.view_init(90, 270)

    ax2.grid(False)
    ax2.set_xlabel(r'$x_1$')
    ax2.set_ylabel(r'$x_2$')

# plot using subplots
# fig = plt.figure("Z: stackoverflow")
# plot(Z)
# fig = plt.figure("H: babsey")
# plot(H)
# fig = plt.figure("V: scipy")
# plot(V)
# fig = plt.figure("V: scipy_hist")
# plot(V_hist)
# fig = plt.figure("T: me")
# plot(T)


# plt.figure("scatter samples")
# plt.scatter(*lcrn_targets, label="me")
# plt.scatter(*V_sample.T, label="scipy")
# plt.legend()
# # Looks a little bit odd such that scipy has a broader dist.

# plt.figure("hist x_targets")
# x_1 = np.arange(-3, 3, .1)
# plt.hist(lcrn_targets[0], density=True, bins=x_1, label="me")
# plt.hist(V_sample.T[0], density=True, bins=x_1, label="scipy")
# # DIFFERENCE!!!!
# plt.legend()


plt.figure("Radius")
x_2 = np.arange(0, 5, .1)
plt.hist(np.linalg.norm(lcrn_targets, axis=0), density=True, bins=x_2, label="me")
plt.hist(np.linalg.norm(V_sample, axis=1), density=True, bins=x_2, label="scipy")
plt.legend()


plt.show()
