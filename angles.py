#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary:
    This script generates preferred directions with a better distribution of clusters.
    Perlin Noise or Simplex noise tend to have clusters of always the same direction (troughs and peaks in correlation are closer in space than the in-between regions).

"""
#===============================================================================
# PROGRAM METADATA
#===============================================================================
__author__ = 'Hauke Wernecke'
__contact__ = 'hower@kth.se'
__version__ = '0.1'

#===============================================================================
# IMPORT STATEMENTS
#===============================================================================

import numpy as np
import math
import matplotlib.pyplot as plt
import itertools
from dataclasses import dataclass

plt.rcParams['figure.figsize'] = 8, 8
#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================

REPEAT_X = 8
REPEAT_Y = REPEAT_X

sin_piover3 = np.sin(np.pi / 3)
sin_piover6 = np.sin(np.pi / 6)

@dataclass
class Hexagon:
    center: np.ndarray
    length: float = 1

    def __post_init__(self):
        print(self.center)
        self._calc_coordinates()

    @property
    def coordinates(self):
        return self.bottom, self.bottom_left, self.top_left, self.top, self.top_right, self.bottom_right

    @property
    def neighbour_vertical_shift(self):
        return 2 * self.length * sin_piover3

    @property
    def neighbour_horizontal_shift(self):
        return self.length + self.length * sin_piover6


    def _calc_coordinates(self):
        self.bottom = np.asarray([self.center[0], self.center[1] - self.length])
        self.top = np.asarray([self.center[0], self.center[1] + self.length])

        vertical_shift = self.length * .5 # self.length * np.sin(np.pi / 6)
        horizontal_shift = self.length * np.sin(np.pi / 3)

        self.bottom_left = np.asarray([self.center[0] - horizontal_shift, self.center[1] - vertical_shift])
        self.top_left = np.asarray([self.center[0] - horizontal_shift, self.center[1] + vertical_shift])
        self.bottom_right = np.asarray([self.center[0] + horizontal_shift, self.center[1] - vertical_shift])
        self.top_right = np.asarray([self.center[0] + horizontal_shift, self.center[1] + vertical_shift])


    def is_inside(self, xy:np.ndarray):
        xy = np.asarray(xy)
        distances = np.empty(len(self.coordinates))
        for i, coor in enumerate(self.coordinates):
            distances[i] = np.linalg.norm(coor - xy)
        return (distances < 2 * self.length).all()


    def add_left_neighbour(self):
        if not hasattr(self, "_left_neighbour"):
            shifted_center = np.asarray([self.center[0] - self.neighbour_vertical_shift, self.center[1]])
            self._left_neighbour = Hexagon(center=shifted_center, length=self.length)

    def add_right_neighbour(self):
        if not hasattr(self, "_right_neighbour"):
            shifted_center = np.asarray([self.center[0] + self.neighbour_vertical_shift, self.center[1]])
            self._right_neighbour = Hexagon(center=shifted_center, length=self.length)


    def add_bottom_left_neighbour(self):
        if not hasattr(self, "_bottom_left_neighbour"):
            shifted_center = np.asarray([self.center[0] - self.neighbour_vertical_shift / 2, self.center[1] - self.neighbour_horizontal_shift])
            self._bottom_left_neighbour = Hexagon(center=shifted_center, length=self.length)

    def add_bottom_right_neighbour(self):
        if not hasattr(self, "_bottom_right_neighbour"):
            shifted_center = np.asarray([self.center[0] + self.neighbour_vertical_shift / 2, self.center[1] - self.neighbour_horizontal_shift])
            self._bottom_right_neighbour = Hexagon(center=shifted_center, length=self.length)


    def add_top_left_neighbour(self):
        if not hasattr(self, "_top_left_neighbour"):
            shifted_center = np.asarray([self.center[0] - self.neighbour_vertical_shift / 2, self.center[1] + self.neighbour_horizontal_shift])
            self._top_left_neighbour = Hexagon(center=shifted_center, length=self.length)

    def add_top_right_neighbour(self):
        if not hasattr(self, "_top_right_neighbour"):
            shifted_center = np.asarray([self.center[0] + self.neighbour_vertical_shift / 2, self.center[1] + self.neighbour_horizontal_shift])
            self._top_right_neighbour = Hexagon(center=shifted_center, length=self.length)


# class Angle(Hexagon):

#     def __post_init__(self):
#         super().__post_init__()
#         # from bottom, bottom_left, top_left, top, top_right, bottom_right
#         self.angles = np.full(shape=6, fill_value=None, dtype=float)

#     def set_random_angles(self):
#         for idx in np.ndindex(self.angles.shape):
#             if self.angles[idx] is None:
#                 self.angles[idx] = np.random.uniform(-np.pi, high=np.pi)

#     def add_left_neighbour(self):
#         super().add_left_neighbour()
#         self._left_neighbour.angles[5] = self.angles[1]
#         self._left_neighbour.angles[4] = self.angles[2]

#         if hasattr(self._top_left_neighbour, name)





def main():
    # Center along x axis share the integer coordinates
    length = .5 / sin_piover3

    plt.figure()
    hexs = Hexagon(np.asarray((0, 0)), length)
    for c in hexs.coordinates:
        plt.scatter(*c, marker="o", c="k")
    points = [
        [0, 0],
        [1, 1],
        [.5, .5],
        [.25, .25],
    ]
    for p in points:
        print(p, hexs.is_inside(p))

    hexs.add_left_neighbour()
    for c in hexs._left_neighbour.coordinates:
        plt.scatter(*c, marker="o", c="r")
    hexs.add_right_neighbour()
    for c in hexs._right_neighbour.coordinates:
        plt.scatter(*c, marker="o", c="g")

    hexs.add_bottom_left_neighbour()
    for c in hexs._bottom_left_neighbour.coordinates:
        plt.scatter(*c, marker="o", c='#2ca02c')
    hexs.add_bottom_right_neighbour()
    for c in hexs._bottom_right_neighbour.coordinates:
        plt.scatter(*c, marker="o", c="y")

    hexs.add_top_left_neighbour()
    for c in hexs._top_left_neighbour.coordinates:
        plt.scatter(*c, marker="o", c='#1f77b4')
    hexs.add_top_right_neighbour()
    for c in hexs._top_right_neighbour.coordinates:
        plt.scatter(*c, marker="o", c='#ff7f0e')

    SIDE = 4
    num = 40
    grid_points = np.linspace(0, SIDE, num=num, endpoint=True)
    x_points, y_points = np.meshgrid(grid_points, grid_points)




def main_2D():
    seed = np.random.randint(0, 1000)
    # seed = 33
    # seed = 724
    print("seed:", seed)
    np.random.seed(seed)
    grid = (REPEAT_X, REPEAT_Y)
    grid_angles = np.random.uniform(-np.pi, high=np.pi, size=grid)
    # grid_angles_factor = grid_angles.sum() % (2 * np.pi)
    # grid_angles /= grid_angles_factor * 2 * np.pi
    num = 60
    grid_points = np.linspace(0, REPEAT_X, num=num, endpoint=True)
    x_points, y_points = np.meshgrid(grid_points, grid_points)

    avg_angles = np.empty(shape=(num, num))
    for c, column in enumerate(grid_points):
        for r, row in enumerate(grid_points):
            point = column % REPEAT_X, row % REPEAT_Y
            # print(column, row, point)

            a = get_angle_grid(point)
            angles = np.empty(shape=a.shape)
            for idx in np.ndindex(a.shape):
                # idx_tmp = [i % repeat for i, repeat in zip(idx, grid)]
                a_tmp = a[idx]
                a_tmp = [i % repeat for i, repeat in zip(a_tmp, grid)]
                angles[idx] = grid_angles[tuple(a_tmp)]

            d = get_distances(point, a)
            d = np.asarray(d)
            factor = 1
            weights = np.exp(-factor*(d**2))

            avg_angle = average_weighted_angles(angles.ravel(), weights.ravel())
            avg_angles[r, c] = avg_angle

            # if r ==6 and (c == 3 or c == 4) :
            if r == 178 and (c == 23 or c == 4) :
                plt.figure(f"{r}, {c}, weights")
                plt.imshow(weights, vmin=0, cmap="hot")
                plt.figure(f"{r}, {c}, angles")
                plt.imshow(angles, vmin=-np.pi, vmax=np.pi, cmap="hsv")


            # print(c)
            # if c == 51:
            # # plt.figure(c)
            # # im = plt.imshow(weights)
            # # plt.colorbar(im)
            # break
    plt.figure("angles")
    im = plt.imshow(grid_angles, vmin=-np.pi, vmax=np.pi, cmap="hsv", extent=(-.5, REPEAT_X-.5, -.5, REPEAT_Y-.5), origin="lower")
    plt.colorbar(im)

    # for i in range(2):
    #     plt.figure(f"angles_diff_{i}")
    #     diff = np.diff(avg_angles, axis=i)
    #     diff[diff > 2*np.pi] %= 2 * np.pi
    #     diff[diff < -2*np.pi] %= 2 * np.pi
    #     im = plt.imshow(diff, vmin=-np.pi, vmax=np.pi, cmap="hsv", extent=(0, REPEAT_X, 0, REPEAT_Y), origin="lower")
    #     plt.colorbar(im)
    gradient = np.gradient(avg_angles)
    for grad in gradient:
        plt.figure()
        plt.imshow(grad % np.pi, cmap="hsv", origin="lower")
        plt.colorbar()

    plt.figure("angles_fine_grid")
    im = plt.imshow(avg_angles, vmin=-np.pi, vmax=np.pi, cmap="hsv", origin="lower")
    plt.colorbar(im)


    directions = 8
    direction_matrix = np.zeros(shape=avg_angles.size, dtype=int)

    a = np.argsort(avg_angles.ravel())
    no_per_direction = avg_angles.size // directions

    ## Approach 1:
    # Binning into the directions
    for direction in np.arange(directions):
        # Find these index which correspond to the (lowest) quantile and assign the direction 0 to it.
        idx_of_no_per_direction = a[direction * no_per_direction:(direction + 1) * no_per_direction]
        direction_matrix[idx_of_no_per_direction] = direction

    direction_matrix = direction_matrix.reshape(avg_angles.shape)
    plt.figure()
    im = plt.imshow(direction_matrix, cmap="hsv")
    plt.colorbar(im)


    plt.show()



def get_angle_grid(point:tuple, length:int=4):
    a = np.empty((length, length), dtype=object)
    ab = np.empty((length, length), dtype=object)
    for x in np.arange(length):
        for y in np.arange(length):
            ab[x, y] = int(np.floor(point[0])) - 1 + x, int(np.floor(point[1])) - 1 + y
            a[x, y] = int(np.floor(point[0])) - REPEAT_X + x, int(np.floor(point[1])) - REPEAT_Y + y
    a = np.roll(a, 1, axis=[0, 1])
    return ab


    a00 = tuple(np.floor(point).astype(int))
    a10 = int(np.ceil(point[0])), int(np.floor(point[1]))
    a01 = int(np.floor(point[0])), int(np.ceil(point[1]))
    a11 = tuple(np.ceil(point).astype(int))
    return a00, a10, a01, a11
    a = np.zeros((2, 2, 2), dtype=int)
    # set x positions
    a[:, 0, 0] = np.floor(point[0]).astype(int)
    a[:, 1, 0] = np.ceil(point[0]).astype(int)
    a[0, :, 1] = np.floor(point[1]).astype(int)
    a[1, :, 1] = np.ceil(point[1]).astype(int)
    return a

def get_distances(point:tuple, coordinates:np.ndarray=None):
    d = np.empty(shape=coordinates.shape)
    for r, row in enumerate(coordinates):
        for c, coor in enumerate(row):
            d[r, c] = np.linalg.norm(np.asarray(point) - np.asarray(coor), ord=2)
    return d
    point = np.asarray(point)
    in_grid = point % 1
    d00 = np.linalg.norm(in_grid, ord=2)
    d11 = np.linalg.norm(1 - in_grid, ord=2)
    c0_r1 = in_grid[0], 1 - in_grid[1]
    c1_r0 = 1 - in_grid[0], in_grid[1]
    d10 = np.linalg.norm(c0_r1, ord=2)
    d01 = np.linalg.norm(c1_r0, ord=2)
    return d00, d10, d01, d11



def main_1D():

    grid = 20
    N = 1000
    positions = np.linspace(0, grid-1, N)

    angles = np.random.uniform(-np.pi, high=np.pi, size=grid)
    print(angles)

    angular_gradient = np.empty(shape=N)

    for idx, p in enumerate(positions):
        # find closest angles
        low = int(np.floor(p))
        high = int(np.ceil(p))

        if low == high:
            angular_gradient[idx] = angles[low]
            continue

        strength = np.asarray([1 - p % 1, (p % 1)]) ** 1.41
        tmp_angles = np.asarray([angles[low], angles[high]])
        avg_angle = average_weighted_angles(tmp_angles, strength)
        # avg_angle = average_angles(tmp_angles, strength)
        angular_gradient[idx] = avg_angle

    # im = plt.imshow(angular_gradient[:, None].T, vmin=0, vmax=2*np.pi, cmap="hsv")
    # plt.colorbar(im)
    plt.plot(positions, angular_gradient)
    plt.plot(angles, marker="*", ls="None")




#===============================================================================
# METHODS
#===============================================================================

def plot_weight_ratio(factor:float=1.):
    X_max = 2
    x = np.linspace(0, X_max, 100)
    weight_1 = np.exp(-factor*(x**2)) - np.exp(-factor*(np.sqrt(3)**2))
    plt.figure(f"{factor}")
    plt.plot(x, weight_1)
    plt.plot(x, weight_1[::-1])
    plt.plot(x, weight_1 / weight_1[::-1])
    plt.axvline(.5 * X_max, c="k")
    plt.axhline(1, c="k")
    plt.ylim(0, 10)


# Function to average weighted sum of angles
def average_weighted_angles(angles, weights):
    # Convert angles to unit vectors
    unit_vectors = [(math.cos(angle), math.sin(angle)) for angle in angles]

    # Calculate weighted sum of unit vectors
    weighted_sum = [weights[i] * unit_vectors[i][0] for i in range(len(angles))], [weights[i] * unit_vectors[i][1] for i in range(len(angles))]

    # Calculate total weight
    total_weight = sum(weights)

    # Calculate average unit vector
    average_unit_vector = (sum(weighted_sum[0]) / total_weight, sum(weighted_sum[1]) / total_weight)

    # Convert average unit vector to angle
    average_angle = math.atan2(average_unit_vector[1], average_unit_vector[0])

    return average_angle



if __name__ == '__main__':
    # main()
    main_2D()
    # plot_weight_ratio(factor=.5)
    # plot_weight_ratio(factor=1)
    # plot_weight_ratio(factor=2)
