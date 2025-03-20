"""
Created on Fri Oct 18 20:21:53 2024

@author: erlend
"""

import numpy as np
import pykrige
import xarray as xr
from scipy import stats

def interpolate_neighbourhood_block_kriging(
        xgrid, 
        ygrid, 
        obs, 
        x0, 
        variogram, 
        nnear,
    ):
    """Interpolate observations using neighbourhood block kriging

    Interpolate CML and rain gauge data using an neigbourhood based 
    implementation of block kriging as outlined in Goovaerts, P. (2008). 
    Kriging and Semivariogram Deconvolution in the Presence of Irregular 
    Geographical Units. Mathematical Geosciences, 40, 101–128. 
    https://doi.org/10.1007/s11004-007-9129-1

    Parameters
    ----------
    xgrid numpy.array
        x coordinates as a meshgrid
    ygrid numpy array
        y coodrinates as a meshgrid
    obs: numpy.array
        Observations to interpolate
    x0: numpy.array
        CML geometry as created by calculate_cml_geometry.
    variogram: function
        A user defined python function defining the variogram. Takes a distance
        h and returns the expected variance.
    nnear: int
        Number of neighbors to use for interpolation

    Returns
    -------
    interpolated_field: numpy.array
        Numpy array with the same structure as xgrid/ygrid containing
        the interpolated field.

    """

    # Calculate lengths between all points along all CMLs
    lengths_point_l = merge_functions.block_points_to_lengths(x0)

    # Estimate mean variogram over link geometries
    cov_block = variogram(lengths_point_l).mean(axis=(2, 3))

    # Create Kriging matrix
    mat = np.zeros([cov_block.shape[0] + 1, cov_block.shape[1] + 1])
    mat[: cov_block.shape[0], : cov_block.shape[1]] = cov_block
    mat[-1, :-1] = np.ones(cov_block.shape[1])  # non-bias condition
    mat[:-1, -1] = np.ones(cov_block.shape[0])  # lagrange multipliers

    # Grid to visit
    xgrid_t, ygrid_t = xgrid.ravel(), ygrid.ravel()

    # array for storing CML-radar merge
    estimate = np.zeros(xgrid_t.shape)

    # Compute the contributions from nearby CMLs to points in grid
    for i in range(xgrid_t.size):
        # Compute lengths between all points along all links
        delta_x = x0[:, 1] - xgrid_t[i]
        delta_y = x0[:, 0] - ygrid_t[i]
        lengths = np.sqrt(delta_x**2 + delta_y**2)

        # Get the n closest links
        indices = np.argpartition(np.nanmin(lengths, axis=1), nnear)[:nnear]
        ind_mat = np.append(indices, mat.shape[0] - 1)

        # Calc the inverse, only dependent on geometry
        a_inv = np.linalg.pinv(mat[np.ix_(ind_mat, ind_mat)])

        # Estimate expected variance for all links
        target = variogram(lengths[indices]).mean(axis=1)

        # Add non bias condition
        target = np.append(target, 1)

        # Compute the kriging weights
        w = (a_inv @ target)[:-1]

        # Estimate rainfall amounts at location i
        estimate[i] = obs[indices] @ w

    # Return dataset with interpolated values
    return estimate.reshape(xgrid.shape)

def interpolate_block_kriging(
        xgrid, 
        ygrid, 
        obs, 
        x0, 
        variogram, 
    ):
    """Interpolate observations using block kriging

    Interpolate CML and rain gauge data using an implementation of 
    block kriging as outlined in Goovaerts, P. (2008). Kriging and 
    Semivariogram Deconvolution in the Presence of Irregular 
    Geographical Units. Mathematical Geosciences, 40, 101–128. 
    https://doi.org/10.1007/s11004-007-9129-1


    Parameters
    ----------
    xgrid numpy.array
        x coordinates as a meshgrid
    ygrid numpy array
        y coodrinates as a meshgrid
    obs: numpy.array
        Observations to interpolate
    x0: numpy.array
        CML geometry as created by calculate_cml_geometry.
    variogram: function
        A user defined python function defining the variogram. Takes a distance
        h and returns the expected variance.

    Returns
    -------
    interpolated_field: numpy.array
        Numpy array with the same structure as xgrid/ygrid containing
        the interpolated field.
    """
    # Calculate lengths between all points along all CMLs
    lengths_point_l = merge_functions.block_points_to_lengths(x0)

    # Estimate mean variogram over link geometries
    cov_block = variogram(lengths_point_l).mean(axis=(2, 3))

    # Create Kriging matrix
    mat = np.zeros([cov_block.shape[0] + 1, cov_block.shape[1] + 1])
    mat[: cov_block.shape[0], : cov_block.shape[1]] = cov_block
    mat[-1, :-1] = np.ones(cov_block.shape[1])  # non-bias condition
    mat[:-1, -1] = np.ones(cov_block.shape[0])  # lagrange multipliers
    
    # Invert the kriging matrix
    a_inv = np.linalg.pinv(mat)

    # Grid to visit
    xgrid_t, ygrid_t = xgrid.ravel(), ygrid.ravel()

    # array for storing CML-radar merge
    estimate = np.zeros(xgrid_t.shape)

    # Compute the contributions from all CMLs to points in grid
    for i in range(xgrid_t.size):
        delta_x = x0[:, 1] - xgrid_t[i]
        delta_y = x0[:, 0] - ygrid_t[i]
        lengths = np.sqrt(delta_x**2 + delta_y**2)

        # Estimate expected variance for all links
        target = variogram(lengths).mean(axis=1)

        # Add non bias condition
        target = np.append(target, 1)

        # Compute the kriging weights
        w = (a_inv @ target)[:-1]

        # Estimate rainfall amounts at location i
        estimate[i] = obs @ w

    # Return dataset with interpolated values
    return estimate.reshape(xgrid.shape)

