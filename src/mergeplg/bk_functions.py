"""
Created on Fri Oct 18 20:21:53 2024

@author: erlend
"""

import numpy as np
import pykrige


def interpolate_neighbourhood_block_kriging(
    xgrid,
    ygrid,
    obs,
    x0,
    variogram,
    nnear,
):
    """Interpolate observations using neighbourhood block kriging

    Interpolate CML and rain gauge data using an neighbourhood based
    implementation of block kriging as outlined in Goovaerts, P. (2008).
    Kriging and Semivariogram Deconvolution in the Presence of Irregular
    Geographical Units. Mathematical Geosciences, 40, 101 - 128.
    https://doi.org/10.1007/s11004-007-9129-1

    Parameters
    ----------
    xgrid numpy.array
        x coordinates as a meshgrid
    ygrid numpy array
        y coordinates as a meshgrid
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
        Array with the same structure as xgrid and ygrid containing
        the interpolated field.

    """
    # Calculate lengths between all points within all CMLs
    lengths_within_l = within_block_l(x0)

    # Estimate covariance within blocks
    cov_within = variogram.variogram_function(
        variogram.variogram_model_parameters,
        lengths_within_l,
    ).mean(axis=(1, 2))

    # Mean covariance within block pairs
    cov_within = 0.5 * (cov_within + cov_within.reshape(-1, 1))

    # Calculate lengths between all points along all CMLs
    lengths_point_l = block_points_to_lengths(x0)

    # Average variance across blocks
    cov_block = variogram.variogram_function(
        variogram.variogram_model_parameters,
        lengths_point_l,
    ).mean(axis=(2, 3))

    # Subtract within covariance from block covariance
    cov_mat = -1 * (cov_block - cov_within)

    # Add nugget value to diagonal as subtracting cov_within_sub removes
    # block nugget
    nugget = variogram.variogram_function(
        variogram.variogram_model_parameters,
        np.array([0.0]),
    )
    np.fill_diagonal(cov_mat, nugget)

    # Create Kriging matrix
    mat = np.zeros([cov_mat.shape[0] + 1, cov_mat.shape[1] + 1])
    mat[: cov_mat.shape[0], : cov_mat.shape[1]] = cov_mat
    mat[-1, :-1] = np.ones(cov_mat.shape[1])  # non-bias condition
    mat[:-1, -1] = np.ones(cov_mat.shape[0])  # lagrange multipliers

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
        indices = np.argpartition(np.nanmin(lengths, axis=1), nnear - 1)[:nnear]
        ind_mat = np.append(indices, mat.shape[0] - 1)

        # Calc the inverse, only dependent on geometry
        # a_inv = np.linalg.pinv(mat[np.ix_(ind_mat, ind_mat)])

        # Estimate expected variance for all links
        cov_line_point = variogram.variogram_function(
            variogram.variogram_model_parameters,
            lengths[indices],
        ).mean(axis=1)

        # Subtract withinblock covariance of the blocks
        target = -1 * (cov_line_point - cov_within[indices, indices])

        # Add non bias condition
        target = np.append(target, 1)

        # Solve the kriging system
        w = np.linalg.solve(mat[np.ix_(ind_mat, ind_mat)], target)[:-1]

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
    Geographical Units. Mathematical Geosciences, 40, 101 - 128.
    https://doi.org/10.1007/s11004-007-9129-1


    Parameters
    ----------
    xgrid numpy.array
        x coordinates as a meshgrid
    ygrid numpy array
        y coordinates as a meshgrid
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
        Array with the same structure as xgrid/ygrid containing
        the interpolated field.
    """
    # Calculate lengths between all points within all CMLs
    lengths_within_l = within_block_l(x0)

    # Estimate covariance within blocks
    cov_within = variogram.variogram_function(
        variogram.variogram_model_parameters,
        lengths_within_l,
    ).mean(axis=(1, 2))

    # Mean covariance within block pairs
    cov_within = 0.5 * (cov_within + cov_within.reshape(-1, 1))

    # Calculate lengths between all points along all CMLs
    lengths_point_l = block_points_to_lengths(x0)

    # Estimate mean variogram over link geometries
    cov_block = -variogram.variogram_function(
        variogram.variogram_model_parameters,
        lengths_point_l,
    ).mean(axis=(2, 3))

    # Subtract within covariance from block covariance
    cov_mat = -1 * (cov_block - cov_within)

    # Add nugget value to diagonal as subtracting cov_within_sub removes
    # block nugget
    nugget = variogram.variogram_function(
        variogram.variogram_model_parameters,
        np.array([0.0]),
    )
    np.fill_diagonal(cov_mat, nugget)

    # Create Kriging matrix
    mat = np.zeros([cov_mat.shape[0] + 1, cov_mat.shape[1] + 1])
    mat[: cov_mat.shape[0], : cov_mat.shape[1]] = cov_mat
    mat[-1, :-1] = np.ones(cov_mat.shape[1])  # non-bias condition
    mat[:-1, -1] = np.ones(cov_mat.shape[0])  # lagrange multipliers

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
        cov_line_point = -variogram.variogram_function(
            variogram.variogram_model_parameters,
            lengths,
        ).mean(axis=1)

        # Subtract withinblock covariance of the blocks
        target = -1 * (cov_line_point - np.diag(cov_within))

        # Add non bias condition
        target = np.append(target, 1)

        # Compute the kriging weights
        w = (a_inv @ target)[:-1]

        # Estimate rainfall amounts at location i
        estimate[i] = obs @ w

    # Return dataset with interpolated values
    return estimate.reshape(xgrid.shape)


def merge_ked_blockkriging(rad_field, xgrid, ygrid, rad, obs, x0, variogram, n_closest):
    """Merge ground and radar using Kriging with external drift

    Marges the provided radar field

    Parameters
    ----------
    rad_field: numpy.array
        Gridded radar data corresponding to xgrid and ygrid.
    xgrid: numpy.array
        X-grid for radar field, as a meshgrid.
    ygrid: numpy.array
        Y-grid for the radar field, as a meshgrid.
    rad: numpy array
        Radar observations at the ground (obs) locations.
    obs: numpy.array
        Ground observations.
    x0: numpy.array
        Ground observations geometry as created by calculate_cml_geometry.
    variogram: function
        A user defined function defining the variogram. Takes a distance
        h and returns the expected variance.
    n_closest: int
        Number of closest ground observations (obs) to use for interpolation

    Returns
    -------
    interpolated_field: numpy.array
        Array with the same structure as xgrid/ygrid containing
        the interpolated field.
    """
    # Calculate lengths between all points within all CMLs
    lengths_within_l = within_block_l(x0)

    # Estimate covariance within blocks
    cov_within = variogram.variogram_function(
        variogram.variogram_model_parameters,
        lengths_within_l,
    ).mean(axis=(1, 2))

    # Mean covariance within block pairs
    cov_within = 0.5 * (cov_within + cov_within.reshape(-1, 1))

    # Array for storing merged values
    rain = np.full(xgrid.shape, np.nan)

    # Calculate lengths between all points along all CMLs
    lengths_point_l = block_points_to_lengths(x0)

    # Estimate mean variogram over link geometries
    cov_block = -variogram.variogram_function(
        variogram.variogram_model_parameters,
        lengths_point_l,
    ).mean(axis=(2, 3))

    # Subtract within covariance from block covariance
    cov_mat = -1 * (cov_block - cov_within)

    # Add nugget value to diagonal as subtracting cov_within_sub removes
    # block nugget
    nugget = variogram.variogram_function(
        variogram.variogram_model_parameters,
        np.array([0.0]),
    )
    np.fill_diagonal(cov_mat, nugget)

    # Create Kriging matrix
    mat = np.zeros([cov_mat.shape[0] + 2, cov_mat.shape[1] + 2])
    mat[: cov_mat.shape[0], : cov_mat.shape[1]] = cov_mat
    mat[-2, :-2] = np.ones(cov_mat.shape[1])  # non-bias condition
    mat[-1, :-2] = rad  # Radar drift
    mat[:-2, -2] = np.ones(cov_mat.shape[0])  # lagrange multipliers
    mat[:-2, -1] = rad  # Radar drift

    # Skip radar pixels with np.nan
    mask = np.isnan(rad_field)

    # Gridpoints to use
    xgrid_t, ygrid_t = xgrid[~mask], ygrid[~mask]
    rad_field_t = rad_field[~mask]

    # array for storing CML-radar merge
    estimate = np.zeros(xgrid_t.shape)

    # Compute the contributions from all CMLs to points in grid
    for i in range(xgrid_t.size):
        # compute target, that is R.H.S of eq 15 (jewel2013)
        delta_x = x0[:, 1] - xgrid_t[i]
        delta_y = x0[:, 0] - ygrid_t[i]
        lengths = np.sqrt(delta_x**2 + delta_y**2)

        # Get the n closest links
        indices = np.argpartition(lengths.min(axis=1), n_closest - 1)[:n_closest]
        ind_mat = np.append(indices, [mat.shape[0] - 2, mat.shape[0] - 1])

        # Calc the inverse, only dependent on geometry
        a_inv = np.linalg.pinv(mat[np.ix_(ind_mat, ind_mat)])

        # Estimate expected variance for all links
        cov_line_point = -variogram.variogram_function(
            variogram.variogram_model_parameters,
            lengths[indices],
        ).mean(axis=1)

        # Subtract withinblock covariance of the blocks
        target = -1 * (cov_line_point - cov_within[indices, indices])

        target = np.append(target, 1)  # non bias condition
        target = np.append(target, rad_field_t[i])  # radar value

        # compuite weights
        w = (a_inv @ target)[:-2]

        # its then the sum of the CML values (eq 8, see paragraph after eq 15)
        estimate[i] = obs[indices] @ w

    rain[~mask] = estimate

    return rain.reshape(xgrid.shape)


def within_block_l(x0):
    """Calculate the lengths within all CMLs.

    Given the numpy array x0 created by the function 'calculate_cml_geometry'
    this function calculates the length between all points within all CMLs.

    Parameters
    ----------
    x0: np.array
        Array with coordinates for all CMLs. The array is organized into a 3D
        matrix with the following structure:
            (number of n CMLs [0, ..., n],
             y/x-cooridnate [0(y), 1(x)],
             interval [0, ..., disc])

    Returns
    -------
    lengths_withinblock_l: np.array
        Array with lengths between all points along within the CMLs. The array
        is organized into a 4D matrix with the following structure:
            (cml_i: [0, ..., number of cmls],
             cml_i: [0, ..., number of cmls],
             length_i: [0, ..., number of points along cml].
             length_i: [0, ..., number of points along cml]).

        Accessing the length between point 0 along cml 0 and point 0 along
        cml 1 can then be done by lengths_point_l[0, 1, 0, 0]. The mean length
        can be calculated by lengths_point_l.mean(axis = (2, 3)).
    """
    # Estimate delta x within all blocks
    delta_x = np.array([x0[i][1] - x0[i][1].reshape(-1, 1) for i in range(x0.shape[0])])
    delta_y = np.array([x0[i][0] - x0[i][0].reshape(-1, 1) for i in range(x0.shape[0])])

    # Lengths within all blocks
    return np.sqrt(delta_x**2 + delta_y**2)


def block_points_to_lengths(x0):
    """Calculate the lengths between all discretized points along all CMLs.

    Given the numpy array x0 created by the function 'calculate_cml_geometry'
    this function calculates the length between all points along all CMLs.

    Parameters
    ----------
    x0: np.array
        Array with coordinates for all CMLs. The array is organized into a 3D
        matrix with the following structure:
            (number of n CMLs [0, ..., n],
             y/x-cooridnate [0(y), 1(x)],
             interval [0, ..., disc])

    Returns
    -------
    lengths_point_l: np.array
        Array with lengths between all points along the CMLs. The array is
        organized into a 4D matrix with the following structure:
            (cml_i: [0, ..., number of cmls],
             cml_i: [0, ..., number of cmls],
             length_i: [0, ..., number of points along cml].
             length_i: [0, ..., number of points along cml]).

        Accessing the length between point 0 along cml 0 and point 0 along
        cml 1 can then be done by lengths_point_l[0, 1, 0, 0]. The mean length
        can be calculated by lengths_point_l.mean(axis = (2, 3)).

    """
    # Calculate the x distance between all points
    delta_x = np.array(
        [
            x0[i][1] - x0[j][1].reshape(-1, 1)
            for i in range(x0.shape[0])
            for j in range(x0.shape[0])
        ]
    )

    # Calculate the y-distance between all points
    delta_y = np.array(
        [
            x0[i][0] - x0[j][0].reshape(-1, 1)
            for i in range(x0.shape[0])
            for j in range(x0.shape[0])
        ]
    )

    # Calculate corresponding length between all points
    lengths_point_l = np.sqrt(delta_x**2 + delta_y**2)

    # Reshape to (n_lines, n_lines, disc, disc)
    return lengths_point_l.reshape(
        int(np.sqrt(lengths_point_l.shape[0])),
        int(np.sqrt(lengths_point_l.shape[0])),
        lengths_point_l.shape[1],
        lengths_point_l.shape[2],
    )


def construct_variogram(
    obs,
    x0,
    variogram_parameters,
    variogram_model,
):
    """Construct variogram

    Construct the variogram using pykrige variogram model and parameters
    provided by user.

    Returns
    -------
    variogram: function
        Variogram function that returns the expected variance given the
        distance between observations.

    """
    # If x0 contains block data, get approximate midpoints
    if len(x0.shape) > 2:
        x0 = x0[:, :, int(x0.shape[2] / 2)]

    return pykrige.OrdinaryKriging(
        x0[:, 1],  # x-midpoint coordinate
        x0[:, 0],  # y-midpoint coordinate
        obs,
        variogram_parameters=variogram_parameters,
        variogram_model=variogram_model,
    )
