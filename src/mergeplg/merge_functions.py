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

def merge_ked_blockkriging(
        rad_field,
        xgrid,
        ygrid,
        rad, 
        obs, 
        x0, 
        variogram, 
        n_closest
    ):
    """Merge ground and radar using Kriging with external drift

    Marges the provided radar field 

    Parameters
    ----------
    rad_field: numpy.array
        Gridded radar data curresponding to xgrid and ygrid.
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
        Numpy array with the same structure as xgrid/ygrid containing
        the interpolated field.
    """
    # Array for storing merged values
    rain = np.full(xgrid.shape, np.nan)

    # Calculate lengths between all points along all CMLs
    lengths_point_l = block_points_to_lengths(x0)

    # Estimate mean variogram over link geometries
    cov_block = variogram(lengths_point_l).mean(axis=(2, 3))

    mat = np.zeros([cov_block.shape[0] + 2, cov_block.shape[1] + 2])
    mat[: cov_block.shape[0], : cov_block.shape[1]] = cov_block
    mat[-2, :-2] = np.ones(cov_block.shape[1])  # non-bias condition
    mat[-1, :-2] = rad  # Radar drift
    mat[:-2, -2] = np.ones(cov_block.shape[0])  # lagrange multipliers
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
        indices = np.argpartition(lengths.min(axis=1), n_closest)[:n_closest]
        ind_mat = np.append(indices, [mat.shape[0] - 2, mat.shape[0] - 1])

        # Calc the inverse, only dependent on geometry
        a_inv = np.linalg.pinv(mat[np.ix_(ind_mat, ind_mat)])

        target = variogram(lengths[indices]).mean(axis=1)

        target = np.append(target, 1)  # non bias condition
        target = np.append(target, rad_field_t[i])  # radar value

        # compuite weights
        w = (a_inv @ target)[:-2]

        # its then the sum of the CML values (eq 8, see paragraph after eq 15)
        estimate[i] = obs[indices] @ w

    rain[~mask] = estimate

    return rain.reshape(xgrid.shape)


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


def calculate_cml_line(ds_cmls, discretization=8):
    """Calculate the position of points along CMLs.

    Calculates the discretized CML line coordinates by dividing the CMLs into
    discretization-number of intervals. The ds_cmls xarray object must contain the
    projected coordinates (site_0_x, site_0_y, site_1_x site_1_y) defining
    the start and end point of the CML. If no such projection is available
    the user can, as an approximation, rename the lat/lon coordinates so that
    they are accepted into this function. Beware that for lat/lon coordinates
    the line geometry is not perfectly represented.

    Parameters
    ----------
    ds_cmls: xarray.Dataset
        CML geometry as a xarray object. Must contain the coordinates
        (site_0_x, site_0_y, site_1_x site_1_y)
    discretization: int
        Number of intervals to discretize lines into.

    Returns
    -------
    x0: xr.DataArray
        Array with coordinates for all CMLs. The array is organized into a 3D
        matrix with the following structure:
            (number of n CMLs [0, ..., n],
             y/x-cooridnate [0(y), 1(x)],
             interval [0, ..., discretization])

    """
    # Calculate discretized positions along the lines, store in numy array
    xpos = np.zeros([ds_cmls.cml_id.size, discretization + 1])  # shape (line, position)
    ypos = np.zeros([ds_cmls.cml_id.size, discretization + 1])

    # For all CMLs
    for block_i, cml_id in enumerate(ds_cmls.cml_id):
        x_a = ds_cmls.sel(cml_id=cml_id).site_0_x.data
        y_a = ds_cmls.sel(cml_id=cml_id).site_0_y.data
        x_b = ds_cmls.sel(cml_id=cml_id).site_1_x.data
        y_b = ds_cmls.sel(cml_id=cml_id).site_1_y.data

        # for all dicretization steps in link estimate its place on the grid
        for i in range(discretization + 1):
            xpos[block_i, i] = x_a + (i / discretization) * (x_b - x_a)
            ypos[block_i, i] = y_a + (i / discretization) * (y_b - y_a)

    # Store x and y coordinates in the same array (n_cmls, y/x, discretization)
    x0_cml = np.array([ypos, xpos]).transpose([1, 0, 2])

    # Turn into xarray dataarray and return
    return xr.DataArray(
        x0_cml,
        coords={
            "cml_id": ds_cmls.cml_id.data,
            "yx": ["y", "x"],
            "discretization": np.arange(discretization + 1),
        },
    )


def calculate_cml_midpoint(da_cml):
    """Calculate DataArray with midpoints of CMLs

    Calculates the CML midpoints and stores the results in an xr.DataArray.
    The da_cml xarray object must contain the projected coordinates (site_0_x,
    site_0_y, site_1_x site_1_y) defining the start and end point of the CML.

    Parameters
    ----------
    da_cml: xarray.DataArray
        CML geometry as a xarray object. Must contain the coordinates
        (site_0_x, site_0_y, site_1_x site_1_y)

    Returns
    -------
    x0: xr.DataArray
        Array with midpoints for all CMLs. The array is organized into a 2D
        matrix with the following structure:
            (number of n CMLs [0, ..., n],
             y/x-cooridnate [y, x],
    """
    # CML midpoint coordinates as columns
    x = ((da_cml.site_0_x + da_cml.site_1_x) / 2).data
    y = ((da_cml.site_0_y + da_cml.site_1_y) / 2).data

    # CML midpoint coordinates as columns
    x0_cml = np.hstack([y.reshape(-1, 1), x.reshape(-1, 1)])

    # Create dataarray and return
    return xr.DataArray(
        x0_cml,
        coords={
            "cml_id": da_cml.cml_id.data,
            "yx": ["y", "x"],
        },
    )


def calculate_gauge_midpoint(da_gauge):
    """Calculate DataArray with coordinates of raingauge

    Calculates the gauge coordinates and stores the results in an xr.DataArray.
    The da_gauge xarray object must contain the projected coordinates (y, x)
    defining the position of the raingauge.

    Parameters
    ----------
    da_gauge: xarray.DataArray
        Gauge coordinate as a xarray object. Must contain the coordinates (y, x)

    Returns
    -------
    x0: xr.DataArray
        Array with coordinates for all gauges. The array is organized into a 2D
        matrix with the following structure:
            (number of n gauges [0, ..., n],
             y/x-cooridnate [y, x],
    """
    x0_gauge = np.hstack(
        [
            da_gauge.y.data.reshape(-1, 1),
            da_gauge.x.data.reshape(-1, 1),
        ]
    )

    # Create dataarray return
    return xr.DataArray(
        x0_gauge,
        coords={
            "id": da_gauge.id.data,
            "yx": ["y", "x"],
        },
    )
