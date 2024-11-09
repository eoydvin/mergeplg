"""
Created on Fri Oct 18 20:21:53 2024

@author: erlend
"""

import numpy as np
import pykrige
import xarray as xr
from scipy import stats
from sklearn.neighbors import KNeighborsRegressor


def merge_additive_idw(da_rad, cml_diff, x0):
    """Merge CML and radar using an additive approach and the CML midpoint.

    Merges the CML and radar field by interpolating the difference between
    radar and CML using IDW from sklearn.

    Parameters
    ----------
    da_rad: xarray.DataArray
        Gridded radar data.
    cml_diff: numpy.array
        Difference between the CML and radar observations at the CML locations.
    x0: numpy.array
        Coordinates of CML midpoints given as [[cml_1_y, cml_1_x], ..
        [cml_n_y, cml_n_x] using the same order as cml_diff.

    Returns
    -------
    da_rad_out: xarray.DataArray
        DataArray with the same structure as the ds_rad but with the CML
        adjusted radar field.

    """
    # Get radar grid as numpy arrays
    xgrid, ygrid = da_rad.xs.data, da_rad.ys.data

    # Create array for storing interpolated values
    shift = np.zeros(xgrid.shape)

    # Gridpoints to interpolate, skip cells with nan
    mask = np.isnan(da_rad.data)

    # Check that we have any data
    if np.sum(~mask) > 0:
        coord_pred = np.hstack(
            [ygrid[~mask].reshape(-1, 1), xgrid[~mask].reshape(-1, 1)]
        )

        # IDW interpolator kdtree, only supports IDW p=1
        idw_interpolator = KNeighborsRegressor(
            n_neighbors=cml_diff.size if cml_diff.size <= 8 else 8,
            weights="distance",  # Use distance for setting weights
        )
        idw_interpolator.fit(x0, cml_diff)

        estimate = idw_interpolator.predict(coord_pred)
        shift[~mask] = estimate

    # create xarray object similar to ds_rad
    ds_rad_out = da_rad.rename("R").to_dataset().copy()

    # Set areas with nan to zero
    shift[np.isnan(shift)] = 0

    # Do adjustment
    adjusted = shift + ds_rad_out.R.data

    # Set negative values to zero
    adjusted = np.where(adjusted > 0, adjusted, 0)

    # Store shift data
    ds_rad_out["adjusted"] = (("y", "x"), adjusted)

    # Return dataset with adjusted values
    return ds_rad_out.adjusted


def merge_additive_blockkriging(da_rad, cml_diff, x0, variogram):
    """Merge CML and radar using an additive block kriging.

    Marges the provided radar field in ds_rad to CML observations by
    interpolating the difference between the CML and radar observations using
    block kriging. This takes into account the full path integrated CML
    rainfall rate.

    Parameters
    ----------
    ds_rad: xarray.DataArray
        Gridded radar data. Must contain the x and y meshgrid given as xs
        and ys.
    cml_diff: numpy.array
        Difference between the CML and radar observations at the CML locations.
    x0: numpy.array
        CML geometry as created by calculate_cml_geometry.
    variogram: function
        A user defined python function defining the variogram. Takes a distance
        h and returns the expected variance.

    Returns
    -------
    da_rad_out: xarray.DataArray
        DataArray with the same structure as the ds_rad but with the CML
        adjusted radar field.

    """
    # Grid coordinates
    xgrid, ygrid = da_rad.xs.data, da_rad.ys.data

    # Array for storing interpolated values
    shift = np.full(xgrid.shape, np.nan)

    # Calculate lengths between all points along all CMLs
    lengths_point_l = block_points_to_lengths(x0)

    # Estimate mean variogram over link geometries
    cov_block = variogram(lengths_point_l).mean(axis=(2, 3))

    # Create Kriging matrix
    mat = np.zeros([cov_block.shape[0] + 1, cov_block.shape[1] + 1])
    mat[: cov_block.shape[0], : cov_block.shape[1]] = cov_block
    mat[-1, :-1] = np.ones(cov_block.shape[1])  # non-bias condition
    mat[:-1, -1] = np.ones(cov_block.shape[0])  # lagrange multipliers

    # Calc the inverse, only dependent on geometry
    a_inv = np.linalg.pinv(mat)

    # Skip radar pixels with np.nan
    mask = np.isnan(da_rad.data)

    # Grid to visit
    xgrid_t, ygrid_t = xgrid[~mask], ygrid[~mask]

    # array for storing CML-radar merge
    estimate = np.zeros(xgrid_t.shape)

    # Compute the contributions from all CMLs to points in grid
    for i in range(xgrid_t.size):
        # Compute lengths between all points along all links
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
        estimate[i] = cml_diff @ w

    # Store shift values
    shift[~mask] = estimate

    # create xarray object similar to ds_rad
    ds_rad_out = da_rad.rename("R").to_dataset().copy()

    # Set areas with nan to zero
    shift[np.isnan(shift)] = 0

    # Do adjustment
    adjusted = shift + ds_rad_out.R.data

    # Set negative values to zero
    adjusted = np.where(adjusted > 0, adjusted, 0)

    # Store shift data
    ds_rad_out["adjusted_rainfall"] = (("y", "x"), adjusted)

    # Return dataset with adjusted values
    return ds_rad_out.adjusted_rainfall


def merge_ked_blockkriging(da_rad, cml_rad, cml_obs, x0, variogram):
    """Merge CML and radar using an additive block kriging.

    Marges the provided radar field in ds_rad to CML observations by
    interpolating the difference between the CML and radar observations using
    block kriging. This takes into account the full path integrated CML
    rainfall rate.

    Parameters
    ----------
    da_rad: xarray.DataArray
        Gridded radar data. Must contain the x and y meshgrid given as xs
        and ys.
    cml_rad: numpy array
        Radar observations at the CML locations.
    cml_obs: numpy.array
        CML observations.
    x0: numpy.array
        CML geometry as created by calculate_cml_geometry.
    variogram: function
        A user defined python function defining the variogram. Takes a distance
        h and returns the expected variance.

    Returns
    -------
    da_rad_out: xarray.DataArray
        DataArray with the same structure as the ds_rad but with the CML
        adjusted radar field.
    """
    # Grid coordinates
    xgrid, ygrid = da_rad.xs.data, da_rad.ys.data

    # Array for storing merged values
    rain = np.full(xgrid.shape, np.nan)

    # Calculate lengths between all points along all CMLs
    lengths_point_l = block_points_to_lengths(x0)

    # Estimate mean variogram over link geometries
    cov_block = variogram(lengths_point_l).mean(axis=(2, 3))

    mat = np.zeros([cov_block.shape[0] + 2, cov_block.shape[1] + 2])
    mat[: cov_block.shape[0], : cov_block.shape[1]] = cov_block
    mat[-2, :-2] = np.ones(cov_block.shape[1])  # non-bias condition
    mat[-1, :-2] = cml_rad  # Radar drift
    mat[:-2, -2] = np.ones(cov_block.shape[0])  # lagrange multipliers
    mat[:-2, -1] = cml_rad  # Radar drift

    # Calc the inverse, only dependent on geometry (and radar for KED)
    a_inv = np.linalg.pinv(mat)

    # Skip radar pixels with np.nan
    mask = np.isnan(da_rad.data)

    # Gridpoints to use
    xgrid_t, ygrid_t = xgrid[~mask], ygrid[~mask]
    rad_field_t = da_rad.data[~mask]

    # array for storing CML-radar merge
    estimate = np.zeros(xgrid_t.shape)

    # Compute the contributions from all CMLs to points in grid
    for i in range(xgrid_t.size):
        # compute target, that is R.H.S of eq 15 (jewel2013)
        delta_x = x0[:, 1] - xgrid_t[i]
        delta_y = x0[:, 0] - ygrid_t[i]
        lengths = np.sqrt(delta_x**2 + delta_y**2)

        target = variogram(lengths).mean(axis=1)

        target = np.append(target, 1)  # non bias condition
        target = np.append(target, rad_field_t[i])  # radar value

        # compuite weights
        w = (a_inv @ target)[:-2]

        # its then the sum of the CML values (eq 8, see paragraph after eq 15)
        estimate[i] = cml_obs @ w

    rain[~mask] = estimate

    # Create a new xarray dataset
    ds_rad_out = da_rad.rename("R").to_dataset().copy()

    ds_rad_out["adjusted_rainfall"] = (("y", "x"), rain)
    return ds_rad_out.adjusted_rainfall


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


def estimate_variogram(obs, x0, variogram_model="exponential"):
    """Estimate variogram from CML and/or rain gauge data

    Estimates the variogram using the CML midpoints to estimate the distances.
    Uses pykrige as backend.

    Returns
    -------
    variogram: function
        Variogram function that returns the expected variance given the
        distance between observations.

    """
    # If x0 contains block data, get approximate midpoints
    if len(x0.shape) > 2:
        x0 = x0[:, :, int(x0.shape[1] / 2)]

    try:
        # Fit variogram using pykrige
        ok = pykrige.OrdinaryKriging(
            x0[:, 1],  # x coordinate
            x0[:, 0],  # y coordinate
            obs,
            variogram_model=variogram_model,
        )
        
        # construct variogram using pykrige
        def variogram(h):
            return ok.variogram_function(ok.variogram_model_parameters, h)
        
        # Return variogram and parameters
        return variogram, [ok.variogram_model_parameters, ok.variogram_function]
    
    # If an error occurs just use a linear variogram
    except:
        def variogram(h):
            return h    

        # Return the linear variogram
        return variogram, [1, variogram]


def estimate_transformation(obs):
    """Estimate transformation from CML and/or rain gauge data

    Estimate the transformation function using a gamma distribution and scipy
    as backend.

    Returns
    -------
    transformation: function
        Transformation function that transforms rainfall data to Gaussian
        distribution
    backtransformation: function
        Backtransformation function that transforms rainfall data from Gaussian
        distribution to Gamma distribution.
    gamma_param: list
        Parameters for the gamma distribution estimated from observations
    """
    # Estimate parameters of Gamma distribution
    k, loc, scale = stats.gamma.fit(obs)

    # Define transformation function
    def transformation(h):
        return stats.norm(0, 1).ppf(stats.gamma(k, loc=loc, scale=scale).cdf(h))

    # Define backtransformation function
    def backtransformation(h):
        return stats.gamma(k, loc=loc, scale=scale).ppf(stats.norm(0, 1).cdf(h))

    gamma_param = [k, loc, scale]

    return transformation, backtransformation, gamma_param
