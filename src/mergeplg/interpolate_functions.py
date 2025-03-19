"""
Created on Fri Oct 18 20:21:53 2024

@author: erlend
"""

import numpy as np
import pykrige
import xarray as xr
from scipy import stats

from .radolan import idw


def interpolate_idw(
    xgrid, ygrid, obs, x0, p=2, idw_method="radolan", n_closest=8, max_distance=60000
):
    """Interpolate observations using IDW

    Interpolate CML and rain gauge data usgin Invdisttree from radolan.

    Parameters
    ----------
    xgrid numpy.array
        x coordinates as a meshgrid
    ygrid numpy array
        y coodrinates as a meshgrid
    obs: numpy.array
        Observations to interpolate
    x0: numpy.array
        Coordinates of observations given as [[obs_1_y, obs_1_x], ..
        [obs_n_y, obs_n_x] using the same order as obs.
    p: float
        IDW interpolation parameter
    idw_method: str
        by default "radolan"
    n_closest: int
        number of neighbours to use for interpolation
    max_distance: float
        Max allowed distance for including a observation

    Returns
    -------
    interpolated_field: numpy.array
        Numpy array with the same structure as xgrid/ygrid containing
        the interpolated field.

    """

    coord_pred = np.hstack(
        [ygrid.reshape(-1, 1), xgrid.reshape(-1, 1)]
    )

    # IDW interpolator invdisttree
    idw_interpolator = idw.Invdisttree(x0)
    interpolated = idw_interpolator(
        q=coord_pred,
        z=obs,
        nnear=obs.size if obs.size <= n_closest else n_closest,
        p=p,
        idw_method=idw_method,
        max_distance=max_distance,
    )

    # Return dataset with adjusted values
    return interpolated.reshape(xgrid.shape)


def interpolate_neighbourhood_block_kriging(
        xgrid, 
        ygrid, 
        obs, 
        x0, 
        variogram, 
        n_closest,
        max_distance=60000,
    ):
    """Interpolate observations using neighbourhood block kriging

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
    n_closest: int
        Number of neighbors to use for interpolation
    max_distance: float
        Max allowed distance for including a observation
    Returns
    -------
    interpolated_field: numpy.array
        Numpy array with the same structure as xgrid/ygrid containing
        the interpolated field.

    """

    # Calculate lengths between all points along all CMLs
    lengths_point_l = block_points_to_lengths(x0)

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
        length[lengths > max_distance] = np.nan

        # Get the n closest links
        indices = np.argpartition(np.nanmin(lengths, axis=1), n_closest)[:n_closest]
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
        estimate[i] = cml_diff[indices] @ w

    # Return dataset with interpolated values
    return estimate.reshape(xgrid.shape)

def interpolate_block_kriging(
        xgrid, 
        ygrid, 
        obs, 
        x0, 
        variogram, 
        n_closest=8,
        max_distance=60000,
    ):
    """Interpolate observations using Block Kriging

    Interpolate CML and rain gauge data using an implementation of block kriging.

    Parameters
    ----------
    xgrid numpy.array
        x coordinates as a meshgrid
    ygrid numpy array
        y coodrinates as a meshgrid
    obs: numpy.array
        Observations to interpolate
    x0: numpy.array
        Coordinates of observations given as [[obs_1_y, obs_1_x], ..
        [obs_n_y, obs_n_x] using the same order as obs.
    p: float
        IDW interpolation parameter
    idw_method: str
        by default "radolan"
    nnear: int
        number of neighbours to use for interpolation
    max_distance: float
        max distance allowed interpolation distance

    Returns
    -------
    interpolated_field: numpy.array
        Numpy array with the same structure as xgrid/ygrid containing
        the interpolated field.

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

        # Get the n closest links
        indices = np.argpartition(lengths.min(axis=1), n_closest)[:n_closest]
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
        estimate[i] = cml_diff[indices] @ w

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
    except ValueError:

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
