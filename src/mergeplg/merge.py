"""Module for merging CML and radar data."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsRegressor


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


def calculate_cml_geometry(ds_cmls, disc=8):
    """Calculate the position of points along CMLs.

    Calculates the discretized CML geometry by dividing the CMLs into
    disc-number of intervals. The ds_cmls xarray object must contain the
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
    disc: int
        Number of intervals to discretize lines into.

    Returns
    -------
    x0: np.array
        Array with coordinates for all CMLs. The array is organized into a 3D
        atrix with the following structure:
            (number of n CMLs [0, ..., n],
             y/x-cooridnate [0(y), 1(x)],
             interval [0, ..., disc])

    """
    # Calculate discretized positions along the lines, store in numy array
    xpos = np.zeros([ds_cmls.cml_id.size, disc + 1])  # shape (line, position)
    ypos = np.zeros([ds_cmls.cml_id.size, disc + 1])

    # For all CMLs
    for block_i, cml_id in enumerate(ds_cmls.cml_id):
        x_a = ds_cmls.sel(cml_id=cml_id).site_0_x.data
        y_a = ds_cmls.sel(cml_id=cml_id).site_0_y.data
        x_b = ds_cmls.sel(cml_id=cml_id).site_1_x.data
        y_b = ds_cmls.sel(cml_id=cml_id).site_1_y.data

        # for all dicretization steps in link estimate its place on the grid
        for i in range(disc + 1):
            xpos[block_i, i] = x_a + (i / disc) * (x_b - x_a)
            ypos[block_i, i] = y_a + (i / disc) * (y_b - y_a)

    # Store x and y coordinates in the same array (n_cmls, y/x, disc)
    return np.array([ypos, xpos]).transpose([1, 0, 2])


def merge_additive_idw(ds_diff, ds_rad, where_rad=True, min_obs=5):
    """Merge CML and radar using an additive approach and the CML midpoint.

    Merges the CML and radar field by interpolating the difference between
    radar and CML using IDW from sklearn. Note that a drawback of this approach
    is that the current sklearn implementation do not implement the IDW
    p-parameter but assumes p=1.

    Parameters
    ----------
    ds_diff: xarray.DataArray
        Difference between the CML and radar observations at the CML locations.
        Must contain the CML midpoint x and y position given as xarray
        coordinates mid_x and mid_y.
    ds_rad: xarray.DataArray
        Gridded radar data. Must contain the x and y meshgrid given as xs
        and ys.
    where_rad: bool
        If set to True only do adjustment in cells where radar observes
        rainfall. If set to false to adjustment in those cells as well.
    min_obs: int
        Minimum number of observations needed in order to do adjustment.

    Returns
    -------
    da_rad_out: xarray.DataArray
        DataArray with the same structure as the ds_rad but with the CML
        adjusted radar field.

    """
    # Get radar grid as numpy arrays
    xgrid, ygrid = ds_rad.xs.data, ds_rad.ys.data

    # Create array for storing interpolated values
    shift = np.zeros(xgrid.shape)

    # Select time step
    cml_obs = ds_diff.data
    rad_field = ds_rad.data

    # Remove CMLs that has no radar observations (dry spells)
    keep = ~np.isnan(cml_obs) if where_rad else np.full(cml_obs.shape, True)

    # Select the CMLs to keep
    cml_i_keep = np.where(keep)[0]
    cml_obs = cml_obs[cml_i_keep]

    # Check that we have enough observations for doing adjustment
    if cml_i_keep.size >= min_obs:
        x = ds_diff.isel(cml_id=cml_i_keep).x.data
        y = ds_diff.isel(cml_id=cml_i_keep).y.data
        z = cml_obs

        # Create sklearn designmatrix
        coord_train = np.hstack([y.reshape(-1, 1), x.reshape(-1, 1)])

        # Gridpoints to interpolate, skip cells with nan
        mask = np.isnan(rad_field)

        # Skip radar pixels with zero
        if where_rad:
            mask = mask | (rad_field == 0)

        # Check that we have any data
        if np.sum(~mask) > 0:
            coord_pred = np.hstack(
                [ygrid[~mask].reshape(-1, 1), xgrid[~mask].reshape(-1, 1)]
            )

            # IDW interpolator kdtree, only supports IDW p=1
            idw_interpolator = KNeighborsRegressor(
                n_neighbors=cml_i_keep.size if cml_i_keep.size <= 8 else 8,
                weights="distance",  # Use distance for setting weights
            )
            idw_interpolator.fit(coord_train, z)

            estimate = idw_interpolator.predict(coord_pred)
            shift[~mask] = estimate

    # create xarray object similar to ds_rad
    ds_rad_out = ds_rad.rename("R").to_dataset().copy()

    # Store shift data
    ds_rad_out["shift"] = (("y", "x"), shift)

    # Remove adjustment effect where we do not have radar observations
    if where_rad:
        ds_rad_out["shift"] = ds_rad_out["shift"].where(ds_rad_out.R > 0, 0)

    # Adjust field
    ds_rad_out["adjusted"] = (
        ("y", "x"),
        ds_rad_out["shift"].data + ds_rad_out.R.data,
    )

    # Set negative values to zero
    ds_rad_out["adjusted"] = ds_rad_out.adjusted.where(
        ds_rad_out.adjusted > 0,
        0,
    )

    # Return dataset with adjusted values
    return ds_rad_out.adjusted


def merge_additive_blockkriging(
    ds_diff,
    ds_rad,
    x0,
    variogram,
    where_rad=True,
    min_obs=5,
):
    """Merge CML and radar using an additive block kriging.

    Marges the provided radar field in ds_rad to CML observations by
    interpolating the difference between the CML and radar observations using
    block kriging. This takes into account the full path integrated CML
    rainfall rate.

    Parameters
    ----------
    ds_diff: xarray.DataArray
        Difference between the CML and radar observations at the CML locations.
        Must contain the CML midpoint x and y position given as xarray
        coordinates mid_x and mid_y.
    ds_rad: xarray.DataArray
        Gridded radar data. Must contain the x and y meshgrid given as xs
        and ys.
    x0: numpy.array
        CML geometry as created by calculate_cml_geometry.
    variogram: function
        A user defined python function defining the variogram. Takes a distance
        h and returns the expected variance.
    where_rad: bool
        If set to True only do adjustment in cells where radar observes
        rainfall. If set to false to adjustment in those cells as well.
    min_obs: int
        Minimum number of observations needed in order to do adjustment.

    Returns
    -------
    da_rad_out: xarray.DataArray
        DataArray with the same structure as the ds_rad but with the CML
        adjusted radar field.

    """
    # Grid coordinates
    xgrid, ygrid = ds_rad.xs.data, ds_rad.ys.data

    # Array for storing interpolated values
    shift = np.zeros(xgrid.shape)

    # To numpy for fast lookup
    diff = ds_diff.data
    rad_field = ds_rad.data

    # Remove CMLs that has no radar observations (dry spells)
    keep = ~np.isnan(diff) if where_rad else np.full(diff.shape, True)

    # Select the CMLs to keep
    cml_i_keep = np.where(keep)[0]
    diff = diff[cml_i_keep]

    # Adjust radar if enough observations
    if cml_i_keep.size >= min_obs:
        # Calculate lengths between all points along all CMLs
        lengths_point_l = block_points_to_lengths(x0)

        # estimate mean variogram over link geometries
        cov_block = variogram(lengths_point_l[cml_i_keep, :][:, cml_i_keep]).mean(
            axis=(2, 3)
        )

        # Create Kriging matrix
        mat = np.zeros([cov_block.shape[0] + 1, cov_block.shape[1] + 1])
        mat[: cov_block.shape[0], : cov_block.shape[1]] = cov_block
        mat[-1, :-1] = np.ones(cov_block.shape[1])  # non-bias condition
        mat[:-1, -1] = np.ones(cov_block.shape[0])  # lagrange multipliers

        # Calc the inverse, only dependent on geometry (and radar for KED)
        a_inv = np.linalg.pinv(mat)

        # Skip radar pixels with np.nan
        mask = np.isnan(rad_field)

        # Skip radar pixels with zero
        if where_rad:
            mask = mask | (rad_field == 0)

        # Grid to visit
        xgrid_t, ygrid_t = xgrid[~mask], ygrid[~mask]

        # array for storing CML-radar merge
        estimate = np.zeros(xgrid_t.shape)

        # Compute the contributions from all CMLs to points in grid
        for i in range(xgrid_t.size):
            # Compute lengths between all points along all links
            delta_x = x0[cml_i_keep, 1] - xgrid_t[i]
            delta_y = x0[cml_i_keep, 0] - ygrid_t[i]
            lengths = np.sqrt(delta_x**2 + delta_y**2)

            # Estimate expected variance for all links
            target = variogram(lengths).mean(axis=1)

            # Add non bias condition
            target = np.append(target, 1)

            # Compute the kriging weights
            w = (a_inv @ target)[:-1]

            # Estimate rainfall amounts at location i
            estimate[i] = diff @ w

        shift[~mask] = estimate

    # create xarray object similar to ds_rad
    ds_rad_out = ds_rad.rename("R").to_dataset().copy()

    # Store shift data
    ds_rad_out["shift"] = (("y", "x"), shift)

    # Remove adjustment effect where we do not have radar observations
    if where_rad:
        ds_rad_out["shift"] = ds_rad_out["shift"].where(ds_rad_out.R > 0, 0)

    # Adjust field
    ds_rad_out["adjusted"] = (
        ("y", "x"),
        ds_rad_out["shift"].data + ds_rad_out.R.data,
    )

    # Set negative values to zero
    ds_rad_out["adjusted"] = ds_rad_out.adjusted.where(
        ds_rad_out.adjusted > 0,
        0,
    )

    # Return dataset with adjusted values
    return ds_rad_out.adjusted
