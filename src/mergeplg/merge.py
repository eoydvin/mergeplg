"""Module for merging CML and radar data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import poligrain as plg
import pykrige
import xarray as xr
from scipy.stats import gamma, norm
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
        matrix with the following structure:
            (number of n CMLs [0, ..., n],
             y/x-cooridnate [0(y), 1(x)],
             interval [0, ..., disc])

    """
    # Calculate discretized positions along the lines, store in numy array
    xpos = np.zeros([ds_cmls.obs_id.size, disc + 1])  # shape (line, position)
    ypos = np.zeros([ds_cmls.obs_id.size, disc + 1])

    # For all CMLs
    for block_i, obs_id in enumerate(ds_cmls.obs_id):
        x_a = ds_cmls.sel(obs_id=obs_id).site_0_x.data
        y_a = ds_cmls.sel(obs_id=obs_id).site_0_y.data
        x_b = ds_cmls.sel(obs_id=obs_id).site_1_x.data
        y_b = ds_cmls.sel(obs_id=obs_id).site_1_y.data

        # for all dicretization steps in link estimate its place on the grid
        for i in range(disc + 1):
            xpos[block_i, i] = x_a + (i / disc) * (x_b - x_a)
            ypos[block_i, i] = y_a + (i / disc) * (y_b - y_a)

    # Store x and y coordinates in the same array (n_cmls, y/x, disc)
    return np.array([ypos, xpos]).transpose([1, 0, 2])



test


class Merge:
    """Common code for all merging methods.

    Performs basic checks such as checking that the CML and radar data has the
    same timestamps. Stores CML data, radar data, radar grid and hyperparamters
    to self. Calculates the radar rainfall values along all CMLs for all
    timesteps.

    Parameters
    ----------
    da_cml: xarray.DataArray
        CML observations. Must contain the transformed coordinates site_0_x,
        site_1_x, site_0_y and site_1_y.
    ds_rad: xarray.DataArray
        Gridded radar data. Must contain the x and y meshgrid given as xs
        and ys.
    grid_location_radar: str
        String indicating the grid location of the radar. Used for calculating
        the radar values along each CML.
    min_obs: int
        Minimum number of observations needed in order to do adjustment.

    Returns
    -------
    Nothing
    """

    def __init__(
        self,
        da_rad,
        da_cml=None,
        da_gauge=None,
        grid_location_radar="center",
        min_obs=5,
    ):
        # Check that there is radar or gauge data, if not raise an error
        if (da_cml is None) and (da_gauge is None):
            raise ValueError('Please provide cml or gauge data')

        # If CML data is present, compute radar rainfall along line
        if da_cml is not None:
            # Turn into dataset
            da_cml = da_cml.to_dataset(name="obs")

            # Calculate intersect weights
            intersect_weights = (
                plg.spatial.calc_sparse_intersect_weights_for_several_cmls(
                    x1_line=da_cml.site_0_lon.data,
                    y1_line=da_cml.site_0_lat.data,
                    x2_line=da_cml.site_1_lon.data,
                    y2_line=da_cml.site_1_lat.data,
                    cml_id=da_cml.cml_id.data,
                    x_grid=da_rad.lon.data,
                    y_grid=da_rad.lat.data,
                    grid_point_location=grid_location_radar,
                )
            )

            # Calculate radar ranfall along CMLs
            da_cml["radar"] = plg.spatial.get_grid_time_series_at_intersections(
                grid_data=da_rad,
                intersect_weights=intersect_weights,
            )

            # Set x and y coordinates of the CML to imitate a rain gauge, this
            # is used in code where the midpoint of the CML is used.
            da_cml = da_cml.assign_coords(
                x=("cml_id", (da_cml.site_0_x.data + da_cml.site_1_x.data) / 2)
            )
            da_cml = da_cml.assign_coords(
                y=("cml_id", (da_cml.site_0_y.data + da_cml.site_1_y.data) / 2)
            )

            # Add sensor type metadata
            da_cml = da_cml.assign_coords(
                sensor_type=("cml_id", np.tile("cml", da_cml.cml_id.size))
            )

            # Rename observation coordinate to obs_id
            da_cml = da_cml.rename({"cml_id": "obs_id"})

        # If gauge data is present, compute radar rainfall at point
        if da_gauge is not None:
            # Turn into dataset
            da_gauge = da_gauge.to_dataset(name="obs")

            # Calculate gridpoints for gauges
            get_grid_at_points = plg.spatial.GridAtPoints(
                da_gridded_data=da_rad,
                da_point_data=da_gauge,
                nnear=1,
                stat="best",
            )
            
            # Calculate radar rainfall at gauge possitions
            da_gauge["radar"] = get_grid_at_points(
                da_gridded_data=da_rad,
                da_point_data=da_gauge.obs,
            )

            # Set x and y coordinates of the raingauge to imitate a CML, this
            # is used in the block kriging code where points are treated as 
            # lines. 
            da_gauge = da_gauge.assign_coords(site_0_x=("id", da_gauge.x.data))
            da_gauge = da_gauge.assign_coords(site_1_x=("id", da_gauge.x.data))
            da_gauge = da_gauge.assign_coords(site_0_y=("id", da_gauge.y.data))
            da_gauge = da_gauge.assign_coords(site_1_y=("id", da_gauge.y.data))

            # Add sensor type metadata
            da_gauge = da_gauge.assign_coords(
                sensor_type=("cml_id", np.tile("gauge", da_gauge.id.size))
            )

            # Rename observation coordinate to obs_id
            da_gauge = da_gauge.rename({"id": "obs_id"})

        # CML and radar data present
        if (da_gauge is not None) and (da_cml is not None):
            
            # Check that all time dimensions are the same
            if not (da_cml.time == da_rad.time).all():
                raise ValueError("""The time coordinates of the cml dataset
                                 do not match the radar""")
                                 
            if not (da_gauge.time == da_rad.time).all():
                raise ValueError("""The time coordinates of the gauge dataset
                                 do not match the radar""")
                                 
            da_cml, da_gauge = xr.align(da_cml, da_gauge, join="inner")
            self.ds_obs = xr.merge([da_cml, da_gauge])

        # CML present, but not gauge
        elif (da_gauge is None) and (da_cml is not None):
            # Check that both time dimensions are the same
            if not (da_cml.time == da_rad.time).all():
                raise ValueError("""The time coordinates of the cml dataset
                                 do not match the radar""")
                                 
            self.ds_obs = da_cml

        # CML not present, gauge is
        elif (da_gauge is not None) and (da_cml is None):
            # Check that both time dimensions are similar
            if not (da_gauge.time == da_rad.time).all():
                raise ValueError("""The time coordinates of the gauge dataset
                                 do not match the radar""")
            self.ds_obs = da_gauge

        # This exception should not be raised
        else:
            raise ValueError('Gauge data or cml data not present')

        # Store radar
        self.da_rad = da_rad

        # Stor hyperparameters
        self.min_obs_ = min_obs


class MergeAdditiveIDW(Merge):
    """Merge CML and radar using an additive IDW (CML midpoint).

    Marges the provided radar field in ds_rad to CML observations by
    interpolating the difference between the CML and radar observations using
    IDW. This uses the rainfall rate at the CML midpoint. The variogram is
    calculated using a exponential model and variogram parameters are
    automatically fit for each timestep using CML observations.

    Parameters
    ----------
    da_rad: xarray.DataArray
        Gridded radar data. Must contain the x and y meshgrid given as xs
        and ys.
    da_cml: xarray.DataArray
        CML observations. Must contain the transformed coordinates site_0_x,
        site_1_x, site_0_y and site_1_y.
    grid_location_radar: str
        String indicating the grid location of the radar. Used for calculating
        the radar values along each CML.
    min_obs: int
        Minimum number of unique observations needed in order to do adjustment.

    Returns
    -------
    Nothing
    """

    def __init__(
        self,
        da_rad,
        da_cml=None,
        da_gauge=None,
        grid_location_radar="center",
        min_obs=5,
    ):
        Merge.__init__(self, da_rad, da_cml, da_gauge, grid_location_radar, min_obs)

        # Calculate the difference between radar and CML for all timesteps
        self.r_diff = xr.where(
            self.ds_obs.radar > 0,
            self.ds_obs.obs - self.ds_obs.radar,
            np.nan,
        )

        # Store coordinates as columns
        self.x0 = np.hstack(
            [
                self.r_diff.y.data.reshape(-1, 1),
                self.r_diff.x.data.reshape(-1, 1),
            ]
        )

    def __call__(self, time):
        """Adjust radar field.

        Evaluates if we have enough observations to adjust the radar field.
        Then adjust radar field to observations using additive Block Kriging.

        Parameters
        ----------
        time : datetime
            Timestep for doing adjustment. For instance select first timestep
            using ds_cml.time[0].

        Returns
        -------
        da_rad_out: xarray.DataArray
            DataArray with the same structure as the ds_rad but with the CML
            adjusted radar field.
        """
        # Select timestep and get observations
        diff = self.r_diff.sel(time=time).data

        # Get index of not-nan obs
        keep = np.where(~np.isnan(diff))[0]

        # Check that that there is enough observations
        if keep.size > self.min_obs_:
            # Select radar timestep
            da_rad = self.da_rad.sel(time=time)

            # Set zero cells to nan
            da_rad = xr.where(da_rad > 0, da_rad, np.nan)

            # do addtitive IDW merging
            adjusted_radar = merge_additive_idw(
                da_rad,
                diff[keep],
                self.x0[keep, :],
            )

            # Return adjusted where radar
            return xr.where(np.isnan(adjusted_radar), 0, adjusted_radar)

        # Else return the unadjusted radar
        return self.da_rad.sel(time=time)


class MergeAdditiveBlockKriging(Merge):
    """Merge CML and radar using an additive block kriging.

    Marges the provided radar field in ds_rad to CML observations by
    interpolating the difference between the CML and radar observations using
    block kriging. This takes into account the full path integrated CML
    rainfall rate. The variogram is calculated using a exponential model and
    variogram parameters are automatically fit for each timestep using CML
    observations.

    Parameters
    ----------
    da_cml: xarray.DataArray
        CML observations. Must contain the transformed coordinates site_0_x,
        site_1_x, site_0_y and site_1_y.
    ds_rad: xarray.DataArray
        Gridded radar data. Must contain the x and y meshgrid given as xs
        and ys.
    grid_location_radar: str
        String indicating the grid location of the radar. Used for calculating
        the radar values along each CML.
    min_obs: int
        Minimum number of unique observations needed in order to do adjustment.
    disc: int
        Number of points to discretize the CML into.

    Returns
    -------
    Nothing
    """

    def __init__(
        self,
        da_rad,
        da_cml=None,
        da_gauge=None,
        grid_location_radar="center",
        min_obs=5,
        disc=9,
    ):
        Merge.__init__(self, da_rad, da_cml, da_gauge, grid_location_radar, min_obs)

        # Calculate the difference between radar and CML for all timesteps
        self.r_diff = xr.where(
            self.ds_obs.radar > 0,
            self.ds_obs.obs - self.ds_obs.radar,
            np.nan,
        )

        # Midpoint of CML, used for variogram fitting
        x = (da_cml.site_0_x + da_cml.site_1_x) / 2
        y = (da_cml.site_0_y + da_cml.site_1_y) / 2

        # Calculate CML geometry
        self.x0 = calculate_cml_geometry(self.r_diff, disc=disc)

        # dictionary for storing kriging parameters and timestep
        kriging_param = {}

        # Calculate kriging parameters for all timesteps
        for time in self.r_diff.time.data:
            # Difference values for this timestep
            values = self.r_diff.sel(time=time).data

            # Keep cml obs wen not nan
            keep = np.where(~np.isnan(values))[0]

            # Check that we have enough unique obs (req. for variogram)
            if np.unique(values[keep]).size >= self.min_obs_:
                # Estimate variogram parameters
                sill, hr, nugget = pykrige.OrdinaryKriging(
                    x[keep],
                    y[keep],
                    values[keep],
                    variogram_model="exponential",
                    # enable_plotting=True
                ).variogram_model_parameters

                # Store variogram parameters
                kriging_param[time] = [sill, hr, nugget]

            # If not enough nonzero observations, store nan
            else:
                kriging_param[time] = [np.nan, np.nan, np.nan]

        # Convert to pandas dataframe.
        self.kriging_param = pd.DataFrame.from_dict(
            kriging_param, orient="index", columns=["sill", "hr", "nugget"]
        )

    def __call__(self, time):
        """Adjust radar field.

        Defines variogram and evaluates if we have enough observations to
        adjust the radar field. Then adjust radar field to observations using
        Additive Block Kriging.

        Parameters
        ----------
        time : datetime
            Timestep for doing adjustment. For instance select first timestep
            using ds_cml.time[0].

        Returns
        -------
        da_rad_out: xarray.DataArray
            DataArray with the same structure as the ds_rad but with the CML
            adjusted radar field.
        """
        # Getvariogram parameters
        nugget = self.kriging_param.loc[time.data]["nugget"]
        sill = self.kriging_param.loc[time.data]["sill"]
        hr = self.kriging_param.loc[time.data]["hr"]

        # Define variogram
        def variogram(h):  # Exponential variogram
            return nugget + (sill - nugget) * (1 - np.exp(-h * 3 / hr))

        # Select data
        diff = self.r_diff.sel(time=time).data

        # Keep when not nan
        keep = np.where(~np.isnan(diff))[0]

        # If enough unique observations
        if np.unique(diff[keep]).size > self.min_obs_:
            # Select radar timestep
            da_rad = self.da_rad.sel(time=time)

            # Set zero cells to nan
            da_rad = xr.where(da_rad > 0, da_rad, np.nan)

            # Do additive merging using block kriging
            adjusted_radar = merge_additive_blockkriging(
                da_rad,
                diff[keep],
                self.x0[keep],
                variogram,
            )

            # Return adjusted field, setting nan back to zero
            return xr.where(np.isnan(adjusted_radar), 0, adjusted_radar)

        # Else return the unadjusted radar
        return self.da_rad.sel(time=time)


class MergeBlockKrigingExternalDrift(Merge):
    """Merge CML and radar using block-kriging with external drift.

    Marges the provided radar field in ds_rad to CML observations by
    interpolating the difference between the CML and radar observations using
    block kriging. This takes into account the full path integrated CML
    rainfall rate. The variogram is calculated using a exponential model and
    variogram parameters are automatically fit for each timestep using CML
    observations.

    Parameters
    ----------
    da_cml: xarray.DataArray
        CML observations. Must contain the transformed coordinates site_0_x,
        site_1_x, site_0_y and site_1_y.
    ds_rad: xarray.DataArray
        Gridded radar data. Must contain the x and y meshgrid given as xs
        and ys.
    grid_location_radar: str
        String indicating the grid location of the radar. Used for calculating
        the radar values along each CML.
    min_obs: int
        Minimum number of observations needed in order to do adjustment.
    disc: int
        Number of points to discretize the CML into.

    Returns
    -------
    da_rad_out: xarray.DataArray
        DataArray with the same structure as the ds_rad but with the CML
        adjusted radar field.

    Returns
    -------
    Nothing
    """

    def __init__(
        self,
        da_rad,
        da_cml=None,
        da_gauge=None,
        grid_location_radar="center",
        min_obs=5,
        disc=9,
    ):
        Merge.__init__(self, da_rad, da_cml, da_gauge, grid_location_radar, min_obs)

        # Only use CMLs where weather radar observes rainfall. This is done
        # because too many zeros makes it hard to work with the transformation
        # function. We will also just adjust the radar field where we have
        # observations.

        # Midpoint of CML, used for variogram fitting, assuming
        x = (da_cml.site_0_x + da_cml.site_1_x) / 2
        y = (da_cml.site_0_y + da_cml.site_1_y) / 2

        # Calculate CML geometry
        self.x0 = calculate_cml_geometry(self.ds_obs, disc=disc)

        # dictionary for storing kriging parameters and timestep
        kriging_param = {}
        transform_param = {}

        # Calculate kriging parameters for all timesteps
        for time in self.ds_obs.time.data:
            # Difference values for this timestep
            values = self.ds_obs.sel(time=time).obs.data

            # Keep cml obs where radar observes rainfall
            keep = np.where(~np.isnan(values) & (self.ds_obs.sel(time=time).radar > 0))[
                0
            ]

            # Unique values are needed in order to estimate the variogram
            if np.unique(values[keep]).size >= self.min_obs_:
                # Fit a gamma distribution to data and store variables
                k, loc, scale = gamma.fit(values[keep])
                transform_param[time] = [k, loc, scale]

                # Estimate variogram parameters using transformed values
                sill, hr, nugget = pykrige.OrdinaryKriging(
                    x[keep],
                    y[keep],
                    norm(0, 1).ppf(gamma(k, loc=loc, scale=scale).cdf(values[keep])),
                    variogram_model="exponential",
                    # enable_plotting=True
                ).variogram_model_parameters

                # Store variogram parameters
                kriging_param[time] = [sill, hr, nugget]

            # If not enough nonzero observations, store nan
            else:
                transform_param[time] = [np.nan, np.nan, np.nan]
                kriging_param[time] = [np.nan, np.nan, np.nan]

        # Store kriging parameters in dataframe
        self.kriging_param = pd.DataFrame.from_dict(
            kriging_param, orient="index", columns=["sill", "hr", "nugget"]
        )

        # Store transform parameters in dataframe
        self.transform_param = pd.DataFrame.from_dict(
            transform_param, orient="index", columns=["k", "loc", "scale"]
        )

    def __call__(self, time):
        """Adjust radar field.

        Defines variogram and evaluates if we have enough observations to
        adjust the radar field. Then adjust radar field to observations using
        KED.

        Parameters
        ----------
        time : datetime
            Timestep for doing adjustment. For instance select first timestep
            using ds_cml.time[0].

        Returns
        -------
        da_rad_out: xarray.DataArray
            DataArray with the same structure as the ds_rad but with the CML
            adjusted radar field.

        """
        # Get variogram parameters
        nugget = self.kriging_param.loc[time.data]["nugget"]
        sill = self.kriging_param.loc[time.data]["sill"]
        hr = self.kriging_param.loc[time.data]["hr"]

        # Define variogram
        def variogram(h):  # Exponential variogram
            return nugget + (sill - nugget) * (1 - np.exp(-h * 3 / hr))

        # Get transformation parameters
        k = self.transform_param.loc[time.data]["k"]
        loc = self.transform_param.loc[time.data]["loc"]
        scale = self.transform_param.loc[time.data]["scale"]

        # Select timestp
        cml_obs = self.ds_obs.sel(time=time).obs.data
        cml_rad = self.ds_obs.sel(time=time).radar.data
        da_rad = self.da_rad.sel(time=time)

        # Select stations where CML and radar is not nan
        keep = np.where(
            ~np.isnan(cml_obs) & ~np.isnan(cml_rad) & (cml_obs > 0) & (cml_rad > 0)
        )[0]

        # Select observations meeting the requirement
        cml_obs = cml_obs[keep]
        cml_rad = cml_rad[keep]

        # If enough unique observations
        if np.unique(cml_obs).size > self.min_obs_:
            # Mask zero values in map, the KED function skips these values
            da_rad = xr.where(da_rad > 0, da_rad, np.nan)

            # Transform CML data, the fields are adjusted to these data
            cml_obs_t = norm(0, 1).ppf(gamma(k, loc=loc, scale=scale).cdf(cml_obs))

            # KED blockkriging
            adjusted_radar = merge_ked_blockkriging(
                da_rad,
                cml_rad,
                cml_obs_t,
                self.x0[keep],
                variogram,
            )

            # Backtransform adjusted radar field
            adjusted_radar = gamma(k, loc=loc, scale=scale).ppf(
                norm(0, 1).cdf(adjusted_radar)
            )

            adjusted_radar = xr.DataArray(
                data=adjusted_radar,
                coords=da_rad.coords,
                dims=da_rad.dims,
                name="adjusted_radar",
            )

            return xr.where(np.isnan(adjusted_radar), 0, adjusted_radar)

        # Else return the unadjusted radar
        return self.da_rad.sel(time=time)


def merge_additive_idw(da_rad, cml_diff, x0):
    """Merge CML and radar using an additive approach and the CML midpoint.

    Merges the CML and radar field by interpolating the difference between
    radar and CML using IDW from sklearn. Note that a drawback of this approach
    is that the current sklearn implementation do not implement the IDW
    p-parameter but assumes p=1.

    Parameters
    ----------
    da_rad: xarray.DataArray
        Gridded radar data. Must contain the x and y meshgrid given as xs
        and ys.
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
