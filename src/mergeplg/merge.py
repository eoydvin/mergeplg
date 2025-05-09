"""Module for merging CML and radar data."""

from __future__ import annotations

import numpy as np
import poligrain as plg
import xarray as xr

from mergeplg import bk_functions, interpolate


class MergeBase:
    """Base class for merging

    Base class providing functions for retrieving radar observations
    at CMLs and gauges
    """

    def __init__(self):
        self.gauge_ids = None
        self.intersect_weights = None
        self.get_grid_at_points = None
        self.grid_location_radar = None

    def _get_rad(self, da_rad, da_cmls=None, da_gauges=None):
        """Get radar value at CML and rain gauge positions

        Updates the functions "grid_at_points" and
        "get_grid_time_series_at_intersections" by running "_update_weights".
        Then uses these functions to get radar values at rain gauges and CMLs.
        """
        # Update weights if geometry changes
        self._update_weights(da_rad, da_cml=da_cmls, da_gauge=da_gauges)

        # Get gauge obs and radar at gauge pixel
        if da_gauges is not None:
            rad_gauges = self.get_grid_at_points(
                da_gridded_data=da_rad.expand_dims("time"),
                da_point_data=da_gauges.expand_dims("time"),
            ).data.ravel()
        else:
            rad_gauges = []

        # Get CML obs and radar along CML
        if da_cmls is not None:
            rad_cmls = (
                plg.spatial.get_grid_time_series_at_intersections(
                    grid_data=da_rad.expand_dims("time"),
                    intersect_weights=self.intersect_weights,
                )
            ).data.ravel()
        else:
            rad_cmls = []

        return np.concatenate([rad_cmls, rad_gauges]).astype(float)

    def _update_weights(self, da_grid, da_cml=None, da_gauge=None):
        """Update grid weights for CML and gauge

        Constructs the CML intersect weights, for retrieving rainfall rates along
        gridded data. Already calculated weights are reued. Also constructs
        function used for getting rainfall rates from rain gauges.

        Parameters
        ----------
        da_grid: xarray.DataArray
            Gridded rainfall data. Must contain the projected coordinates
            (x_grid, y_grid).
        da_cml: xarray.DataArray
            CML observations. Must contain the projected coordinates for the CML
            (site_0_x, site_0_y, site_1_x, site_1_y).
        da_gauge: xarray.DataArray
            Gauge observations. Must contain the projected coordinates for the
            rain gauge positions (y, x).
        """
        # Check that there is CML or gauge data, if not raise an error
        if (da_cml is None) and (da_gauge is None):
            msg = "Please provide cml or gauge data"
            raise ValueError(msg)

        # If CML is present
        if da_cml is not None:
            # If intersect weights not computed, compute all weights
            if self.intersect_weights is None:
                # Calculate CML radar grid intersection weights
                self.intersect_weights = (
                    plg.spatial.calc_sparse_intersect_weights_for_several_cmls(
                        x1_line=da_cml.site_0_x.data,
                        y1_line=da_cml.site_0_y.data,
                        x2_line=da_cml.site_1_x.data,
                        y2_line=da_cml.site_1_y.data,
                        cml_id=da_cml.cml_id.data,
                        x_grid=da_grid.x_grid.data,
                        y_grid=da_grid.y_grid.data,
                        grid_point_location=self.grid_location_radar,
                    )
                )

            # Update weights, reusing already computed weights
            else:
                # New cml names
                cml_id_new = np.sort(da_cml.cml_id.data)

                # cml names of previous update
                cml_id_old = np.sort(self.intersect_weights.cml_id.data)

                # Identify cml_id that is in the new and old array
                cml_id_keep = np.intersect1d(cml_id_new, cml_id_old)

                # Slice the stored intersect weights, keeping only new ones
                self.intersect_weights = self.intersect_weights.sel(cml_id=cml_id_keep)

                # Identify new cml_id
                cml_id_not_in_old = np.setdiff1d(cml_id_new, cml_id_old)

                # If new cml_ids available
                if cml_id_not_in_old.size > 0:
                    # Slice da_cml to get only missing coords
                    da_cml_add = da_cml.sel(cml_id=cml_id_not_in_old)

                    # Intersect weights of CMLs to add
                    intersect_weights_add = (
                        plg.spatial.calc_sparse_intersect_weights_for_several_cmls(
                            x1_line=da_cml_add.site_0_x.data,
                            y1_line=da_cml_add.site_0_y.data,
                            x2_line=da_cml_add.site_1_x.data,
                            y2_line=da_cml_add.site_1_y.data,
                            cml_id=da_cml_add.cml_id.data,
                            x_grid=da_grid.x_grid.data,
                            y_grid=da_grid.y_grid.data,
                            grid_point_location=self.grid_location_radar,
                        )
                    )

                    # Add new intersect weights
                    self.intersect_weights = xr.concat(
                        [self.intersect_weights, intersect_weights_add], dim="cml_id"
                    )

            # Update final self.intersect_weights
            self.intersect_weights = self.intersect_weights.sel(
                cml_id=da_cml.cml_id.data
            )

        # If gauge data is present
        if da_gauge is not None:
            # If intersect weights not computed, compute all weights
            if self.gauge_ids is None:
                # Calculate gridpoints for gauges
                self.get_grid_at_points = plg.spatial.GridAtPoints(
                    da_gridded_data=da_grid,
                    da_point_data=da_gauge,
                    nnear=1,
                    stat="best",
                )

                # Store gauge names for check
                self.gauge_ids = da_gauge.id.data

            # Update weights, if new gauge data is provided
            else:
                # Get names of new gauges
                gauge_id_new = da_gauge.id.data

                # Get names of gauges in previous update
                gauge_id_old = self.gauge_ids

                # Check that equal, element order is important
                if not np.array_equal(gauge_id_new, gauge_id_old):
                    # Calculate new gauge positions
                    self.get_grid_at_points = plg.spatial.GridAtPoints(
                        da_gridded_data=da_grid,
                        da_point_data=da_gauge,
                        nnear=1,
                        stat="best",
                    )


class MergeDifferenceIDW(interpolate.InterpolateIDW, MergeBase):
    """Merge ground and radar difference using IDW.

    Merges the provided radar field in ds_rad with gauge or CML observations
    by interpolating the difference (additive or multiplicative)
    between the ground and radar observations using IDW as interpolator.
    """

    def __init__(
        self,
        ds_rad,
        ds_cmls=None,
        ds_gauges=None,
        grid_location_radar="center",
        min_observations=1,
        p=2,
        idw_method="radolan",
        nnear=8,
        max_distance=60000,
        method="additive",
    ):
        """
        Initialize merging object.

        Parameters
        ----------
        ds_rad: xarray.Dataset
            Gridded radar data. Must contain the coordinates x_grid and y_grid.
        ds_cmls: xarray.Dataset
            CML geometry. Must contain the projected midpoint
            coordinates (x, y).
        ds_gauges: xarray.Dataset
            Gauge geometry. Must contain the coordinates projected
            coordinates (x, y).
        grid_location_radar: str
            Position of radar.
        min_observations:
            Number of observations needed to perform interpolation.
        p: float
            Tuning parameter for idw_method = standard.
        idw_method: str
            IDW method.
        nnear: int
            Number of nearest observations to use.
        max_distance: float
            Largest distance allowed for including an observation.
        method: str
            If set to additive, performs additive merging. If set to
            multiplicative, performs multiplicative merging.
        """
        # Init interpolation
        interpolate.InterpolateIDW.__init__(
            self,
            ds_grid=ds_rad,
            ds_cmls=ds_cmls,
            ds_gauges=ds_gauges,
            min_observations=min_observations,
            p=p,
            idw_method=idw_method,
            nnear=nnear,
            max_distance=max_distance,
        )

        # Init mergerging
        MergeBase.__init__(self)
        self.grid_location_radar = grid_location_radar
        self.method = method
        self._update_weights(ds_rad, da_cml=ds_cmls, da_gauge=ds_gauges)

    def __call__(self, da_rad, da_cmls=None, da_gauges=None):
        """Interpolate observations for one time step using IDW

        Interpolate observations for one time step.

        Parameters
        ----------
        da_rad: xarray.DataArray
            Dataset providing the grid for merging. Must contain
            projected x_grid and y_grid coordinates.
        da_cmls: xarray.DataArray
            CML observations. Must contain the projected midpoint
            coordinates (x, y) as well as the rainfall measurements stored
            under variable name 'R'.
        da_gauges: xarray.DataArray
            Gauge observations. Must contain the coordinates projected
            coordinates (x, y) as well as the rainfall measurements stored
            under variable name 'R'.

        Returns
        -------
        da_field_out: xarray.DataArray
            DataArray with the same coordinates as ds_rad but with the
            interpolated field.
        """
        # Get updated interpolator object
        self._interpolator = self._maybe_update_interpolator(
            self.y_grid, self.x_grid, da_cmls, da_gauges)

        # Update weights used for merging
        self._update_weights(da_rad, da_cml=da_cmls, da_gauge=da_gauges)

        # Get observations and radar for current time step
        obs = self._get_obs(da_cmls, da_gauges)
        rad = self._get_rad(da_rad, da_cmls, da_gauges)

        # If few observations return zero grid
        if (~np.isnan(obs)).sum() <= self.min_observations:
            return da_rad

        # Calculate radar-ground difference
        if self.method == "additive":
            diff = np.where(rad > 0, obs - rad, np.nan)

        elif self.method == "multiplicative":
            mask_zero = rad > 0.0
            diff = np.full_like(obs, np.nan, dtype=np.float64)
            diff[mask_zero] = obs[mask_zero] / rad[mask_zero]
        else:
            msg = "Method must be multiplicative or additive"
            raise ValueError(msg)

        # Coordinates to predict
        coord_pred = np.hstack(
            [self.y_grid.reshape(-1, 1), self.x_grid.reshape(-1, 1)]
        )

        # Interpolate difference
        interpolated = self._interpolator(
            q=coord_pred,
            z=diff,
            nnear=obs.size if obs.size <= self.nnear else self.nnear,
            p=self.p,
            idw_method=self.idw_method,
            max_distance=self.max_distance,
        ).reshape(self.x_grid.shape)

        interpolated = xr.DataArray(
            data=interpolated, coords=self.grid_coords, dims=self.grid_dims
        )

        # Adjust radar field
        if self.method == "additive":
            adjusted = interpolated + da_rad
            adjusted.data[adjusted < 0] = 0
        elif self.method == "multiplicative":
            adjusted = interpolated * da_rad
            adjusted.data[adjusted < 0] = 0

        if (da_cmls is not None):
            time = da_cmls.time
        elif (da_gauges is not None):
            time = da_gauges.time

        da = xr.DataArray(data=adjusted, coords=self.grid_coords, dims=self.grid_dims)
        da.coords['time'] = time
        return da

class MergeDifferenceOrdinaryKriging(interpolate.InterpolateOrdinaryKriging, MergeBase):
    """Merge CML and radar using ordinary kriging

    Merges the provided radar field in ds_rad with gauge or CML observations
    by interpolating the difference (additive or multiplicative)
    between the ground and radar observations using ordinary kriging. The class
    uses neighbouring observations for interpolation. It also by default uses
    the full line geometry for interpolation, but can treat the lines as
    points by setting full_line to False.
    """

    def __init__(
        self,
        ds_rad,
        ds_cmls=None,
        ds_gauges=None,
        variogram_model="spherical",
        variogram_parameters=None,
        grid_location_radar="center",
        discretization=8,
        min_observations=1,
        method="additive",
        nnear=8,
        max_distance=60000,
        full_line=True,
    ):
        """
        Initialize merging object.

        Parameters
        ----------
        ds_rad: xarray.Dataset
            Dataset providing the grid for merging. Must contain
            projected x_grid and y_grid coordinates.
        ds_cmls: xarray.Dataset
            CML geometry. Must contain the projected midpoint
            coordinates (x, y).
        ds_gauges: xarray.Dataset
            Gauge geometry. Must contain the coordinates projected
            coordinates (x, y).
        variogram_model: str
            Must be a valid variogram type in pykrige.
        variogram_parameters: str
            Must be a valid parameters corresponding to variogram_model.
        discretization: int
            Number of points to divide the line into.
        min_observations: int
            Number of observations needed to perform interpolation.
        method: str
            If set to additive, performs additive merging. If set to
            multiplicative, performs multiplicative merging.
        nnear: int
            Number of closest links to use for interpolation
        max_distance: float
            Largest distance allowed for including an observation.
        full_line: bool
            Whether to use the full line for block kriging. If set to false, the
            x0 geometry is reformatted to simply reflect the midpoint of the CML.
        """
        # Init interpolator
        interpolate.InterpolateOrdinaryKriging.__init__(
            self,
            ds_grid=ds_rad,
            ds_cmls=ds_cmls,
            ds_gauges=ds_gauges,
            variogram_model=variogram_model,
            variogram_parameters=variogram_parameters,
            discretization=discretization,
            min_observations=min_observations,
            nnear=nnear,
            max_distance=max_distance,
            full_line=full_line,
        )

        # Init weights update
        MergeBase.__init__(self)
        self.grid_location_radar = grid_location_radar
        self.method = method
        self._update_weights(ds_rad, da_cml=ds_cmls, da_gauge=ds_gauges)

    def __call__(
        self,
        da_rad,
        da_cmls=None,
        da_gauges=None,
        da_cmls_sigma=None,
        da_gauges_sigma=None,
    ):
        """Interpolate observations for one time step.

        Interpolates ground observations for one time step.

        Parameters
        ----------
        da_rad: xarray.DataArray
            Radar data. Must contain
            projected x_grid and y_grid coordinates.
        da_cmls: xarray.DataArray
            CML observations. Must contain the projected coordinates (site_0_x,
            site_1_x, site_0_y, site_1_y).
        da_gauges: xarray.DataArray
            Gauge observations. Must contain the projected
            coordinates (x, y).
        da_cmls_sigma: xarray.DataAarray
            CML uncertainties corresponding to the data in ds_cmls. If set to
            None, sigma is set to zero. Adds to the variogram nugget.
        da_gauges_sigma: xarray.DataArray
            Gauge uncertainties corresponding to the data in ds_gauges. If set to
            None, sigma is set to zero. Adds to the variogram nugget.

        Returns
        -------
        da_field_out: xarray.DataArray
            DataArray with the same structure as the ds_rad but with the
            interpolated field.
        """
        # Get updated interpolator object
        self._interpolator = self._maybe_update_interpolator(
            self.y_grid, self.x_grid, da_cmls, da_gauges
        )

        # Update weights used for merging
        self._update_weights(da_rad, da_cml=da_cmls, da_gauge=da_gauges)

        # Get observations, sigma and radar for current time step
        obs, sigma = self._get_obs_sigma(
            da_cmls, da_gauges, da_cmls_sigma, da_gauges_sigma
        )
        rad = self._get_rad(da_rad, da_cmls, da_gauges)

        # Calculate radar-ground difference
        if self.method == "additive":
            diff = np.where(rad > 0, obs - rad, np.nan)

        elif self.method == "multiplicative":
            mask_zero = rad > 0.0
            diff = np.full_like(obs, np.nan, dtype=np.float64)
            diff[mask_zero] = obs[mask_zero] / rad[mask_zero]

        else:
            msg = "Method must be multiplicative or additive"
            raise ValueError(msg)

        # If few observations return the radar grid
        if (~np.isnan(obs)).sum() <= self.min_observations:
            return da_rad

        # Interpolate the difference
        interpolated = self._interpolator(diff, sigma).reshape(self.x_grid.shape)
        interpolated = xr.DataArray(
            data=interpolated, coords=self.grid_coords, dims=self.grid_dims
        )

        # Adjust radar field
        if self.method == "additive":
            adjusted = interpolated + da_rad
            adjusted.data[adjusted < 0] = 0
        elif self.method == "multiplicative":
            adjusted = interpolated * da_rad
            adjusted.data[adjusted < 0] = 0

        if (da_cmls is not None):
            time = da_cmls.time
        elif (da_gauges is not None):
            time = da_gauges.time

        da = xr.DataArray(data=adjusted, coords=self.grid_coords, dims=self.grid_dims)
        da.coords['time'] = time
        return da


class MergeKrigingExternalDrift(interpolate.InterpolateKrigingBase, MergeBase):
    """Merge CML and radar using kriging with external drift.

    Merges the provided radar field in ds_rad to CML and rain gauge
    observations by using a block kriging variant of kriging with external
    drift.
    """

    def __init__(
        self,
        ds_rad,
        ds_cmls=None,
        ds_gauges=None,
        variogram_model="spherical",
        variogram_parameters=None,
        grid_location_radar="center",
        discretization=8,
        min_observations=1,
        nnear=8,
        max_distance=60000,
    ):
        """
        Initialize merging object.

        Parameters
        ----------
        ds_rad: xarray.Dataset
            Dataset providing the grid for merging. Must contain
            projected x_grid and y_grid coordinates.
        ds_cmls: xarray.Dataset
            CML geometry. Must contain the projected midpoint
            coordinates (x, y).
        ds_gauges: xarray.Dataset
            Gauge geometry. Must contain the coordinates projected
            coordinates (x, y).
        variogram_model: str
            Must be a valid variogram type in pykrige.
        variogram_parameters: str
            Must be a valid parameters corresponding to variogram_model.
        discretization: int
            Number of points to divide the line into.
        min_observations: int
            Number of observations needed to perform interpolation.
        nnear: int
            Number of closest links to use for interpolation
        max_distance: float
            Largest distance allowed for including an observation.

        """
        # Init interpolator
        interpolate.InterpolateKrigingBase.__init__(
            self,
            ds_grid=ds_rad,
            ds_cmls=ds_cmls,
            ds_gauges=ds_gauges,
            variogram_model=variogram_model,
            variogram_parameters=variogram_parameters,
            discretization=discretization,
            min_observations=min_observations,
            nnear=nnear,
            max_distance=max_distance,
            full_line=True,  # False not implemented for KED
        )

        # Init weights update
        MergeBase.__init__(self)
        self.grid_location_radar = grid_location_radar
        self._update_weights(ds_rad, da_cml=ds_cmls, da_gauge=ds_gauges)

    def _init_interpolator(self, y_grid, x_grid, ds_cmls=None, ds_gauges=None):
        return bk_functions.BKEDTree(
            self.variogram,
            y_grid,
            x_grid,
            ds_cmls=ds_cmls,
            ds_gauges=ds_gauges,
            discretization=self.discretization,
            nnear=self.nnear,
            max_distance=self.max_distance,
        )

    def __call__(
        self,
        da_rad,
        da_cmls=None,
        da_gauges=None,
        da_cmls_sigma=None,
        da_gauges_sigma=None,
    ):
        """Interpolate observations for one time step.

        Interpolates ground observations for one time step.

        Parameters
        ----------
        da_rad: xarray.DataArray
            Radar data. Must contain
            projected x_grid and y_grid coordinates.
        da_cmls: xarray.DataArray
            CML observations. Must contain the projected coordinates (site_0_x,
            site_1_x, site_0_y, site_1_y).
        da_gauges: xarray.DataArray
            Gauge observations. Must contain the projected
            coordinates (x, y).
        da_cmls_sigma: xarray.DataAarray
            CML uncertainties corresponding to the data in ds_cmls. If set to
            None, sigma is set to zero. Adds to the variogram nugget.
        da_gauges_sigma: xarray.DataArray
            Gauge uncertainties corresponding to the data in ds_gauges. If set to
            None, sigma is set to zero. Adds to the variogram nugget.

        Returns
        -------
        da_field_out: xarray.DataArray
            DataArray with the same structure as the ds_rad but with the
            interpolated field.
        """
        # Get updated interpolator object
        self._interpolator = self._maybe_update_interpolator(
            self.y_grid, self.x_grid, da_cmls, da_gauges
        )

        # Update weights used for merging
        self._update_weights(da_rad, da_cml=da_cmls, da_gauge=da_gauges)

        # Get observations, sigma and radar for current time step
        obs, sigma = self._get_obs_sigma(
            da_cmls, da_gauges, da_cmls_sigma, da_gauges_sigma
        )
        rad = self._get_rad(da_rad, da_cmls, da_gauges)

        # Default decision on which observations to ignore
        ignore = np.isnan(rad) & (obs == 0) & (rad == 0)
        obs[ignore] = np.nan  # obs nan ar ignored in interpolator

        # If few observations return zero grid
        if (~np.isnan(obs)).sum() <= self.min_observations:
            return da_rad

        # Set zero values to nan, these are ignored in interpolator
        rad_field = da_rad.data
        rad_field[rad_field <= 0] = np.nan

        # KED merging
        adjusted = self._interpolator(
            rad_field.ravel(),
            obs,
            rad,
            sigma,
        ).reshape(self.x_grid.shape)

        # Remove negative values
        adjusted[(adjusted < 0) | np.isnan(adjusted)] = 0
        
        if (da_cmls is not None):
            time = da_cmls.time
        elif (da_gauges is not None):
            time = da_gauges.time

        da = xr.DataArray(data=adjusted, coords=self.grid_coords, dims=self.grid_dims)
        da.coords['time'] = time
        return da
