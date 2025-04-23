"""Module for merging CML and radar data."""

from __future__ import annotations

import numpy as np
import poligrain as plg
import xarray as xr

from mergeplg import bk_functions

from .radolan import idw


class Interpolator:
    """Base class.

    Base class providing functions for testing if the cml and rain gauge
    positions are the same. Updates the interpolator and radar grid weights if
    positions changes.
    """

    def __init__(self):
        self.x_gauge = None
        self.y_gauge = None
        self.x_0_cml = None
        self.x_1_cml = None
        self.y_0_cml = None
        self.y_1_cml = None
        self.gauge_ids = None
        self.intersect_weights = None
        self.get_grid_at_points = None
        self.grid_location_radar = None
        self._interpolator = None

    def _init_interpolator(self, ds_grid, ds_cmls=None, ds_gauges=None):
        # Needs to return the interpolator
        raise NotImplementedError()

    def _maybe_update_interpolator(self, ds_grid, ds_cmls=None, ds_gauges=None):
        """Update observations and interpolator

        Function updates the interpolator function if the positions changes.

        Parameters
        ----------
        ds_gauges: xarray.DataArray
            Gauge observations. Must contain the projected
            coordinates (x, y).
        ds_cmls: xarray.DataArray
            CML observations. Must contain the projected coordinates (site_0_x,
            site_1_x, site_0_y, site_1_y).

        Returns
        -------
        self._interpolator: function
            Initialized interpolator function.

        """
        if ds_gauges is not None:
            # Check if coordinates are the same
            x_gauge = ds_gauges.x.data
            y_gauge = ds_gauges.y.data
            x_gauge_equal = np.array_equal(x_gauge, self.x_gauge)
            y_gauge_equal = np.array_equal(y_gauge, self.y_gauge)
            gauges_equal = x_gauge_equal and y_gauge_equal

            # Update gauge coordinates
            self.x_gauge = x_gauge
            self.y_gauge = y_gauge

        # Set gauge coordinates to None if gauges not present
        else:
            self.x_gauge = None
            self.y_gauge = None
            gauges_equal = None  # Ignored

        if ds_cmls is not None:
            # Check if coordinates are the same
            x_0_cml = ds_cmls.site_0_x.data
            x_1_cml = ds_cmls.site_1_x.data
            y_0_cml = ds_cmls.site_0_y.data
            y_1_cml = ds_cmls.site_1_y.data
            cml_x0_eq = np.array_equal(x_0_cml, self.x_0_cml)
            cml_x1_eq = np.array_equal(x_1_cml, self.x_1_cml)
            cml_y0_eq = np.array_equal(y_0_cml, self.y_0_cml)
            cml_y1_eq = np.array_equal(y_1_cml, self.y_1_cml)
            cmls_equal = cml_x0_eq and cml_x1_eq and cml_y0_eq and cml_y1_eq

            # Update CML coordinates
            self.x_0_cml = x_0_cml
            self.x_1_cml = x_1_cml
            self.y_0_cml = y_0_cml
            self.y_1_cml = y_1_cml

        # Set CML coordinates to None if CML not present
        else:
            self.x_0_cml = None
            self.x_1_cml = None
            self.y_0_cml = None
            self.y_1_cml = None
            cmls_equal = None

        # Update interpolator if needed
        if (ds_gauges is not None) and (ds_cmls is not None):
            if (not cmls_equal) or (not gauges_equal):  # CMLS or gauges changed
                interpolator = self._init_interpolator(ds_grid, ds_cmls, ds_gauges)
            else:
                interpolator = self._interpolator

        elif ds_gauges is not None:
            if not gauges_equal:  # Gauges changed
                interpolator = self._init_interpolator(ds_grid, ds_gauges=ds_gauges)
            else:
                interpolator = self._interpolator

        elif ds_cmls is not None:
            if not cmls_equal:  # CMLs changed
                interpolator = self._init_interpolator(ds_grid, ds_cmls=ds_cmls)
            else:
                interpolator = self._interpolator
        else:
            msg = "Please provide CML or rain gauge data"
            raise ValueError(msg)

        return interpolator

    def _update_weights(self, da_grid, da_cml=None, da_gauge=None):
        """Update grid weights for CML and gauge

        Constructs the CML intersect weights, for retrieving rainfall rates along
        gridded data. Also constructs function used for getting rainfall rates
        from rain gauges.

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


class MergeDifferenceIDW(Interpolator):
    """Merge ground and radar difference using IDW.

    Merges the provided radar field in ds_rad with gauge or CML observations
    by interpolating the difference (additive or multiplicative)
    between the ground and radar observations using IDW.
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
        # Store interpolation variables
        Interpolator.__init__(self)
        self.min_observations = min_observations
        self.grid_location_radar = grid_location_radar
        self.p = p
        self.idw_method = idw_method
        self.nnear = nnear
        self.max_distance = max_distance
        self.method = method

        # Init base class
        self._interpolator = self._maybe_update_interpolator(ds_rad, ds_cmls, ds_gauges)
        self._update_weights(ds_rad, da_cml=ds_cmls, da_gauge=ds_gauges)

    def _init_interpolator(self, _ds_rad, ds_cmls=None, ds_gauges=None):
        # Get CML and gauge coordinates if present
        if ds_cmls is not None:
            cml_x = 0.5 * (ds_cmls.site_0_x.data + ds_cmls.site_1_x.data)
            cml_y = 0.5 * (ds_cmls.site_0_y.data + ds_cmls.site_1_y.data)
        else:
            cml_x = []
            cml_y = []

        if ds_gauges is not None:
            gauge_x = ds_gauges.x.data
            gauge_y = ds_gauges.y.data
        else:
            gauge_x = []
            gauge_y = []

        # Concat and store ordered
        y = np.concatenate([cml_y, gauge_y])
        x = np.concatenate([cml_x, gauge_x])
        yx = np.hstack([y.reshape(-1, 1), x.reshape(-1, 1)])

        return idw.Invdisttree(yx)

    def _get_obs_rad(self, da_rad, da_cmls=None, da_gauges=None):
        # Get gauge obs and radar at gauge pixel
        if da_gauges is not None:
            obs_gauges = da_gauges.data.ravel()
            rad_gauges = self.get_grid_at_points(
                da_gridded_data=da_rad.expand_dims("time"),
                da_point_data=da_gauges.expand_dims("time"),
            ).data.ravel()
        else:
            obs_gauges = []
            rad_gauges = []

        # Get CML obs and radar along CML
        if da_cmls is not None:
            obs_cmls = da_cmls.data.ravel()
            rad_cmls = (
                plg.spatial.get_grid_time_series_at_intersections(
                    grid_data=da_rad.expand_dims("time"),
                    intersect_weights=self.intersect_weights,
                )
            ).data.ravel()
        else:
            obs_cmls = []
            rad_cmls = []

        obs = np.concatenate([obs_cmls, obs_gauges]).astype(float)
        rad = np.concatenate([rad_cmls, rad_gauges]).astype(float)

        return obs, rad

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
        self._interpolator = self._maybe_update_interpolator(da_rad, da_cmls, da_gauges)

        # Update weights used for merging
        self._update_weights(da_rad, da_cml=da_cmls, da_gauge=da_gauges)

        # Get correct order of observations
        obs, rad = self._get_obs_rad(da_rad, da_cmls, da_gauges)

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
            [da_rad.y_grid.data.reshape(-1, 1), da_rad.x_grid.data.reshape(-1, 1)]
        )

        # IDW interpolator invdisttree
        interpolated = self._interpolator(
            q=coord_pred,
            z=diff,
            nnear=obs.size if obs.size <= self.nnear else self.nnear,
            p=self.p,
            idw_method=self.idw_method,
            max_distance=self.max_distance,
        ).reshape(da_rad.x_grid.shape)

        interpolated = xr.DataArray(
            data=interpolated, coords=da_rad.coords, dims=da_rad.dims
        )

        # Adjust radar field
        if self.method == "additive":
            adjusted = interpolated + da_rad
            adjusted.data[adjusted < 0] = 0
        elif self.method == "multiplicative":
            adjusted = interpolated * da_rad
            adjusted.data[adjusted < 0] = 0

        return adjusted


class MergeDifferenceOrdinaryKriging(Interpolator):
    """Merge CML and radar using ordinary kriging

    Merges the provided radar field in ds_rad with gauge or CML observations
    by interpolating the difference (additive or multiplicative)
    between the ground and radar observations using ordinary kriging. The class
    defaults to interpolation using neighbouring observations, but it can
    also consider all observations by setting n_closest to False. It also
    by default uses the full line geometry for interpolation, but can treat
    the lines as points by setting full_line to False.
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
        Interpolator.__init__(self)
        self.discretization = discretization
        self.grid_location_radar = grid_location_radar
        self.min_observations = min_observations
        self.method = method
        self.nnear = nnear
        self.max_distance = max_distance
        self.full_line = full_line

        # Construct variogram using parameters provided by user
        if variogram_parameters is None:
            variogram_parameters = {"sill": 0.9, "range": 5000, "nugget": 0.1}
        obs = np.array([0, 1])  # dummy variables
        coord = np.array([[0, 1], [1, 0]])

        self.variogram = bk_functions.construct_variogram(
            obs, coord, variogram_parameters, variogram_model
        )

        self._interpolator = self._maybe_update_interpolator(ds_rad, ds_cmls, ds_gauges)
        self._update_weights(ds_rad, da_cml=ds_cmls, da_gauge=ds_gauges)

        # Construct variogram using parameters provided by user
        if variogram_parameters is None:
            variogram_parameters = {"sill": 0.9, "range": 5000, "nugget": 0.1}
        obs = np.array([0, 1])  # dummy variables
        coord = np.array([[0, 1], [1, 0]])

        self.variogram = bk_functions.construct_variogram(
            obs, coord, variogram_parameters, variogram_model
        )

    def _get_obs_rad_sigma(
        self,
        da_rad,
        da_cmls=None,
        da_gauges=None,
        da_cmls_sigma=None,
        da_gauges_sigma=None,
    ):
        if da_gauges is not None:
            obs_gauges = da_gauges.data.ravel()
            rad_gauges = self.get_grid_at_points(
                da_gridded_data=da_rad.expand_dims("time"),
                da_point_data=da_gauges.expand_dims("time"),
            ).data.ravel()
            if da_gauges_sigma is not None:
                sigma_gauges = da_gauges_sigma.data.ravel()
            else:
                sigma_gauges = np.zeros(obs_gauges.size)
        else:
            obs_gauges = []
            rad_gauges = []
            sigma_gauges = []

        if da_cmls is not None:
            obs_cmls = da_cmls.data.ravel()
            rad_cmls = (
                plg.spatial.get_grid_time_series_at_intersections(
                    grid_data=da_rad.expand_dims("time"),
                    intersect_weights=self.intersect_weights,
                )
            ).data.ravel()
            if da_cmls_sigma is not None:
                sigma_cmls = da_cmls_sigma.data.ravel()
            else:
                sigma_cmls = np.zeros(obs_cmls.size)
        else:
            obs_cmls = []
            rad_cmls = []
            sigma_cmls = []

        # Stack observations and sigma in the order expected by interpolator
        obs = np.concatenate([obs_cmls, obs_gauges]).astype(float)
        rad = np.concatenate([rad_cmls, rad_gauges]).astype(float)
        sigma = np.concatenate([sigma_cmls, sigma_gauges]).astype(float)

        return obs, rad, sigma

    def _init_interpolator(self, ds_grid, ds_cmls=None, ds_gauges=None):
        return bk_functions.OBKrigTree(
            self.variogram,
            ds_grid=ds_grid,
            ds_cmls=ds_cmls,
            ds_gauges=ds_gauges,
            discretization=self.discretization,
            nnear=self.nnear,
            max_distance=self.max_distance,
            full_line=self.full_line,
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
        self._interpolator = self._maybe_update_interpolator(da_rad, da_cmls, da_gauges)

        # Update weights used for merging
        self._update_weights(da_rad, da_cml=da_cmls, da_gauge=da_gauges)

        # Get correct order of observations and sigma (if sigma is present)
        obs, rad, sigma = self._get_obs_rad_sigma(
            da_rad, da_cmls, da_gauges, da_cmls_sigma, da_gauges_sigma
        )

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

        # If few observations return zero grid
        if (~np.isnan(obs)).sum() <= self.min_observations:
            return da_rad

        # Interpolate the difference
        interpolated = self._interpolator(diff, sigma).reshape(da_rad.x_grid.shape)
        interpolated = xr.DataArray(
            data=interpolated, coords=da_rad.coords, dims=da_rad.dims
        )

        # Adjust radar field
        if self.method == "additive":
            adjusted = interpolated + da_rad
            adjusted.data[adjusted < 0] = 0
        elif self.method == "multiplicative":
            adjusted = interpolated * da_rad
            adjusted.data[adjusted < 0] = 0

        return adjusted


class MergeKrigingExternalDrift(Interpolator):
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
        Interpolator.__init__(self)
        self.discretization = discretization
        self.grid_location_radar = grid_location_radar
        self.min_observations = min_observations
        self.nnear = nnear
        self.max_distance = max_distance

        # Construct variogram using parameters provided by user
        if variogram_parameters is None:
            variogram_parameters = {"sill": 0.9, "range": 5000, "nugget": 0.1}
        obs = np.array([0, 1])  # dummy variables
        coord = np.array([[0, 1], [1, 0]])

        self.variogram = bk_functions.construct_variogram(
            obs, coord, variogram_parameters, variogram_model
        )

        self._interpolator = self._maybe_update_interpolator(ds_rad, ds_cmls, ds_gauges)
        self._update_weights(ds_rad, da_cml=ds_cmls, da_gauge=ds_gauges)

        # Construct variogram using parameters provided by user
        if variogram_parameters is None:
            variogram_parameters = {"sill": 0.9, "range": 5000, "nugget": 0.1}
        obs = np.array([0, 1])  # dummy variables
        coord = np.array([[0, 1], [1, 0]])

        self.variogram = bk_functions.construct_variogram(
            obs, coord, variogram_parameters, variogram_model
        )

    def _get_obs_rad_sigma(
        self,
        da_rad,
        da_cmls=None,
        da_gauges=None,
        da_cmls_sigma=None,
        da_gauges_sigma=None,
    ):
        if da_gauges is not None:
            obs_gauges = da_gauges.data.ravel()
            rad_gauges = self.get_grid_at_points(
                da_gridded_data=da_rad.expand_dims("time"),
                da_point_data=da_gauges.expand_dims("time"),
            ).data.ravel()
            if da_gauges_sigma is not None:
                sigma_gauges = da_gauges_sigma.data.ravel()
            else:
                sigma_gauges = np.zeros(obs_gauges.size)
        else:
            obs_gauges = []
            rad_gauges = []
            sigma_gauges = []

        if da_cmls is not None:
            obs_cmls = da_cmls.data.ravel()
            rad_cmls = (
                plg.spatial.get_grid_time_series_at_intersections(
                    grid_data=da_rad.expand_dims("time"),
                    intersect_weights=self.intersect_weights,
                )
            ).data.ravel()
            if da_cmls_sigma is not None:
                sigma_cmls = da_cmls_sigma.data.ravel()
            else:
                sigma_cmls = np.zeros(obs_cmls.size)
        else:
            obs_cmls = []
            rad_cmls = []
            sigma_cmls = []

        # Stack observations and sigma in the order expected by interpolator
        obs = np.concatenate([obs_cmls, obs_gauges]).astype(float)
        rad = np.concatenate([rad_cmls, rad_gauges]).astype(float)
        sigma = np.concatenate([sigma_cmls, sigma_gauges]).astype(float)

        return obs, rad, sigma

    def _init_interpolator(self, ds_grid, ds_cmls=None, ds_gauges=None):
        return bk_functions.BKEDTree(
            self.variogram,
            ds_rad=ds_grid,
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
        self._interpolator = self._maybe_update_interpolator(da_rad, da_cmls, da_gauges)

        # Update weights used for merging
        self._update_weights(da_rad, da_cml=da_cmls, da_gauge=da_gauges)

        # Get correct order of observations and sigma (if sigma is present)
        obs, rad, sigma = self._get_obs_rad_sigma(
            da_rad, da_cmls, da_gauges, da_cmls_sigma, da_gauges_sigma
        )

        # Default decision on which observations to ignore
        ignore = np.isnan(rad) & (obs == 0) & (rad == 0)
        obs[ignore] = np.nan  # obs nan ar ignored in interpolator

        # If few observations return zero grid
        if (~np.isnan(obs)).sum() <= self.min_observations:
            return da_rad

        # Set zero values to nan, these are ignored in interpolator
        rad_field = da_rad.data
        rad_field[rad_field <= 0] = np.nan

        # do KED merging
        adjusted = self._interpolator(
            rad_field.ravel(),
            obs,
            rad,
            sigma,
        ).reshape(da_rad.x_grid.shape)

        # Remove negative values
        adjusted[(adjusted < 0) | np.isnan(adjusted)] = 0
        return xr.DataArray(data=adjusted, coords=da_rad.coords, dims=da_rad.dims)
