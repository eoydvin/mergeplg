"""Module for merging CML and radar data."""

from __future__ import annotations

import numpy as np
import xarray as xr

from mergeplg import bk_functions

from .radolan import idw


class Interpolator:
    """Base class.

    Base class providing functions for testing if the cml and rain gauge
    positions are the same. Updates the interpolator if positions changes.
    """

    def _init_interpolator(self, ds_grid, ds_cmls=None, ds_gauges=None):
        # Needs to return the interpolator
        raise NotImplementedError()

    def __init__(self):
        self.x_gauge = None
        self.y_gauge = None
        self.x_0_cml = None
        self.x_1_cml = None
        self.y_0_cml = None
        self.y_1_cml = None
        self._interpolator = None

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
            gauges_equal = None

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


class InterpolateIDW(Interpolator):
    """Interpolate CML and rain gauge using IDW

    Interpolates the provided CML and rain gauge observations using
    inverse distance weighting. The function uses the n nearest observations.
    """

    def __init__(
        self,
        ds_grid,
        ds_cmls=None,
        ds_gauges=None,
        min_observations=1,
        p=2,
        idw_method="standard",
        nnear=8,
        max_distance=60000,
    ):
        """Initialize interpolator object

        Parameters
        ----------
        ds_grid: xarray.Dataset
            Dataset providing the grid for interpolation. Must contain
            projected x_grid and y_grid coordinates.
        ds_cmls: xarray.Dataset
            CML geometry. Must contain the projected midpoint
            coordinates (x, y).
        ds_gauges: xarray.Dataset
            Gauge geometry. Must contain the coordinates projected
            coordinates (x, y).
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
        """
        # Init base class
        Interpolator.__init__(self)
        self._interpolator = self._maybe_update_interpolator(
            ds_grid, ds_cmls, ds_gauges
        )

        # Store interpolation variables
        self.min_observations = min_observations
        self.p = p
        self.idw_method = idw_method
        self.nnear = nnear
        self.max_distance = max_distance

    def _init_interpolator(self, _ds_grid, ds_cmls=None, ds_gauges=None):
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

    def _get_obs(self, da_cmls=None, da_gauges=None):
        obs_cmls = da_cmls.data.ravel() if da_cmls is not None else []
        obs_gauges = da_gauges.data.ravel() if da_gauges is not None else []

        return np.concatenate([obs_cmls, obs_gauges]).astype(float)

    def __call__(self, da_grid, da_cmls=None, da_gauges=None):
        """Interpolate observations for one time step using IDW

        Interpolate observations for one time step.

        Parameters
        ----------
        da_grid: xarray.DataArray
            Dataarray providing the grid for interpolation. Must contain
            projected x_grid and y_grid coordinates.
        da_cmls: xarray.Dataset
            CML observations. Must contain the projected midpoint
            coordinates (x, y) as well as the rainfall measurements stored
            under variable name 'R'.
        da_gauges: xarray.Dataset
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
            da_grid, da_cmls, da_gauges
        )

        # Get correct order of observations
        observations = self._get_obs(da_cmls, da_gauges)

        # Grid specific information
        coord_pred = np.hstack(
            [da_grid.y_grid.data.reshape(-1, 1), da_grid.x_grid.data.reshape(-1, 1)]
        )

        # If few observations return zero grid
        if (~np.isnan(observations)).sum() <= self.min_observations:
            return xr.DataArray(
                data=np.full_like(da_grid.x_grid, np.nan),
                coords=da_grid.coords,
                dims=da_grid.dims,
            )

        # IDW interpolator invdisttree
        interpolated = self._interpolator(
            q=coord_pred,
            z=observations,
            nnear=observations.size if observations.size <= self.nnear else self.nnear,
            p=self.p,
            idw_method=self.idw_method,
            max_distance=self.max_distance,
        ).reshape(da_grid.x_grid.shape)

        return xr.DataArray(data=interpolated, coords=da_grid.coords, dims=da_grid.dims)


class InterpolateOrdinaryKriging(Interpolator):
    """Interpolate CML and rain gauge using neighbourhood ordinary kriging

    Interpolates the provided CML and rain gauge observations using
    ordinary kriging. The class defaults to interpolation using neighbouring
    observations. It also by default uses the full line geometry for
    interpolation, but can treat the lines as points by setting full_line
    to False.
    """

    def __init__(
        self,
        ds_grid,
        ds_cmls=None,
        ds_gauges=None,
        variogram_model="spherical",
        variogram_parameters=None,
        discretization=8,
        min_observations=1,
        nnear=8,
        max_distance=60000,
        full_line=True,
    ):
        """Initialize interpolator object

        Parameters
        ----------
        ds_grid: xarray.Dataset
            Dataset providing the grid for interpolation. Must contain
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
        min_observations:
            Number of observations needed to perform interpolation.
        nnear: int
            Number of closest links to use for interpolation
        max_distance: float
            Largest distance allowed for including an observation.
        full_line: bool
            Whether to use the full line for block kriging. If set to false, the
            x0 geometry is reformatted to simply reflect the midpoint of the CML.
        """
        self.discretization = discretization
        self.min_observations = min_observations
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

        Interpolator.__init__(self)
        self._interpolator = self._maybe_update_interpolator(
            ds_grid, ds_cmls, ds_gauges
        )

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

    def _get_obs_sigma(
        self, da_cmls=None, da_gauges=None, da_cmls_sigma=None, da_gauges_sigma=None
    ):
        if da_gauges is not None:
            obs_gauges = da_gauges.data.ravel()
            if da_gauges_sigma is not None:
                sigma_gauges = da_gauges_sigma.data.ravel()
            else:
                sigma_gauges = np.zeros(obs_gauges.size)
        else:
            obs_gauges = []
            sigma_gauges = []

        if da_cmls is not None:
            obs_cmls = da_cmls.data.ravel()
            if da_cmls_sigma is not None:
                sigma_cmls = da_cmls_sigma.data.ravel()
            else:
                sigma_cmls = np.zeros(obs_cmls.size)
        else:
            obs_cmls = []
            sigma_cmls = []

        # Stack observations and sigma in the order expected by interpolator
        obs = np.concatenate([obs_cmls, obs_gauges]).astype(float)
        sigma = np.concatenate([sigma_cmls, sigma_gauges]).astype(float)

        return obs, sigma

    def __call__(
        self,
        da_grid,
        da_cmls=None,
        da_gauges=None,
        da_cmls_sigma=None,
        da_gauges_sigma=None,
    ):
        """Interpolate observations for one time step.

        Interpolates ground observations for one time step.

        Parameters
        ----------
        da_grid: xarray.DataArray
            Dataarray providing the grid for interpolation. Must contain
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
            da_grid, da_cmls, da_gauges
        )

        # Get correct order of observations and sigma (if sigma is present)
        obs, sigma = self._get_obs_sigma(
            da_cmls, da_gauges, da_cmls_sigma, da_gauges_sigma
        )

        # If few observations return grid of nans
        if (~np.isnan(obs)).sum() <= self.min_observations:
            return xr.DataArray(
                data=np.full_like(da_grid.x_grid, np.nan),
                coords=da_grid.coords,
                dims=da_grid.dims,
            )

        # Neighbourhood kriging with uncertainty
        interpolated = self._interpolator(obs, sigma).reshape(da_grid.x_grid.shape)

        return xr.DataArray(data=interpolated, coords=da_grid.coords, dims=da_grid.dims)
