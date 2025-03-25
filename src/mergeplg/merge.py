"""Module for merging CML and radar data."""

from __future__ import annotations

import numpy as np
import pykrige
import xarray as xr

from mergeplg import bk_functions
from mergeplg.base import Base

from .radolan import idw


class MergeDifferenceIDW(Base):
    """Merge ground and radar difference using IDW.

    Merges the provided radar field in ds_rad with gauge or CML observations
    by interpolating the difference (additive or multiplicative)
    between the ground and radar observations using IDW.
    """

    def __init__(
        self,
        grid_location_radar="center",
        min_observations=5,
    ):
        Base.__init__(self, grid_location_radar)

        # Minimum number of observations needed to perform merging
        self.min_observations = min_observations

    def update(self, da_rad, da_cml=None, da_gauge=None):
        """Update weights and x0 geometry for CML and gauge

        This function uses the midpoint of the CML as CML reference.
        """
        # Update x0 and radar weights
        self.update_x0_(da_cml=da_cml, da_gauge=da_gauge)
        self.update_weights_(da_rad, da_cml=da_cml, da_gauge=da_gauge)

    def adjust(
        self,
        da_rad,
        da_cml=None,
        da_gauge=None,
        p=2,
        idw_method="radolan",
        nnear=8,
        max_distance=60000,
        method="additive",
        keep_function=None,
    ):
        """Adjust radar field for one time step.

        Adjust radar field for one time step. The function assumes that the
        weights are updated using the update class method.

        Parameters
        ----------
        da_rad: xarray.DataArray
            Gridded radar data. Must contain the lon and lat coordinates as
            well as the projected coordinates xs and ys as a meshgrid.
        da_cml: xarray.DataArray
            CML observations. Must contain the lat/lon coordinates for the CML
            (site_0_lon, site_0_lat, site_1_lon, site_1_lat) as well as the
            projected midpoint coordinates (x, y).
        da_gauge: xarray.DataArray
            Gauge observations. Must contain the coordinates for the rain gauge
            positions (lat, lon) as well as the projected coordinates (x, y).
        p: float
            IDW interpolation parameter
        idw_method: str
            by default "radolan"
        n_closest: int
            Number of neighbours to use for interpolation.
        max_distance: float
            max distance allowed interpolation distance
        method: str
            Set to 'additive' to use additive approach, or 'multiplicative' to
            use the multiplicative approach.
        keep_function: function
            Function that evaluates what differences to keep or not

        Returns
        -------
        da_rad_out: xarray.DataArray
            DataArray with the same structure as the ds_rad but with the CML
            adjusted radar field.
        """
        # Update weights and x0 geometry for CML and gauge
        self.update(da_rad, da_cml=da_cml, da_gauge=da_gauge)

        # Evaluate radar at cml and gauge ground positions
        rad, obs, x0 = self.get_grid_obs_x0_(da_rad, da_cml=da_cml, da_gauge=da_gauge)

        # Calculate radar-ground difference
        if method == "additive":
            diff = np.where(rad > 0, obs - rad, np.nan)

        elif method == "multiplicative":
            mask_zero = rad > 0.0
            diff = np.full_like(obs, np.nan, dtype=np.float64)
            diff[mask_zero] = obs[mask_zero] / rad[mask_zero]

        else:
            msg = "Method must be multiplicative or additive"
            raise ValueError(msg)

        # Default decision on which observations to keep
        if not keep_function:
            keep = np.where(~np.isnan(diff))[0]
        else:
            keep = keep_function([diff, rad, obs, x0])

        # Return gridded data if too few observations
        if obs[keep].size <= self.min_observations:
            return da_rad

        # Ensure same functionality as in kriging
        if not nnear:
            nnear = obs[keep].size

        # Coordinates to predict
        coord_pred = np.hstack(
            [da_rad.ys.data.reshape(-1, 1), da_rad.xs.data.reshape(-1, 1)]
        )

        # IDW interpolator invdisttree
        idw_interpolator = idw.Invdisttree(x0[keep])
        interpolated = idw_interpolator(
            q=coord_pred,
            z=diff[keep],
            nnear=obs[keep].size if obs[keep].size <= nnear else nnear,
            p=p,
            idw_method=idw_method,
            max_distance=max_distance,
        ).reshape(da_rad.xs.shape)

        # Adjust radar field
        if method == "additive":
            adjusted = interpolated + da_rad.isel(time=0).data
            adjusted[adjusted < 0] = 0
        elif method == "multiplicative":
            adjusted = interpolated * da_rad.isel(time=0).data
            adjusted[adjusted < 0] = 0

        return xr.DataArray(data=[adjusted], coords=da_rad.coords, dims=da_rad.dims)


class MergeDifferenceOrdinaryKriging(Base):
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
        grid_location_radar="center",
        discretization=8,
        min_observations=5,
    ):
        Base.__init__(self, grid_location_radar)

        # Number of discretization points along CML
        self.discretization = discretization

        # Minimum number of observations needed to perform merging
        self.min_observations = min_observations

    def update(self, da_rad, da_cml=None, da_gauge=None):
        """Update weights and x0 geometry for CML and gauge assuming block data

        This function uses the full CML geometry and makes the gauge geometry
        appear as a rain gauge.
        """
        self.update_weights_(da_rad, da_cml=da_cml, da_gauge=da_gauge)
        self.update_x0_block_(self.discretization, da_cml=da_cml, da_gauge=da_gauge)

    def adjust(
        self,
        da_rad,
        da_cml=None,
        da_gauge=None,
        variogram_model="spherical",
        variogram_parameters=None,
        nnear=8,
        full_line=True,
        method="additive",
        keep_function=None,
    ):
        """Interpolate observations for one time step.

        Interpolates ground observations for one time step. The function assumes
        that the x0 are updated using the update class method.

        Parameters
        ----------
        da_rad: xarray.DataArray
            Gridded rainfall data. Must contain the lon and lat coordinates as
            well as the projected coordinates xs and ys as a meshgrid.
        da_cml: xarray.DataArray
            CML observations. Must contain the projected midpoint
            coordinates (x, y).
        da_gauge: xarray.DataArray
            Gauge observations. Must contain the projected
            coordinates (x, y).
        variogram_model: str
            Must be a valid variogram type in pykrige.
        variogram_parameters: str
            Must be valid parameters corresponding to variogram_model.
        nnear: int
            Number of closest links to use for interpolation
        full_line: bool
            Whether to use the full line for block kriging. If set to false, the
            x0 geometry is reformatted to simply reflect the midpoint of the CML.
        method: str
            Set to 'additive' to use additive approach, or 'multiplicative' to
            use the multiplicative approach.
        keep_function: function
            Function that evaluates what differences to keep or not

        Returns
        -------
        da_field_out: xarray.DataArray
            DataArray with the same structure as the ds_rad but with the
            interpolated field.
        """
        # Initialize variogram parameters
        if variogram_parameters is None:
            variogram_parameters = {"sill": 0.9, "range": 5000, "nugget": 0.1}

        # Update weights and x0 geometry for CML and gauge
        self.update(da_rad, da_cml=da_cml, da_gauge=da_gauge)

        # Evaluate radar at cml and gauge ground positions
        rad, obs, x0 = self.get_grid_obs_x0_(da_rad, da_cml=da_cml, da_gauge=da_gauge)

        # Calculate radar-ground difference
        if method == "additive":
            diff = np.where(rad > 0, obs - rad, np.nan)

        elif method == "multiplicative":
            mask_zero = rad > 0.0
            diff = np.full_like(obs, np.nan, dtype=np.float64)
            diff[mask_zero] = obs[mask_zero] / rad[mask_zero]

        else:
            msg = "Method must be multiplicative or additive"
            raise ValueError(msg)

        # Default decision on which observations to keep
        if not keep_function:
            keep = np.where(~np.isnan(diff))[0]
        else:
            keep = keep_function([diff, rad, obs, x0])

        # Return gridded data if too few observations
        if obs[keep].size <= self.min_observations:
            return da_rad

        # Force interpolator to use only midpoint, if specified by user
        x0 = x0[:, :, [int(x0.shape[2] / 2)]] if full_line is False else x0

        # Construct variogram using parameters provided by user
        variogram = bk_functions.construct_variogram(
            obs[keep], x0[keep], variogram_parameters, variogram_model
        )

        # If nnear is set to False, use all observations in kriging
        if not nnear:
            interpolated = bk_functions.interpolate_block_kriging(
                da_rad.xs.data,
                da_rad.ys.data,
                diff[keep],
                x0[keep],
                variogram,
            )

        # Else do neighbourhood kriging
        else:
            interpolated = bk_functions.interpolate_neighbourhood_block_kriging(
                da_rad.xs.data,
                da_rad.ys.data,
                diff[keep],
                x0[keep],
                variogram,
                diff[keep].size if diff[keep].size <= nnear else nnear,
            )

        # Adjust radar field
        if method == "additive":
            adjusted = interpolated + da_rad.isel(time=0).data
            adjusted[adjusted < 0] = 0
        elif method == "multiplicative":
            adjusted = interpolated * da_rad.isel(time=0).data
            adjusted[adjusted < 0] = 0

        return xr.DataArray(data=[adjusted], coords=da_rad.coords, dims=da_rad.dims)


class MergeKrigingExternalDrift(Base):
    """Merge CML and radar using kriging with external drift.

    Merges the provided radar field in ds_rad to CML and rain gauge
    observations by using a block kriging variant of kriging with external
    drift.
    """

    def __init__(
        self,
        grid_location_radar="center",
        discretization=8,
        min_observations=5,
    ):
        Base.__init__(self, grid_location_radar)

        # Number of discretization points along CML
        self.discretization = discretization

        # Minimum number of observations needed to perform merging
        self.min_observations = min_observations

    def update(self, da_rad, da_cml=None, da_gauge=None):
        """Update weights and x0 geometry for CML and gauge assuming block data

        This function initializes x0 on block form.
        """
        self.update_weights_(da_rad, da_cml=da_cml, da_gauge=da_gauge)
        self.update_x0_block_(self.discretization, da_cml=da_cml, da_gauge=da_gauge)

    def adjust(
        self,
        da_rad,
        da_cml=None,
        da_gauge=None,
        variogram_model="spherical",
        variogram_parameters=None,
        n_closest=8,
        keep_function=None,
    ):
        """Adjust radar field for one time step.

        Evaluates if we have enough observations to adjust the radar field.
        Then adjust radar field to observations using a block kriging variant
        of kriging with external drift.

        The function allows for the user to supply transformation,
        backtransformation and variogram functions.

        Parameters
        ----------
        da_rad: xarray.DataArray
            Gridded radar data. Must contain the lon and lat coordinates as
            well as the projected coordinates xs and ys as a meshgrid.
        da_cml: xarray.DataArray
            CML observations. Must contain the lat/lon coordinates for the CML
            (site_0_lon, site_0_lat, site_1_lon, site_1_lat) as well as the
            projected coordinates (site_0_x, site_0_y, site_1_x, site_1_y).
        da_gauge: xarray.DataArray
            Gauge observations. Must contain the coordinates for the rain gauge
            positions (lat, lon) as well as the projected coordinates (x, y).
        variogram_model: str
            Must be a valid variogram type in pykrige.
        variogram_parameters: str
            Must be a valid parameters corresponding to variogram_model.
        n_closest: int
            Number of closest links to use for interpolation
        keep_function: Function
            Function that decides what obserations to keep

        Returns
        -------
        da_rad_out: xarray.DataArray
            DataArray with the same structure as the ds_rad but with the CML
            adjusted radar field.
        """
        # Initialize variogram parameters
        if variogram_parameters is None:
            variogram_parameters = {"sill": 0.9, "range": 5000, "nugget": 0.1}

        # Update weights and x0 geometry for CML and gauge
        self.update(da_rad, da_cml=da_cml, da_gauge=da_gauge)

        # Evaluate radar at cml and gauge ground positions
        rad, obs, x0 = self.get_grid_obs_x0_(da_rad, da_cml=da_cml, da_gauge=da_gauge)

        # Default decision on which observations to keep
        if not keep_function:
            keep = np.where(~np.isnan(obs) & ~np.isnan(rad) & (obs > 0) & (rad > 0))[0]
        else:
            keep = keep_function([rad, obs, x0])

        # Return gridded data if too few observations
        if obs[keep].size <= self.min_observations:
            return da_rad

        # Setup pykrige with variogram parameters provided by user
        ok = pykrige.OrdinaryKriging(
            x0[keep, 1, int(x0.shape[2] / 2)],  # x-midpoint coordinate
            x0[keep, 0, int(x0.shape[2] / 2)],  # y-midpoint coordinate
            obs[keep],
            variogram_model=variogram_model,
            variogram_parameters=variogram_parameters,
        )

        # Construct variogram using pykrige
        def variogram(h):
            return ok.variogram_function(ok.variogram_model_parameters, h)

        # Remove radar time dimension
        rad_field = da_rad.isel(time=0).data

        # Set zero values to nan, these are ignored in ked function
        rad_field[rad_field <= 0] = np.nan

        # do addtitive IDW merging
        adjusted = bk_functions.merge_ked_blockkriging(
            rad_field,
            da_rad.xs.data,
            da_rad.ys.data,
            rad[keep],
            obs[keep],
            x0[keep],
            variogram,
            obs[keep].size if obs[keep].size <= n_closest else n_closest,
        )

        # Remove negative values
        adjusted[adjusted < 0] = 0

        return xr.DataArray(data=[adjusted], coords=da_rad.coords, dims=da_rad.dims)
