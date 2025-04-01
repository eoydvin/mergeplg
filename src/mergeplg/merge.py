"""Module for merging CML and radar data."""

from __future__ import annotations

import numpy as np
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
        min_observations=1,
        p=2,
        idw_method="radolan",
        nnear=8,
        max_distance=60000,
        method="additive",
        keep_function=None,
    ):
        """
        p: float
            IDW interpolation parameter
        idw_method: str
            by default "radolan"
        nnear: int
            number of neighbours to use for interpolation
        max_distance: float
            max distance allowed interpolation distance
        """
        Base.__init__(self, grid_location_radar)
        self.min_observations = min_observations
        self.p = p
        self.idw_method=idw_method
        self.nnear=nnear
        self.max_distance=max_distance
        self.method = method
        self.keep_function=keep_function

    def update(self, da_rad, da_cml=None, da_gauge=None):
        """ Update weights and init interpolator.

        Checks cml and gauge names from previous run. Return observations
        in correct order. 
        """
        
        self.update_weights_(da_rad, da_cml=da_cml, da_gauge=da_gauge)
        self.update_interpolator_idw_(da_cml=da_cml, da_gauge=da_gauge)

    def adjust(
        self,
        da_rad,
        da_cml=None,
        da_gauge=None,
    ):
        """Adjust radar field for one time step.

        Adjust radar field for one time step. The function assumes that the
        weights are updated using the update class method.

        Parameters
        ----------
        da_rad: xarray.DataArray
            Gridded radar data. Must contain the coordinates x_grid and y_grid.
        da_cml: xarray.DataArray
            CML observations. Must contain the projected coordinates for the CML
            (site_0_x, site_0_y, site_1_x, site_1_y) as well as the
            projected midpoint coordinates (x, y).
        da_gauge: xarray.DataArray
            Gauge observations. Must contain the projected coordinates (x, y).
        
        Returns
        -------
        da_rad_out: xarray.DataArray
            DataArray with the same structure as the ds_rad but with the CML
            adjusted radar field.
        """
        # Update function with new weights
        self.update(da_rad, da_cml=da_cml, da_gauge=da_gauge)

        # Evaluate radar at cml and gauge ground positions
        rad, obs = self.get_grid_obs_(da_rad, da_cml=da_cml, da_gauge=da_gauge)

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
        
        keep = ~np.isnan(diff)
        
        import matplotlib.pyplot as plt
        plt.scatter(self.y[keep], self.x[keep], diff[keep], markersize=40)
        plt.show()

        # IDW interpolator invdisttree
        interpolated = self.interpolator(
            q=coord_pred,
            z=diff,
            nnear=obs.size if obs.size <= self.nnear else self.nnear,
            p=self.p,
            idw_method=self.idw_method,
            max_distance=self.max_distance,
        ).reshape(da_rad.x_grid.shape)

        # Adjust radar field
        if self.method == "additive":
            adjusted = interpolated + da_rad
            adjusted.data[adjusted < 0] = 0
        elif self.method == "multiplicative":
            adjusted = interpolated * da_rad
            adjusted.data[adjusted < 0] = 0

        return adjusted


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
        Parameters
        ----------
        variogram_model: str
            Must be a valid variogram type in pykrige.
        variogram_parameters: str
            Must be a valid parameters corresponding to variogram_model.
        nnear: int
            Number of closest links to use for interpolation
        max_distance: float
            Largest distance allowed for including an observation.
        full_line: bool
            Whether to use the full line for block kriging. If set to false, the
            x0 geometry is reformatted to simply reflect the midpoint of the CML.
        """
        Base.__init__(self, grid_location_radar)

        self.discretization = discretization
        self.min_observations = min_observations
        self.nnear = nnear
        self.max_distance = max_distance
        self.full_line=full_line
        self.method=method

        # Construct variogram using parameters provided by user
        if variogram_parameters is None:
            variogram_parameters = {"sill": 0.9, "range": 5000, "nugget": 0.1}
        obs = np.array([0, 1]) # dummy variables
        coord = np.array([[0, 1], [1, 0]])

        self.variogram = bk_functions.construct_variogram(
            obs, coord, variogram_parameters, variogram_model
        )

    def update(self, da_rad, da_cml=None, da_gauge=None):
        """Update weights and x0 geometry for CML and gauge assuming block data

        This function uses the full CML geometry and makes the gauge geometry
        appear as a rain gauge.
        """
        self.update_weights_(da_rad, da_cml=da_cml, da_gauge=da_gauge)
        self.update_interpolator_obk_(da_cml, da_gauge)

    def adjust(
        self,
        da_rad,
        da_cml=None,
        da_gauge=None,
    ):
        """Interpolate observations for one time step.

        Interpolates ground observations for one time step. The function assumes
        that the x0 are updated using the update class method.

        Parameters
        ----------
        da_rad: xarray.DataArray
            Gridded rainfall data. Must contain the  projected coordinates x_grid and
            y_grid as a meshgrid.
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

        # Update weights and x0 geometry for CML and gauge
        self.update(da_rad, da_cml=da_cml, da_gauge=da_gauge)

        # Evaluate radar at cml and gauge ground positions
        rad, obs = self.get_grid_obs_(da_rad, da_cml=da_cml, da_gauge=da_gauge)

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

        # Points to interpolate
        points = np.hstack([
            da_rad.y_grid.data.reshape(-1, 1),
            da_rad.x_grid.data.reshape(-1, 1),
        ])

        # Neighbourhood kriging
        interpolated = self.interpolator(
            points,
            diff
        ).reshape(da_rad.x_grid.shape)

        # Adjust radar field
        if self.method == "additive":
            adjusted = interpolated + da_rad
            adjusted.data[adjusted < 0] = 0
        elif self.method == "multiplicative":
            adjusted = interpolated * da_rad
            adjusted.data[adjusted < 0] = 0

        return adjusted


class MergeKrigingExternalDrift(Base):
    """Merge CML and radar using kriging with external drift.

    Merges the provided radar field in ds_rad to CML and rain gauge
    observations by using a block kriging variant of kriging with external
    drift.
    """

    def __init__(
        self,
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
        Parameters
        ----------
        variogram_model: str
            Must be a valid variogram type in pykrige.
        variogram_parameters: str
            Must be a valid parameters corresponding to variogram_model.
        nnear: int
            Number of closest links to use for interpolation
        max_distance: float
            Largest distance allowed for including an observation.
        full_line: bool
            Whether to use the full line for block kriging. If set to false, the
            x0 geometry is reformatted to simply reflect the midpoint of the CML.
        """
        Base.__init__(self, grid_location_radar)

        self.discretization = discretization
        self.min_observations = min_observations
        self.nnear = nnear
        self.max_distance = max_distance
        self.full_line=full_line
        self.method=method

        # Construct variogram using parameters provided by user
        if variogram_parameters is None:
            variogram_parameters = {"sill": 0.9, "range": 5000, "nugget": 0.1}
        obs = np.array([0, 1]) # dummy variables
        coord = np.array([[0, 1], [1, 0]])

        self.variogram = bk_functions.construct_variogram(
            obs, coord, variogram_parameters, variogram_model
        )

    def update(self, da_rad, da_cml=None, da_gauge=None):
        """Update weights and x0 geometry for CML and gauge assuming block data

        This function uses the full CML geometry and makes the gauge geometry
        appear as a rain gauge.
        """
        self.update_weights_(da_rad, da_cml=da_cml, da_gauge=da_gauge)
        self.update_interpolator_ked_(da_cml, da_gauge)

    def adjust(
        self,
        da_rad,
        da_cml=None,
        da_gauge=None,
    ):
        """Adjust radar field for one time step.

        Evaluates if we have enough observations to adjust the radar field.
        Then adjust radar field to observations using a block kriging variant
        of kriging with external drift.

        Parameters
        ----------
        da_rad: xarray.DataArray
            Gridded radar data. Must contain the projected coordinates
            xs and ys as a meshgrid.
        da_cml: xarray.DataArray
            CML observations. Must contain the projected coordinates
            (site_0_x, site_0_y, site_1_x, site_1_y).
        da_gauge: xarray.DataArray
            Gauge observations. Must contain the projected coordinates (x, y).

        Returns
        -------
        da_rad_out: xarray.DataArray
            DataArray with the same structure as the ds_rad but with the CML
            adjusted radar field.
        """

        # Update 
        self.update(da_rad, da_cml=da_cml, da_gauge=da_gauge)

        # Evaluate radar at cml and gauge ground positions
        rad_obs, obs = self.get_grid_obs_(da_rad, da_cml=da_cml, da_gauge=da_gauge)

        # Default decision on which observations to ignore
        ignore = np.isnan(rad_obs) & (obs == 0) & (rad_obs == 0)
        obs[ignore] = np.nan # obs nan ar ignored in KED function

        # If few observations return zero grid
        if (~np.isnan(obs)).sum() <= self.min_observations:
            return da_rad

        # Remove radar time dimension
        rad_field = da_rad.isel(time=0).data if "time" in da_rad.dims else da_rad.data

        # Set zero values to nan, these are ignored in ked function
        rad_field[rad_field <= 0] = np.nan

        # Points to interpolate (correponds to radar)
        points = np.hstack([
            da_rad.y_grid.data.reshape(-1, 1),
            da_rad.x_grid.data.reshape(-1, 1),
        ])


        # do KED merging
        adjusted = self.interpolator(
            points,
            rad_field.ravel(),
            obs,
            rad_obs
        ).reshape(da_rad.x_grid.shape)


        # Remove negative values
        adjusted[(adjusted < 0) | np.isnan(adjusted)] = 0

        if "time" in da_rad.dims:
            da_adjusted = xr.DataArray(
                data=[adjusted], coords=da_rad.coords, dims=da_rad.dims
            )
        else:
            da_adjusted = xr.DataArray(
                data=adjusted, coords=da_rad.coords, dims=da_rad.dims
            )
        return da_adjusted
