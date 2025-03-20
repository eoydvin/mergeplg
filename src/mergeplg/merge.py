"""Module for merging CML and radar data."""

from __future__ import annotations

import numpy as np
import xarray as xr

from mergeplg import merge_functions
from mergeplg import interpolate_functions
from mergeplg.base import Base


class MergeDifferenceIDW(Base):
    """Merge ground and radar difference using IDW.

    Merges the provided radar field in ds_rad with gauge or CML observations
    by interpolating the difference (additive or multiplicative) 
    between the ground and radar observations using IDW. 
    """

    def __init__(
        self,
        grid_location_radar="center",
    ):
        Base.__init__(self, grid_location_radar)

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
        method='additive'
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

        Returns
        -------
        da_rad_out: xarray.DataArray
            DataArray with the same structure as the ds_rad but with the CML
            adjusted radar field.
        """
        # Update weights and x0 geometry for CML and gauge
        self.update(da_rad, da_cml=da_cml, da_gauge=da_gauge)

        # Evaluate radar at cml and gauge ground positions
        rad, obs, x0 = self.get_rad_obs_x0_(da_rad, da_cml=da_cml, da_gauge=da_gauge)

        # Calculate radar-ground difference if radar observes rainfall
        if method == 'additive':
            diff = np.where(rad>0, obs- rad, np.nan)
            keep = np.where(~np.isnan(diff))[0]

        elif method == 'multiplicative':
            mask_zero = rad > 0.0
            diff = np.full_like(obs, np.nan, dtype=np.float64)
            diff[mask_zero] = obs[mask_zero]/rad[mask_zero]
            keep = np.where((~np.isnan(diff)) & (diff < np.nanquantile(diff, 0.95)))[0]

        else:
            msg = "Method must be multiplicative or additive"
            raise ValueError(msg)

        # Interpolate the difference
        interpolated = interpolate_functions.interpolate_idw(
            da_rad.xs.data, 
            da_rad.ys.data,
            diff[keep],
            x0[keep, :],
            p=p,
            idw_method=idw_method,
            nnear=nnear,
            max_distance=max_distance,
        )
        
        # Adjust radar field
        if method == 'additive':
            adjusted = interpolated + da_rad.isel(time = 0).data
        elif method == 'multiplicative':
            adjusted = interpolated*da_rad.isel(time = 0).data

        # Remove negative values
        adjusted[adjusted < 0] = 0

        return xr.DataArray(
            data=[adjusted],
            coords=da_rad.coords,
            dims=da_rad.dims
        )

class MergeDifferenceBlockKriging(Base):
    """Merge CML and radar using block kriging

    Merges the provided radar field in ds_rad with gauge or CML observations
    by interpolating the difference (additive or multiplicative) 
    between the ground and radar observations using Block Kriging.
    """

    def __init__(
        self,
        grid_location_radar="center",
        discretization=8,
    ):
        Base.__init__(self, grid_location_radar)

        # Number of discretization points along CML
        self.discretization = discretization

        # For storing variogram parameters
        self.variogram_param = None

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
        variogram="exponential", 
        nnear=8,
        max_distance=60000,
        full_line = True,
        method='additive'
    ):
        """Interpolate observations for one time step.

        Interpolates ground observations for one time step. The function assumes that the
        x0 are updated using the update class method.

        Parameters
        ----------
        da_rad: xarray.DataArray
            Gridded radar data. Must contain the lon and lat coordinates as
            well as the projected coordinates xs and ys as a meshgrid.
        da_cml: xarray.DataArray
            CML observations. Must contain the projected midpoint 
            coordinates (x, y).
        da_gauge: xarray.DataArray
            Gauge observations. Must contain the projected 
            coordinates (x, y).
        variogram: function or str
            If function: Must return expected variance given distance between
            observations. If string: Must be a valid variogram type in pykrige.
        nnear: int
            Number of closest links to use for interpolation
        max_distance: float
            Largest distance allowed for including an observation.
        full_line: bool
            Wether to use the full line for block kriging. If set to false, the 
            x0 geometry is reformated to simply reflect the midpoint of the CML. 
        method: str
            Set to 'additive' to use additive approach, or 'multiplicative' to
            use the multiplicative approach.

        Returns
        -------
        da_field_out: xarray.DataArray
            DataArray with the same structure as the ds_rad but with the
            interpolated field. 
        """
        # Update weights and x0 geometry for CML and gauge
        self.update(da_rad, da_cml=da_cml, da_gauge=da_gauge)

        # Evaluate radar at cml and gauge ground positions
        rad, obs, x0 = self.get_rad_obs_x0_(da_rad, da_cml=da_cml, da_gauge=da_gauge)

        # Calculate radar-ground difference if radar observes rainfall
        if method == 'additive':
            diff = np.where(rad>0, obs- rad, np.nan)
            keep = np.where(~np.isnan(diff))[0]

        elif method == 'multiplicative':
            mask_zero = rad > 0.0
            diff = np.full_like(obs, np.nan, dtype=np.float64)
            diff[mask_zero] = obs[mask_zero]/rad[mask_zero]
            keep = np.where((~np.isnan(diff)) & (diff < np.nanquantile(diff, 0.95)))[0]

        else:
            msg = "Method must be multiplicative or additive"
            raise ValueError(msg)
        
        # If variogram provided as string, estimate from ground obs.
        if isinstance(variogram, str):
            # Estimate variogram
            param = interpolate_functions.estimate_variogram(
                obs=diff[keep],
                x0=x0[keep],
            )

            variogram, self.variogram_param = param

        # If n_closest is provided as an integer
        if nnear != False:
            # Interpolate using neighbourhood block kriging
            interpolated = interpolate_functions.interpolate_neighbourhood_block_kriging(
                da_rad.xs.data,
                da_rad.ys.data,
                diff[keep], 
                x0[keep, :] if full_line else x0[keep, :, [int(x0.shape[1] / 2)]],
                variogram,
                diff[keep].size - 1 if diff[keep].size <= nnear else nnear,
            )

        # If n_closest is set to False, use full kriging matrix
        else:
            # Interpolate using block kriging
            interpolated = interpolate_functions.interpolate_block_kriging(
                da_rad.xs.data,
                da_rad.ys.data,
                diff[keep],
                x0[keep, :] if full_line else x0[keep, :, [int(x0.shape[1] / 2)]],
                variogram,
            )
        
        # Adjust radar field
        if method == 'additive':
            adjusted = interpolated + da_rad.isel(time = 0).data
        elif method == 'multiplicative':
            adjusted = interpolated*da_rad.isel(time = 0).data


        # Remove negative values
        adjusted[adjusted < 0] = 0

        return xr.DataArray(
            data=[adjusted],
            coords=da_rad.coords,
            dims=da_rad.dims
        )

class MergeBlockKrigingExternalDrift(Base):
    """Merge CML and radar using block-kriging with external drift.

    Merges the provided radar field in ds_rad to CML and rain gauge
    observations by using a block kriging variant of kriging with external
    drift.
    """

    def __init__(
        self,
        grid_location_radar="center",
        discretization=8,
    ):
        Base.__init__(self, grid_location_radar)

        # Number of discretization points along CML
        self.discretization = discretization

    def update(self, da_rad, da_cml=None, da_gauge=None):
        """Update weights and x0 geometry for CML and gauge assuming block data

        This function uses the full CML geometry and makes the gauge geometry
        appear as a rain gauge.
        """
        self.update_weights_(da_rad, da_cml=da_cml, da_gauge=da_gauge)
        self.update_x0_block_(self.discretization, da_cml=da_cml, da_gauge=da_gauge)

    def adjust(
        self, da_rad, da_cml=None, da_gauge=None, variogram="exponential", n_closest=8
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
        variogram: function
            If function: Must return expected variance given distance between
            observations. If string: Must be a valid variogram type in pykrige.
        n_closest: int
            Number of closest links to use for interpolation

        Returns
        -------
        da_rad_out: xarray.DataArray
            DataArray with the same structure as the ds_rad but with the CML
            adjusted radar field.
        """
        # Update weights and x0 geometry for CML and gauge
        self.update(da_rad, da_cml=da_cml, da_gauge=da_gauge)

        # Evaluate radar at cml and gauge ground positions
        rad, obs, x0 = self.get_rad_obs_x0_(da_rad, da_cml=da_cml, da_gauge=da_gauge)

        # Get index of not-nan obs
        keep = np.where(~np.isnan(obs) & ~np.isnan(rad) & (obs > 0) & (rad > 0))[0]

        # If variogram provided as string, estimate from ground obs.
        if isinstance(variogram, str):
            # Estimate variogram
            param = interpolate_functions.estimate_variogram(
                obs=obs[keep],
                x0=x0[keep],
            )

            variogram, self.variogram_param = param

        # get timestamp
        time = da_rad.time.data[0]

        # Remove radar time dimension
        rad_field = da_rad.sel(time=time).data

        # Set zero values to nan, these are ignored in ked function
        rad_field[rad_field <= 0] = np.nan

        # do addtitive IDW merging
        adjusted = merge_functions.merge_ked_blockkriging(
            rad_field,
            da_rad.xs.data,
            da_rad.ys.data,
            rad[keep],
            obs[keep],
            x0[keep],
            variogram,
            obs[keep].size - 1 if obs[keep].size <= n_closest else n_closest,
        )

        # Remove negative values
        adjusted[adjusted < 0] = 0

        return xr.DataArray(
            data=[adjusted],
            coords=da_rad.coords,
            dims=da_rad.dims
        )

