"""Module for merging CML and radar data."""

from __future__ import annotations

import numpy as np
import xarray as xr

from mergeplg import merge_functions
from mergeplg.base import Base


class MergeDifferenceIDW(Base):
    """Merge CML and radar difference using IDW (CML midpoint).

    Merges the provided radar field in ds_rad to CML observations by
    interpolating the difference/ratio between the ground and radar 
    observations using IDW. 
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
        rad, obs, x0 = self.radar_at_ground_(da_rad, da_cml=da_cml, da_gauge=da_gauge)

        # Calculate radar-ground difference if radar observes rainfall
        if method == 'additive':
            diff = np.where(rad>0, obs- rad, np.nan)
            keep = np.where(~np.isnan(diff))[0]
        elif method = 'multiplicative':
            diff = np.where(rad>0, obs/rad, np.nan)
            keep = np.where((~np.isnan(diff)) & (diff < np.nanquantile(diff, 0.95)))[0]

        # If n_closest is provided as an integer
        if n_closest != False:
            # Interpolate using neighbourhood block kriging
            interpolated = interpolate_functions.interpolate_neighbourhood_block_kriging(
                xgrid,
                ygrid,
                obs[keep] if full_line else x0[keep, :, [int(x0.shape[1] / 2)]]
                x0[keep, :],
                variogram,
                diff[keep].size - 1 if diff[keep].size <= n_closest else n_closest,
                max_distance=max_distance,
            )

        # If n_closest is set to False, use full kriging matrix
        else:
            # Interpolate using block kriging
            interpolated = interpolate_functions.interpolate_block_kriging(
                xgrid,
                ygrid,
                obs[keep] if full_line else x0[keep, :, [int(x0.shape[1] / 2)]]
                x0[keep, :],
                variogram,
            )
        
        # interpolate radar-ground difference
        adjusted = merge_functions.merge_multiplicative_idw(
            xr.where(da_rad_t > 0, da_rad_t, np.nan),  # function skips nan
            diff[keep],
            x0[keep, :],
            p=p,
            idw_method=idw_method,
            nnear=nnear,
            max_distance=max_distance,
        )

        # Replace nan with original radar data (so that da_rad nan is kept)
        adjusted = xr.where(np.isnan(adjusted), da_rad_t, adjusted)

        # Re-assign timestamp and return
        return adjusted.assign_coords(time=time)

class MergeAdditiveBlockKriging(Base):
    """Merge CML and radar using additive block kriging

    Merges the provided radar field in ds_rad to CML and rain gauge
    observations by interpolating the difference between radar and ground
    observations using block kriging.
    """

    def __init__(
        self,
        grid_location_radar="center",
        discretization=8,
    ):
        Base.__init__(self, grid_location_radar, min_obs)

        # Number of discretization points along CML
        self.discretization = discretization

        # For storing variogram parameters
        self.variogram_param = None

    def update(self, da_rad, da_cml=None, da_gauge=None):
        """Update weights and x0 geometry for CML and gauge assuming block data

        This function uses the full CML geometry and makes the gauge geometry
        appear as a rain gauge.
        """
        self.update_weights_block_(da_rad, da_cml=da_cml, da_gauge=da_gauge)
        self.update_x0_block_(self.discretization, da_cml=da_cml, da_gauge=da_gauge)

    def adjust(
        self, da_rad, da_cml=None, da_gauge=None, variogram="exponential", n_closest=8
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
            projected coordinates (site_0_x, site_0_y, site_1_x, site_1_y).
        da_gauge: xarray.DataArray
            Gauge observations. Must contain the coordinates for the rain gauge
            positions (lat, lon) as well as the projected coordinates (x, y).
        variogram: function or str
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
        rad, obs, x0 = self.radar_at_ground_(da_rad, da_cml=da_cml, da_gauge=da_gauge)

        # Calculate radar-instrument difference if radar has observation
        diff = np.where(rad > 0, obs - rad, np.nan)

        # Get index of not-nan obs
        keep = np.where(~np.isnan(diff))[0]

        # If variogram provided as string, estimate from ground obs.
        if isinstance(variogram, str):
            # Estimate variogram
            param = merge_functions.estimate_variogram(
                obs=obs[keep],
                x0=x0[keep],
            )

            variogram, self.variogram_param = param

        # get timestamp
        time = da_rad.time.data[0]

        # Remove radar time dimension
        da_rad_t = da_rad.sel(time=time)

        # do addtitive IDW merging
        adjusted = merge_functions.merge_additive_blockkriging(
            xr.where(da_rad_t > 0, da_rad_t, np.nan),  # function skips nan
            diff[keep],
            x0[keep, :],
            variogram,
            diff[keep].size - 1 if diff[keep].size <= n_closest else n_closest,
        )

        # Replace nan with original radar data (so that da_rad nan is kept)
        adjusted = xr.where(np.isnan(adjusted), da_rad_t, adjusted)

        # Re-assign timestamp and return
        return adjusted.assign_coords(time=time)

class MergeBlockKrigingExternalDrift(Base):
    """Merge CML and radar using block-kriging with external drift.

    Merges the provided radar field in ds_rad to CML and rain gauge
    observations by using a block kriging variant of kriging with external
    drift.
    """

    def __init__(
        self,
        grid_location_radar="center",
        min_obs=5,
        discretization=8,
    ):
        Base.__init__(self, grid_location_radar, min_obs)

        # Number of discretization points along CML
        self.discretization = discretization

        # For storing pykrige variogram parameters
        self.variogram_param = None

        # For storing gamma parameters
        self.gamma_param = None

    def update(self, da_rad, da_cml=None, da_gauge=None):
        """Update weights and x0 geometry for CML and gauge assuming block data

        This function uses the full CML geometry and makes the gauge geometry
        appear as a rain gauge.
        """
        self.update_weights_block_(da_rad, da_cml=da_cml, da_gauge=da_gauge)
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
        rad, obs, x0 = self.radar_at_ground_(da_rad, da_cml=da_cml, da_gauge=da_gauge)

        # Get index of not-nan obs
        keep = np.where(~np.isnan(obs) & ~np.isnan(rad) & (obs > 0) & (rad > 0))[0]

        # Check that that there is enough observations
        if keep.size > self.min_obs_:
            # If variogram provided as string, estimate from ground obs.
            if isinstance(variogram, str):
                # Estimate variogram
                param = merge_functions.estimate_variogram(
                    obs=obs[keep],
                    x0=x0[keep],
                )

                variogram, self.variogram_param = param

            # get timestamp
            time = da_rad.time.data[0]

            # Remove radar time dimension
            da_rad_t = da_rad.sel(time=time)

            # do addtitive IDW merging
            adjusted = merge_functions.merge_ked_blockkriging(
                xr.where(da_rad_t > 0, da_rad_t, np.nan),  # function skips nan
                rad[keep],
                obs[keep],
                x0[keep],
                variogram,
                obs[keep].size - 1 if obs[keep].size <= n_closest else n_closest,
            )

            # Replace nan with original radar data (so that da_rad nan is kept)
            adjusted = xr.where(np.isnan(adjusted), da_rad_t, adjusted)

            # Re-assign timestamp and return
            return adjusted.assign_coords(time=time)

        # Else return the unadjusted radar
        return da_rad
