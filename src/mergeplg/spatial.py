"""Module for merging CML and radar data."""

from __future__ import annotations

import numpy as np
import xarray as xr

from mergeplg import merge_functions
from mergeplg import interpolate_functions
from mergeplg.base import Base


class InterpolateIDW(Base):
    """Interpolate CML and rain gauge using IDW (CML midpoint).
    """

    def __init__(
        self,
        grid_location_radar="center",
        min_obs=1,
    ):
        Base.__init__(self, grid_location_radar, min_obs)

    def update(self, da_cml=None, da_gauge=None):
        """Update x0 geometry for CML and gauge

        This function uses the midpoint of the CML as CML reference.
        """
        # Update x0 and radar weights
        self.update_x0_(da_cml=da_cml, da_gauge=da_gauge)

    def interpolate(
        self,
        da_rad=None,
        da_cml=None,
        da_gauge=None,
        xgrid=None,
        ygrid=None,
        p=2,
        idw_method="radolan",
        nnear=8,
        max_distance=60000,
    ):
        """Interpolate observations for one time step using IDW

        Interpolate observations for one time step. The function assumes that 
        the x0 are updated using the update class method.

        Parameters
        ----------
        da_grid: xarray.DataArray
            Dataframe providing the grid for interpolation. Must contain 
            projected xs and ys coordinates.
        da_cml: xarray.DataArray
            CML observations. Must contain the projected midpoint 
            coordinates (x, y).
        da_gauge: xarray.DataArray
            Gauge observations. Must contain the coordinates projected 
            coordinates (x, y).
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
        da_field_out: xarray.DataArray
            DataArray with the same structure as the ds_rad but with the
            interpolated field. 
        """
        # Update weights and x0 geometry for CML and gauge
        self.update(da_rad, da_cml=da_cml, da_gauge=da_gauge)

        # Evaluate radar at cml and gauge ground positions
        obs, x0 = self.get_x0_obs_(da_cml=da_cml, da_gauge=da_gauge)

        # Get index of not-nan obs
        keep = np.where(~np.isnan(obs))[0]

        # Get radar grid as numpy arrays
        xgrid, ygrid = da_rad.xs.data, da_rad.ys.data

        # Check that that there is enough observations
        if keep.size > self.min_obs_:
            # do addtitive IDW merging
            interpolated = merge_functions.merge_multiplicative_idw(
                xgrid, 
                ygrid,
                diff[keep],
                x0[keep, :],
                p=p,
                idw_method=idw_method,
                nnear=nnear,
                max_distance=max_distance,
            )
            return xr.DataArray(
                data=interpolated,
                coords=da_rad.coords,
                dims=da_rad.dims
            )

        # Else return zeros
        return  xr.DataArray(
            data=np.zeros(xgrid.shape),
            coords=da_rad.coords,
            dims=da_rad.dims
        )

class InterpolateNeighbourhoodBlockKriging(Base):
    """Interpolate CML and radar using neighbourhood block kriging

    Interpolates the provided CML and rain gauge observations using neibourhood
    block kriging.
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

        # For storing variogram parameters
        self.variogram_param = None

    def update(self, da_cml=None, da_gauge=None):
        """Update weights and x0 geometry for CML and gauge assuming block data

        This function uses the full CML geometry and makes the gauge geometry
        appear as a rain gauge.
        """
        self.update_x0_block_(self.discretization, da_cml=da_cml, da_gauge=da_gauge)

    def interpolate(
        self, 
        da_rad, 
        da_cml=None, 
        da_gauge=None, 
        variogram="exponential", 
        n_closest=8,
        max_distance=60000,
    ):
        """Interpolate observations for one time step.

        Interpolates ground observations for one time step. The function assumes that the
        x0 are updated using the update class method.

        Parameters
        ----------
        da_grid: xarray.DataArray
            Dataframe providing the grid for interpolation. Must contain 
            projected xs and ys coordinates.
        da_cml: xarray.DataArray
            CML observations. Must contain the projected midpoint 
            coordinates (x, y).
        da_gauge: xarray.DataArray
            Gauge observations. Must contain the coordinates projected 
            coordinates (x, y).
        variogram: function or str
            If function: Must return expected variance given distance between
            observations. If string: Must be a valid variogram type in pykrige.
        n_closest: int
            Number of closest links to use for interpolation

        Returns
        -------
        da_field_out: xarray.DataArray
            DataArray with the same structure as the ds_rad but with the
            interpolated field. 
        """
        # Update x0 geometry for CML and gauge
        self.update(da_cml=da_cml, da_gauge=da_gauge)

        # Evaluate radar at cml and gauge ground positions
        obs, x0 = self.get_x0_obs_(da_cml=da_cml, da_gauge=da_gauge)

        # Get index of not-nan obs
        keep = np.where(~np.isnan(obs))[0]

        # Get radar grid as numpy arrays
        xgrid, ygrid = da_rad.xs.data, da_rad.ys.data

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

            # do addtitive IDW merging
            interpolated = interpolate_functions.interpolate_neighbourhood_block_kriging(
                xgrid,
                ygrid,
                obs[keep]
                x0[keep, :],
                variogram,
                diff[keep].size - 1 if diff[keep].size <= n_closest else n_closest,
                max_distance=max_distance,
            )
            
            return xr.DataArray(
                data=interpolated,
                coords=da_rad.coords,
                dims=da_rad.dims
            )

        # Else return zeros
        return  xr.DataArray(
            data=np.zeros(xgrid.shape),
            coords=da_rad.coords,
            dims=da_rad.dims
        )

