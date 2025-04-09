"""Module for merging CML and radar data."""

from __future__ import annotations

import numpy as np
import xarray as xr

from mergeplg import bk_functions
from mergeplg.base import Base

from .radolan import idw

class Interpolator:
    def __init__(self):
        self.x_gauge = None
        self.y_gauge = None
        self.x_0_cml = None
        self.x_1_cml = None
        self.y_0_cml = None
        self.y_1_cml = None

    def __call__(self, da_gauges=None, da_cmls=None):
        self._interpolator = self._maybe_update_interpolator(da_gauges, da_cmls)
        return self._interpolator
    
    def _init_interpolator(self, da_gauges=None, da_cmls=None):
        # Needs to return the interpolator
        raise NotImplementedError()
    
    def _update_obs_and_interpolator(self, da_gauges=None, da_cmls=None):
        """ Update observations and interpolator
        
        Function returns the rain gauge and CML observations in the correct order
        and updates the interpolator function if the possitions changes. 
        
        Parameters
        ----------
        da_gauges: xarray.DataArray
            Gauge observations. Must contain the projected
            coordinates (x, y).
        da_cmls: xarray.DataArray
            CML observations. Must contain the projected coordinates (site_0_x, 
            site_1_x, site_0_y, site_1_y).
            
        Returns
        -------
        obs: np.array
            CML and rain gauge observations in the order (CML_1 ... CML_n, 
            RG_1 ... RG_n).
        self._interpolator: function
            Initilized interpolator function. 

        """
        # If CMLs and rain gauges are present
        if (da_gauges is not None) and (da_cmls is not None):
            # Store observations for current time step
            obs = np.concatenate(
                [da_cmls.data.ravel(), da_gauges.data.ravel()]
            )
            
            # Check if gauges are the same
            x_gauge = da_gauges.x.data
            y_gauge = da_gauges.y.data
            x_gauge_equal = np.array_equal(x_gauge, self.x_gauge) 
            y_gauge_equal = np.array_equal(y_gauge, self.y_gauge)
            gauges_equal = x_gauge_equal and y_gauge_equal
            
            # Check if CMLs are the same
            x_0_cml = da_cmls.site_0_x.data
            x_1_cml = da_cmls.site_1_x.data
            y_0_cml = da_cmls.site_0_y.data
            y_1_cml = da_cmls.site_1_y.data
            cml_x0_eq = np.array_equal(x_0_cml, self.x_0_cml)
            cml_x1_eq = np.array_equal(x_1_cml, self.x_1_cml)
            cml_y0_eq = np.array_equal(y_0_cml, self.y_0_cml)
            cml_y1_eq = np.array_equal(y_1_cml, self.y_1_cml)
            cmls_equal = cml_x0_eq and cml_x1_eq and cml_y0_eq and cml_y1_eq
            
            if not cmls_equal or not gauges_equal:
                # Update status
                self.x_gauge = x_gauge
                self.y_gauge = y_gauge
                self.x_0_cml = x_0_cml
                self.x_1_cml = x_1_cml
                self.y_0_cml = y_0_cml
                self.y_1_cml = y_1_cml
                return obs, self._init_interpolator(da_gauges, da_cmls)
            
            else:
                return obs, self._interpolator
        
        # If only rain gauges
        elif (da_gauges is not None):
            # Store observations for current time step
            obs = da_gauges.data.ravel()
            
            # Check if gauges are the same
            x_gauge = da_gauges.x.data
            y_gauge = da_gauges.y.data            
            x_gauge_equal = np.array_equal(x_gauge, self.x_gauge) 
            y_gauge_equal = np.array_equal(y_gauge, self.y_gauge)
            gauges_equal = x_gauge_equal and y_gauge_equal
            
            if not gauges_equal:
                # Update status
                self.x_gauge = x_gauge
                self.y_gauge = y_gauge
                self.x_0_cml = None
                self.x_1_cml = None
                self.y_0_cml = None
                self.y_1_cml = None  
                return obs, self._init_interpolator(da_gauges=da_gauges)
            else:
                return obs, self._interpolator
            
        # If only CMLs
        elif (da_cmls is not None):
            # Store observations for current time step
            obs = da_cmls.data.ravel()
            
            # Check if CMLs are the same
            x_0_cml = da_cmls.site_0_x.data
            x_1_cml = da_cmls.site_1_x.data
            y_0_cml = da_cmls.site_0_y.data
            y_1_cml = da_cmls.site_1_y.data            
            cml_x0_eq = np.array_equal(x_0_cml, self.x_0_cml)
            cml_x1_eq = np.array_equal(x_1_cml, self.x_1_cml)
            cml_y0_eq = np.array_equal(y_0_cml, self.y_0_cml)
            cml_y1_eq = np.array_equal(y_1_cml, self.y_1_cml)
            cmls_equal = cml_x0_eq and cml_x1_eq and cml_y0_eq and cml_y1_eq
            
            if not cmls_equal:
                # Update status
                self.x_gauge = None
                self.y_gauge = None
                self.x_0_cml = x_0_cml
                self.x_1_cml = x_1_cml
                self.y_0_cml = y_0_cml
                self.y_1_cml = y_1_cml
                return obs, self._init_interpolator(da_cmls=da_cmls)
            else:
                return obs, self._interpolator

class InterpolateIDW(Interpolator):
    """Interpolate CML and rain gauge using IDW
    
    Interpolates the provided CML and rain gauge observations using
    inverse distance weighting. The function uses the n nearest observations. 
    """    
    
    
    def __init__(
        self,
        min_observations=1,
        p=2,
        idw_method="standard",
        nnear=8,
        max_distance=60000,
        ):
        """
        Parameters
        ----------
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
        
        
        
        Interpolator.__init__(self)
        self.min_observations = min_observations
        self.p = p
        self.idw_method=idw_method
        self.nnear=nnear
        self.max_distance=max_distance

    def _init_interpolator(self, da_gauges=None, da_cmls=None):
        if (da_gauges is not None) and (da_cmls is not None):
            # Use midpoint of CML and rain gauge
            cml_x = 0.5*(da_cmls.site_0_x.data + da_cmls.site_1_x.data)
            cml_y = 0.5*(da_cmls.site_0_y.data + da_cmls.site_1_y.data)
            y = np.concatenate(cml_y, da_gauges.y.data)
            x = np.concatenate(cml_x, da_gauges.x.data)            
            yx = np.hstack([y.reshape(-1, 1), x.reshape(-1, 1)])
            
        elif da_gauges is not None:
            y = da_gauges.y.data
            x = da_gauges.x.data         
            yx = np.hstack([y.reshape(-1, 1), x.reshape(-1, 1)])
        
        elif da_cmls is not None:
            # Use midpoint of CML 
            cml_x = 0.5*(da_cmls.site_0_x.data + da_cmls.site_1_x.data)
            cml_y = 0.5*(da_cmls.site_0_y.data + da_cmls.site_1_y.data)
            yx = np.hstack([cml_y.reshape(-1, 1), cml_x.reshape(-1, 1)])
            
        return idw.Invdisttree(yx)        
    
    def __call__(self, da_grid, da_cmls=None, da_gauges=None):
        """Interpolate observations for one time step using IDW

        Interpolate observations for one time step. 

        Input data can have a time dimension of length 1 or no time dimension.

        Parameters
        ----------
        da_grid: xarray.DataArray
            Dataframe providing the grid for interpolation. Must contain
            projected x_grid and y_grid coordinates.
        da_cmls: xarray.DataArray
            CML observations. Must contain the projected midpoint
            coordinates (x, y).
        da_gauges: xarray.DataArray
            Gauge observations. Must contain the coordinates projected
            coordinates (x, y).

        Returns
        -------
        da_field_out: xarray.DataArray
            DataArray with the same coordinates as ds_rad but with the
            interpolated field.
        """        
        obs, self._interpolator = self._update_obs_and_interpolator(
            da_gauges, 
            da_cmls
        )

        # If few observations return zero grid
        if (~np.isnan(obs)).sum() <= self.min_observations:
            return xr.DataArray(
                data=[np.zeros(da_grid.x_grid.shape)],
                coords=da_grid.coords,
                dims=da_grid.dims,
            )

        # Coordinates to predict
        coord_pred = np.hstack(
            [da_grid.y_grid.data.reshape(-1, 1), da_grid.x_grid.data.reshape(-1, 1)]
        )
        
        # IDW interpolator invdisttree
        interpolated = self._interpolator(
            q=coord_pred,
            z=obs,
            nnear=obs.size if obs.size <= self.nnear else self.nnear,
            p=self.p,
            idw_method=self.idw_method,
            max_distance=self.max_distance,
        ).reshape(da_grid.x_grid.shape)

        da_interpolated = xr.DataArray(
            data=[interpolated], coords=da_grid.coords, dims=da_grid.dims
        )
        
        return da_interpolated
    
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
        variogram_model="spherical",
        variogram_parameters=None,
        discretization=8,
        min_observations=1,
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
        Interpolator.__init__(self)

        self.discretization = discretization
        self.min_observations = min_observations
        self.nnear = nnear
        self.max_distance = max_distance
        self.full_line=full_line

        # Construct variogram using parameters provided by user
        if variogram_parameters is None:
            variogram_parameters = {"sill": 0.9, "range": 5000, "nugget": 0.1}
        obs = np.array([0, 1]) # dummy variables
        coord = np.array([[0, 1], [1, 0]])

        self.variogram = bk_functions.construct_variogram(
            obs, coord, variogram_parameters, variogram_model
        )
        
    def _init_interpolator(self, da_gauges=None, da_cmls=None):
        return bk_functions.OBKrigTree(
            self.variogram,
            ds_cmls=da_cmls, 
            ds_gauges=da_gauges, 
            discretization=self.discretization,
            nnear=self.nnear,
            max_distance=self.max_distance,
            full_line=self.full_line,
        )

    def __call__(self, da_grid, da_cmls=None, da_gauges=None):

        """Interpolate observations for one time step.

        Interpolates ground observations for one time step. 

        Input data can have a time dimension of length 1 or no time dimension.

        Parameters
        ----------
        da_grid: xarray.DataArray
            Dataframe providing the grid for interpolation. Must contain
            projected x_grid and y_grid coordinates.
        da_cmls: xarray.DataArray
            CML observations. Must contain the projected coordinates (site_0_x, 
            site_1_x, site_0_y, site_1_y).
        da_gauges: xarray.DataArray
            Gauge observations. Must contain the projected
            coordinates (x, y).

        Returns
        -------
        da_field_out: xarray.DataArray
            DataArray with the same structure as the ds_rad but with the
            interpolated field.
        """
        
        obs, self._interpolator = self._update_obs_and_interpolator(
            da_gauges, 
            da_cmls
        )
        
        # If few observations return zero grid
        if (~np.isnan(obs)).sum() <= self.min_observations:
            return xr.DataArray(
                data=[np.zeros(da_grid.x_grid.shape)],
                coords=da_grid.coords,
                dims=da_grid.dims,
            )

        # Points to interpolate
        points = np.hstack([
            da_grid.y_grid.data.reshape(-1, 1),
            da_grid.x_grid.data.reshape(-1, 1),
        ])
        
        # Neighbourhood kriging
        interpolated = self._interpolator(
            points,
            obs
        ).reshape(da_grid.x_grid.shape)

        da_interpolated = xr.DataArray(
            data=[interpolated], coords=da_grid.coords, dims=da_grid.dims
        )

        return da_interpolated
