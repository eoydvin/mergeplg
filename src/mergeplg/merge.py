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


def calculate_cml_geometry(ds_cmls, discretization=8):
    """Calculate the position of points along CMLs.

    Calculates the discretized CML geometry by dividing the CMLs into
    discretization-number of intervals. The ds_cmls xarray object must contain the
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
discretization        Number of intervals to discretize lines into.

    Returns
    -------
    x0: np.array
        Array with coordinates for all CMLs. The array is organized into a 3D
        matrix with the following structure:
            (number of n CMLs [0, ..., n],
             y/x-cooridnate [0(y), 1(x)],
             interval [0, ..., discretization])

    """
    # Calculate discretized positions along the lines, store in numy array
    xpos = np.zeros([ds_cmls.cml_id.size, discretization + 1])  # shape (line, position)
    ypos = np.zeros([ds_cmls.cml_id.size, discretization + 1])

    # For all CMLs
    for block_i, cml_id in enumerate(ds_cmls.cml_id):
        x_a = ds_cmls.sel(cml_id=cml_id).site_0_x.data
        y_a = ds_cmls.sel(cml_id=cml_id).site_0_y.data
        x_b = ds_cmls.sel(cml_id=cml_id).site_1_x.data
        y_b = ds_cmls.sel(cml_id=cml_id).site_1_y.data

        # for all dicretization steps in link estimate its place on the grid
        for i in range(discretization + 1):
            xpos[block_i, i] = x_a + (i / discretization) * (x_b - x_a)
            ypos[block_i, i] = y_a + (i / discretization) * (y_b - y_a)

    # Store x and y coordinates in the same array (n_cmls, y/x, discretization)
    return np.array([ypos, xpos]).transpose([1, 0, 2])


class Merge:
    """Common code for all merging methods.

    init just initializes to parameters
    
    method update: updates the geometry of the 
    
    

    Returns
    -------
    Nothing
    """

    def __init__(
        self,
        grid_point_location="center",
        min_obs=5,
    ):
        # Number of observations required to do radar adjustment
        self.min_obs_ = min_obs
        
        # Location of grid point for weather radar, used in intersect weights
        self.grid_point_location = grid_point_location
        
        # Init weights CML 
        self.intersect_weights = None        
        
        # Init gauge possitions and names
        self.get_grid_at_points = None
        self.gauge_id = None
        
        
    def update_(self, da_rad, da_cml = None, da_gauge = None):
        # 
        
        # Check that there is radar or gauge data, if not raise an error
        if (da_cml is None) and (da_gauge is None):
            raise ValueError('Please provide cml or gauge data')
            
        if da_cml is not None:
            # If intersect weights not computed, compute all weights
            if self.intersect_weights is None:
                self.intersect_weights = (
                    plg.spatial.calc_sparse_intersect_weights_for_several_cmls(
                        x1_line=da_cml.site_0_lon.data,
                        y1_line=da_cml.site_0_lat.data,
                        x2_line=da_cml.site_1_lon.data,
                        y2_line=da_cml.site_1_lat.data,
                        cml_id=da_cml.cml_id.data,
                        x_grid=da_rad.lon.data,
                        y_grid=da_rad.lat.data,
                        grid_point_location=self.grid_point_location,
                    )
                )
                
            else: 
                # New cml names
                cml_id_new = np.sort(da_cml.cml_id.data)
                
                # cml names of previous update
                cml_id_old = np.sort(self.intersect_weights.cml_id.data)
        
                # If new CML coord is present or old CML coord is removed
                if not np.array_equal(cml_id_new, cml_id_old):
                    # Identify cml_id that is in the new and old array
                    cml_id_keep = np.intersect1d(cml_id_new, cml_id_old)
                    
                    # Strip the stored intersect weights, keeping only new ones
                    self.intersect_weights = self.intersect_weights.sel(
                        cml_id = cml_id_keep
                        )
                    
                    # Identify cml_id not in the new
                    missing_in_old = np.setdiff1d(cml_id_new, cml_id_old)
                    
                    # Slice da_cml to get only missing coords
                    da_cml_add = da_cml.sel(cml_id = missing_in_old)
                    
                    # Interect weights of CMLs to add
                    intersect_weights_add = (                
                        plg.spatial.calc_sparse_intersect_weights_for_several_cmls(
                            x1_line=da_cml_add.site_0_lon.data,
                            y1_line=da_cml_add.site_0_lat.data,
                            x2_line=da_cml_add.site_1_lon.data,
                            y2_line=da_cml_add.site_1_lat.data,
                            cml_id=da_cml_add.cml_id.data,
                            x_grid=da_rad.lon.data,
                            y_grid=da_rad.lat.data,
                            grid_point_location=self.grid_point_location,
                        )
                    )
                    
                    # Add new intersect weights
                    self.intersect_weights = xr.concat(
                        [self.intersect_weights, intersect_weights_add], 
                        dim='cml_id')

        # If gauge data is present
        if da_gauge is not None:
            # If this is the first update
            if self.get_grid_at_points is None:
                # Calculate gridpoints for gauges
                self.get_grid_at_points = plg.spatial.GridAtPoints(
                    da_gridded_data=da_rad,
                    da_point_data=da_gauge,
                    nnear=1,
                    stat="best",
                )
            
            else:
                
                # Get names of new gauges
                gauge_id_new = da_gauge.id.data
                
                # Get names of gauges in previous update
                gauge_id_old = self.gauge_id
                
                # Check that equal, element order is important
                if not np.array_equal(gauge_id_new, gauge_id_old):
                    # Calculate new gauge possitions
                    self.get_grid_at_points = plg.spatial.GridAtPoints(
                        da_gridded_data=da_rad,
                        da_point_data=da_gauge,
                        nnear=1,
                        stat="best",
                    )
                    
                    # Store new gauge names
                    self.gauge_id = da_gauge.id.data
                    
    def radar_at_ground_(self, da_rad, da_cml=None, da_gauge=None):
        """Evaluate radar at cml and rain gauge ground possitions

        Evaluates weather radar along cml and at rain gauge possitions. 

        Parameters
        ----------
        da_rad : xarray.DataArray
            DataArray with radar data. 

        Returns
        -------
        da_rad_out: xarray.DataArray
            DataArray with the same structure as the ds_rad but with the CML
            adjusted radar field.
        """

        # When CML and gauge
        if (da_cml is not None) and (da_gauge is not None):
            # Check that we have selected only one timestep
            assert da_rad.time.size == 1, "Select only one time step"
            assert da_cml.time.size == 1, "Select only one time step"
            assert da_gauge.time.size == 1, "Select only one time step"
            
            rad_cml = (
                plg.spatial.get_grid_time_series_at_intersections(
                    grid_data=da_rad,
                    intersect_weights=self.intersect_weights,
                )
            ).data.ravel()
            
            # Estimate radar at gauges
            rad_gauge = self.get_grid_at_points(
                da_gridded_data=da_rad,
                da_point_data=da_gauge,  
            ).data.ravel()
            
            # Stack radar observations at cml and gauge in correct order
            observations_radar = np.concatenate([rad_cml, rad_gauge])
            
            # Stack instrument observations at cml and gauge in correct order
            observations_ground = np.concatenate(
                [da_cml.data.ravel(), da_gauge.data.ravel()]
            )
                        
        # When only CML
        elif (da_cml is not None):
            # Check that we have selected only one timestep
            assert da_rad.time.size == 1, "Select only one time step"
            assert da_cml.time.size == 1, "Select only one time step"
            
            # Estimate radar at cml
            observations_radar = (
                plg.spatial.get_grid_time_series_at_intersections(
                    grid_data=da_rad,
                    intersect_weights=self.intersect_weights,
                )
            ).data.ravel()
            
            # Store cml data
            observations_ground = da_cml.data.ravel()
        
        # When only gauge
        elif (da_gauge is not None):      
            # Check that we have selected only one timestep
            assert da_rad.time.size == 1, "Select only one time step"
            assert da_gauge.time.size == 1, "Select only one time step"
            
            # Estimate radar at gauges
            observations_radar = self.get_grid_at_points(
                da_gridded_data=da_rad,
                da_point_data=da_gauge,  
            ).data.ravel()
            
            # Store gauge data
            observations_ground = da_gauge.data.ravel()
            
        # Return radar at ground observations and corresponding 
        return observations_radar, observations_ground


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
        grid_location_radar="center",
        min_obs=5,
    ):
        Merge.__init__(self, grid_location_radar, min_obs)

    def update(self, da_rad, da_cml=None, da_gauge=None):
        # Update cml and gauge weights used for getting radar data
        self.update_(da_rad, da_cml=da_cml, da_gauge=da_gauge)
        
        if da_cml is not None:
            # Calculate CML midpoints
            x = ((da_cml.site_0_x + da_cml.site_1_x) / 2).data
            y = ((da_cml.site_0_y + da_cml.site_1_y) / 2).data
            
            # Store cml coordinates as columns
            self.x0_cml = np.hstack([y.reshape(-1, 1), x.reshape(-1, 1)])
        
        if da_gauge is not None:
            # Store gauge coordinates as columns
            self.x0_gauge = np.hstack([
                    da_gauge.y.data.reshape(-1, 1),
                    da_gauge.x.data.reshape(-1, 1),
                ]
            )
    
    def adjust(self, da_rad, da_cml=None, da_gauge=None):
        """Adjust radar field for one time step.

        Evaluates if we have enough observations to adjust the radar field.
        Then adjust radar field to observations using additive IDW.

        Parameters
        ----------
        da_rad : xarray.DataArray
            DataArray with radar data. 

        Returns
        -------
        da_rad_out: xarray.DataArray
            DataArray with the same structure as the ds_rad but with the CML
            adjusted radar field.
        """

        # When CML and gauge
        if (da_cml is not None) and (da_gauge is not None):
            # Evaluate radar at cml and gauge ground positions
            rad, obs = self.radar_at_ground_(
                da_rad, 
                da_cml=da_cml, 
                da_gauge=da_gauge
            )
            
            # Stack coordinates for cml and gauge
            x0 = np.vstack([self.x0_cml, self.x0_gauge])
        
        # When only CML
        elif (da_cml is not None):
            # Evaluate radar at cml ground positions
            rad, obs = self.radar_at_ground_(
                da_rad, 
                da_cml=da_cml, 
            )
            
            # Store cml coordinates
            x0 = self.x0_cml
        
        # When only gauge
        elif (da_gauge is not None):      
            # Evaluate radar at gauge ground positions
            rad, obs = self.radar_at_ground_(
                da_rad, 
                da_cml=da_cml, 
                da_gauge=da_gauge
            )
            
            # Store gauge coordinates
            x0 = self.x0_gauge
            
        else:
            raise ValueError('Please provide CML or gauge data')
        
        # Calculate radar-instrument difference if radar has observation
        diff = np.where(rad > 0, obs - rad, np.nan)

        # Get index of not-nan obs
        keep = np.where(~np.isnan(diff))[0]

        # Check that that there is enough observations
        if keep.size > self.min_obs_:
            # get timestap
            time = da_rad.time.data[0]
            
            # Remove radar time dimension
            da_rad_t = da_rad.sel(time=time).copy() 

            # do addtitive IDW merging
            adjusted = merge_additive_idw(
                xr.where(da_rad_t > 0, da_rad_t, np.nan), # func skipps nan
                diff[keep],
                x0[keep, :],
            )
            
            # Replace nan with original radar data (so that da_rad nan is kept)
            adjusted = xr.where(np.isnan(adjusted), da_rad_t, adjusted)
            
            # Re-assign timestamp
            adjusted = adjusted.assign_coords(time=time)
            
            return adjusted

        # Else return the unadjusted radar
        return da_rad
    
    
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
        grid_location_radar="center",
        min_obs=5,
        discretization = 8,
    ):
        Merge.__init__(self, grid_location_radar, min_obs)
        
        # Number of discretization points along CML
        self.discretization = discretization

    def update(self, da_rad, da_cml=None, da_gauge=None):
        # Update cml and gauge weights used for getting radar data
        self.update_(da_rad, da_cml=da_cml, da_gauge=da_gauge)
        
        if da_cml is not None:
            # Store cml coordinates as columns
            self.x0_cml = calculate_cml_geometry(
                self.da_cml, 
                discretization=self.discretization
            )
            
        if da_gauge is not None:
            # Make da_gauge imitate CML coordinates, treating the point as a line
            da_gauge = 
            
            self.x0_gauge = np.hstack([
                    da_gauge.y.data.reshape(-1, 1),
                    da_gauge.x.data.reshape(-1, 1),
                ]
            )
    
    def adjust(self, da_rad, da_cml=None, da_gauge=None):
        """Adjust radar field for one time step.

        Evaluates if we have enough observations to adjust the radar field.
        Then adjust radar field to observations using additive IDW.

        Parameters
        ----------
        da_rad : xarray.DataArray
            DataArray with radar data. 

        Returns
        -------
        da_rad_out: xarray.DataArray
            DataArray with the same structure as the ds_rad but with the CML
            adjusted radar field.
        """

        # When CML and gauge
        if (da_cml is not None) and (da_gauge is not None):
            # Evaluate radar at cml and gauge ground positions
            rad, obs = self.radar_at_ground_(
                da_rad, 
                da_cml=da_cml, 
                da_gauge=da_gauge
            )
            
            # Stack coordinates for cml and gauge
            x0 = np.vstack([self.x0_cml, self.x0_gauge])
        
        # When only CML
        elif (da_cml is not None):
            # Evaluate radar at cml ground positions
            rad, obs = self.radar_at_ground_(
                da_rad, 
                da_cml=da_cml, 
            )
            
            # Store cml coordinates
            x0 = self.x0_cml
        
        # When only gauge
        elif (da_gauge is not None):      
            # Evaluate radar at gauge ground positions
            rad, obs = self.radar_at_ground_(
                da_rad, 
                da_cml=da_cml, 
                da_gauge=da_gauge
            )
            
            # Store gauge coordinates
            x0 = self.x0_gauge
        
        else:
            raise ValueError('Please provide CML or gauge data')
        
        # Calculate radar-instrument difference if radar has observation
        diff = np.where(rad > 0, obs - rad, np.nan)

        # Get index of not-nan obs
        keep = np.where(~np.isnan(diff))[0]

        # Check that that there is enough observations
        if keep.size > self.min_obs_:
            # get timestap
            time = da_rad.time.data[0]
            
            # Remove radar time dimension
            da_rad_t = da_rad.sel(time=time).copy() 

            # do addtitive IDW merging
            adjusted = merge_additive_idw(
                xr.where(da_rad_t > 0, da_rad_t, np.nan), # func skipps nan
                diff[keep],
                x0[keep, :],
            )
            
            # Replace nan with original radar data (so that da_rad nan is kept)
            adjusted = xr.where(np.isnan(adjusted), da_rad_t, adjusted)
            
            # Re-assign timestamp
            adjusted = adjusted.assign_coords(time=time)
            
            return adjusted

        # Else return the unadjusted radar
        return da_rad


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
