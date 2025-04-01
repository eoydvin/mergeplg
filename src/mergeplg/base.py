"""Module for merging CML and rain gauge data with gridded data."""

from __future__ import annotations

import numpy as np
import poligrain as plg
import xarray as xr

from .radolan import idw

from mergeplg import bk_functions

class Base:
    """Update weights and geometry and evaluate rainfall grid at CMLs and rain gauges

    Parent class for the merging and interpolation methods. Works by keeping a copy of
    the weights used to obtain values of gridded data at the CML and rain gauge 
    positions (self.intersect_weights, self.get_grid_at_points).
    """

    def __init__(
        self,
        grid_point_location="center",
    ):
        """Construct base class

        Parameters
        ----------
        self.grid_point_location str
            Grid cell reference position. For instance 'center'.
        self.intersect_weights xarray.Dataset
            Weights for getting radar observations along CMLs.
        self.gauge_ids numpy.array
            Name of rain gauges, used to check if gauge weights needs to be updated.
        self.get_grid_at_points function
            Returns the grids value at the rain gauge positions.
        self.x0_cml xarray.DataArray
            Midpoint or discretized coordinates along the CMLs, depending on
            if update_ or update_block_ was used to update geometry
        self.x0_gauge xarray.DataArray
            Rain gauge coordinates.
        """
        # Location of grid point, used in intersect weights
        self.grid_point_location = grid_point_location

        # CML weights and gauge weights
        self.intersect_weights = None
        self.get_grid_at_points = None

        # Names of gauges and cmls, used for checking changes
        self.gauge_ids = None
        self.cml_ids = None

        # Init interpolator
        self.interpolator = None

    def update_weights_(self, da_grid, da_cml=None, da_gauge=None):
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
                        grid_point_location=self.grid_point_location,
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
                            grid_point_location=self.grid_point_location,
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
    
    def update_interpolator_idw_(self, da_cml=None, da_gauge=None): 
        if (da_cml is not None) and (da_gauge is not None):
            cml_id_new = da_cml.cml_id.data
            cml_change = not np.array_equal(cml_id_new, self.cml_ids)
            gauge_id_new = da_gauge.id.data
            gauge_change = not np.array_equal(gauge_id_new, self.gauge_ids)
            
            if gauge_change or cml_change:
                y = np.concatenate(da_cml.y.data, da_gauge.y.data)
                x = np.concatenate(da_cml.x.data, da_gauge.x.data)
                self.cml_ids = cml_id_new
                self.x = x
                self.y = y
                self.gauge_ids = gauge_id_new
                yx = np.hstack([y.reshape(-1, 1), x.reshape(-1, 1)])
                self.interpolator = idw.Invdisttree(yx)
            
            # Get observations
            obs = np.concatenate([
                da_cml.data.flatten(),
                da_gauge.data.flatten(),
            ])

        elif da_cml is not None:
            cml_id_new = da_cml.cml_id.data
            cml_change = not np.array_equal(cml_id_new, self.cml_ids)
            gauge_change = not np.array_equal(None, self.gauge_ids)
            
            if gauge_change or cml_change:
                y = da_cml.y.data
                x = da_cml.x.data
                self.x = x
                self.y = y
                self.cml_ids = cml_id_new
                self.gauge_ids = None
                yx = np.hstack([y.reshape(-1, 1), x.reshape(-1, 1)])
                self.interpolator = idw.Invdisttree(yx)

            # Get and return observations
            obs = da_cml.data.flatten()
        
        elif da_gauge is not None:
            cml_change = not np.array_equal(None, self.cml_ids)
            gauge_id_new = da_gauge.id.data
            gauge_change = not np.array_equal(gauge_id_new, self.gauge_ids)
            
            if gauge_change or cml_change:
                y = da_gauge.y.data
                x = da_gauge.x.data
                self.cml_ids = None
                self.x = x
                self.y = y
                self.gauge_ids = gauge_id_new
                yx = np.hstack([y.reshape(-1, 1), x.reshape(-1, 1)])
                self.interpolator = idw.Invdisttree(yx)

            # Get and return observations
            obs = da_gauge.data.flatten()
        else:
            msg = 'Provide rain gauge or CML data'
            raise ValueError(msg)
        
        return obs

    def update_interpolator_obk_(self, da_cml=None, da_gauge=None):
        if (da_cml is not None) and (da_gauge is not None):
            cml_id_new = da_cml.cml_id.data
            gauge_id_new = da_gauge.id.data
            cml_change = not np.array_equal(cml_id_new, self.cml_ids)
            gauge_change = not np.array_equal(gauge_id_new, self.gauge_ids)
            
            if gauge_change or cml_change:
                self.cml_ids = cml_id_new
                self.gauge_ids = gauge_id_new
                self.interpolator = bk_functions.OBKrigTree(
                    self.variogram,
                    ds_cmls=da_cml, 
                    ds_gauges=da_cml, 
                    discretization=self.discretization,
                    nnear=self.nnear,
                    max_distance=self.max_distance,
                    full_line=self.full_line,
                )
            
            # Get and return observations
            return np.concatenate([
                da_cml.data.flatten(),
                da_gauge.data.flatten(),
            ])

        elif da_cml is not None:
            cml_id_new = da_cml.cml_id.data
            cml_change = not np.array_equal(cml_id_new, self.cml_ids)
            gauge_change = not np.array_equal(None, self.gauge_ids)
            
            if gauge_change or cml_change:
                self.cml_ids = cml_id_new
                self.interpolator = bk_functions.OBKrigTree(
                    self.variogram,
                    ds_cmls=da_cml, 
                    discretization=self.discretization,
                    nnear=self.nnear,
                    max_distance=self.max_distance,
                )
            
            # Get and return observations
            return da_cml.data.flatten()
        
        elif da_gauge is not None:
            cml_change = not np.array_equal(None, self.cml_ids)
            gauge_id_new = da_gauge.id.data
            gauge_change = not np.array_equal(gauge_id_new, self.gauge_ids)
            
            if gauge_change or cml_change:
                self.cml_ids = None
                self.gauge_ids = gauge_id_new
                self.interpolator = bk_functions.OBKrigTree(
                    self.variogram,
                    ds_gauges=da_cml, 
                    discretization=self.discretization,
                    nnear=self.nnear,
                    max_distance=self.max_distance,
                )
            
            # Get and return observations
            return da_gauge.data.flatten()
    
    def update_interpolator_ked_(self, da_cml=None, da_gauge=None):
        if (da_cml is not None) and (da_gauge is not None):
            cml_id_new = da_cml.cml_id.data
            gauge_id_new = da_gauge.id.data
            cml_change = not np.array_equal(cml_id_new, self.cml_ids)
            gauge_change = not np.array_equal(gauge_id_new, self.gauge_ids)
            
            if gauge_change or cml_change:
                self.cml_ids = cml_id_new
                self.gauge_ids = gauge_id_new
                self.interpolator = bk_functions.BKEDTree(
                    self.variogram,
                    ds_cmls=da_cml, 
                    ds_gauges=da_cml, 
                    discretization=self.discretization,
                    nnear=self.nnear,
                    max_distance=self.max_distance,
                    full_line=self.full_line,
                )
            
            # Get and return observations
            return np.concatenate([
                da_cml.data.flatten(),
                da_gauge.data.flatten(),
            ])

        elif da_cml is not None:
            cml_id_new = da_cml.cml_id.data
            cml_change = not np.array_equal(cml_id_new, self.cml_ids)
            gauge_change = not np.array_equal(None, self.gauge_ids)
            if gauge_change or cml_change:
                self.cml_ids = cml_id_new
                self.interpolator = bk_functions.BKEDTree(
                    self.variogram,
                    ds_cmls=da_cml, 
                    discretization=self.discretization,
                    nnear=self.nnear,
                    max_distance=self.max_distance,
                )
            
            # Get and return observations
            return da_cml.data.flatten()
        
        elif da_gauge is not None:
            cml_change = not np.array_equal(None, self.cml_ids)
            gauge_id_new = da_gauge.id.data
            gauge_change = not np.array_equal(gauge_id_new, self.gauge_ids)
            
            if gauge_change or cml_change:
                self.cml_ids = None
                self.gauge_ids = gauge_id_new
                self.interpolator = bk_functions.BKEDTree(
                    self.variogram,
                    ds_gauges=da_cml, 
                    discretization=self.discretization,
                    nnear=self.nnear,
                    max_distance=self.max_distance,
                )
            
            # Get and return observations
            return da_gauge.data.flatten()

    def get_grid_obs_(self, da_grid, da_cml=None, da_gauge=None):
        """Calculate grid and obs for current state

        Returns and ordered list of the gridded data at the position of the
        ground observations and ground observations (CML and rain gauges)

        Parameters
        ----------
        da_grid: xarray.DataArray
            Gridded rainfall data. Must contain the projected x_grid and Y_grid
            coordinates.
        da_cml: xarray.DataArray
            CML observations. Must contain the projected coordinates for the CML
            positions (site_0_x, site_0_y, site_1_x, site_1_y).
        da_gauge: xarray.DataArray
            Gauge observations. Must contain the projected coordinates for the rain
            gauge positions (x, y)
        """
        # If CML and gauge data is provided
        if (da_cml is not None) and (da_gauge is not None):
            # Calculate grid along CMLs using intersect weights
            grid_cml = (
                plg.spatial.get_grid_time_series_at_intersections(
                    grid_data=da_grid,
                    intersect_weights=self.intersect_weights,
                )
            ).data.ravel()

            # Estimate grid at gauges
            grid_gauge = self.get_grid_at_points(
                da_gridded_data=da_grid,
                da_point_data=da_gauge,
            ).data.ravel()

            # Stack grid observations at cml and gauge in correct order
            grid_at_obs = np.concatenate([grid_cml, grid_gauge])

            # Stack instrument observations at cml and gauge in correct order
            observations_ground = np.concatenate(
                [da_cml.data.ravel(), da_gauge.data.ravel()]
            )

        # If only CML data is provided
        elif da_cml is not None:
            # Estimate grid at cml
            grid_at_obs = (
                plg.spatial.get_grid_time_series_at_intersections(
                    grid_data=da_grid,
                    intersect_weights=self.intersect_weights,
                )
            ).data.ravel()

            # Get CML data
            observations_ground = da_cml.data.ravel()

        # If only rain gauge data is provided
        else:
            # Estimate grid at gauges
            grid_at_obs = self.get_grid_at_points(
                da_gridded_data=da_grid,
                da_point_data=da_gauge,
            ).data.ravel()

            # Get gauge data
            observations_ground = da_gauge.data.ravel()

        # Return grid_at_observations, observations
        return grid_at_obs, observations_ground

    def get_obs_(self, da_cml=None, da_gauge=None):
        """Calculate ground observations given data

        Returns and ordered list of ground observations

        Parameters
        ----------
        da_cml: xarray.DataArray
            CML observations. 
        da_gauge: xarray.DataArray
            gauge observations. 
        """
        if (da_cml is not None) and (da_gauge is not None):
            observations_ground = np.concatenate(
                [da_cml.data.ravel(), da_gauge.data.ravel()]
            )

        elif da_cml is not None:
            observations_ground = da_cml.data.ravel()

        else:
            observations_ground = da_gauge.data.ravel()

        return observations_ground
