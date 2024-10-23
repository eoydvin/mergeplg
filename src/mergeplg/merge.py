"""Module for merging CML and radar data."""

from __future__ import annotations

import numpy as np
import poligrain as plg
import xarray as xr

from mergeplg import merge_functions

# from .merge_functions import (
#     calculate_cml_geometry,
#     calculate_cml_midpoint,
#     calculate_gauge_midpoint,
#     estimate_transformation,
#     estimate_variogram,
#     merge_additive_blockkriging,
#     merge_additive_idw,
#     merge_ked_blockkriging,
# )


class Merge:
    """Common code for all merging methods

    Performs basic checks.

    Parameters
    ----------
    grid_location_radar: str
        String indicating the grid location of the radar.
    min_obs: int
        Minimum number of observations needed in order to do adjustment.
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

        # Init weights CML and names
        self.intersect_weights = None

        # Init gauge positions and names
        self.get_grid_at_points = None

        # Init coordinates for gauge and CML
        self.x0_gauge = None
        self.x0_cml = None

    def update_(self, da_rad, da_cml=None, da_gauge=None):
        """Update weights and x0 geometry for CML and gauge

        This function uses the midpoint of the CML as CML reference.

        Parameters
        ----------
        da_rad: xarray.DataArray
            Gridded radar data. Must contain the x and y meshgrid
        da_cml: xarray.DataArray
            CML observations. Must contain the transformed coordinates site_0_x,
            site_1_x, site_0_y and site_1_y. And lon/lat..
        da_gauge: xarray.DataArray
            gauge observations. Must contain the transformed coordinates y, x.
        """
        # Check that there is radar or gauge data, if not raise an error
        if (da_cml is None) and (da_gauge is None):
            msg = "Please provide cml or gauge data"
            raise ValueError(msg)

        # If CML is present
        if da_cml is not None:
            # If intersect weights not computed, compute all weights
            if self.x0_cml is None:
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

                # Calculate CML midpoints
                self.x0_cml = merge_functions.calculate_cml_midpoint(da_cml)

            else:
                # New cml names
                cml_id_new = np.sort(da_cml.cml_id.data)

                # cml names of previous update
                cml_id_old = np.sort(self.intersect_weights.cml_id.data)

                # If new CML coord is present or old CML coord is removed
                if not np.array_equal(cml_id_new, cml_id_old):
                    # Identify cml_id that is in the new and old array
                    cml_id_keep = np.intersect1d(cml_id_new, cml_id_old)

                    # Slice the stored intersect weights, keeping only new ones
                    self.intersect_weights = self.intersect_weights.sel(
                        cml_id=cml_id_keep
                    )

                    # Slice stored CML midpoints, keeping only new ones
                    self.x0_cml = self.x0_cml.sel(cml_id=cml_id_keep)

                    # Identify cml_id not in the new
                    cml_id_not_in_old = np.setdiff1d(cml_id_new, cml_id_old)

                    # Slice da_cml to get only missing coords
                    da_cml_add = da_cml.sel(cml_id=cml_id_not_in_old)

                    # Intersect weights of CMLs to add
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
                        [self.intersect_weights, intersect_weights_add], dim="cml_id"
                    )

                    # Calculate CML midpoint for new CMLs
                    x0_cml_add = merge_functions.calculate_cml_midpoint(da_cml_add)

                    # Add to existing x0
                    self.x0_cml = xr.concat([self.x0_cml, x0_cml_add], dim="cml_id")

            # Sort x0_cml so it follows the same order as da_cml
            self.x0_cml = self.x0_cml.sel(cml_id=da_cml.cml_id.data)

        # If gauge data is present
        if da_gauge is not None:
            # If this is the first update
            if self.x0_gauge is None:
                # Calculate gridpoints for gauges
                self.get_grid_at_points = plg.spatial.GridAtPoints(
                    da_gridded_data=da_rad,
                    da_point_data=da_gauge,
                    nnear=1,
                    stat="best",
                )

                # Calculate gauge coordinates
                self.x0_gauge = merge_functions.calculate_gauge_midpoint(da_gauge)

            else:
                # Get names of new gauges
                gauge_id_new = da_gauge.id.data

                # Get names of gauges in previous update
                gauge_id_old = self.x0_gauge.id.data

                # Check that equal, element order is important
                if not np.array_equal(gauge_id_new, gauge_id_old):
                    # Calculate new gauge positions
                    self.get_grid_at_points = plg.spatial.GridAtPoints(
                        da_gridded_data=da_rad,
                        da_point_data=da_gauge,
                        nnear=1,
                        stat="best",
                    )

                # Calculate gauge coordinates
                self.x0_gauge = merge_functions.calculate_gauge_midpoint(da_gauge)

            # Sort x0_gauge so it follows the same order as da_gauge
            self.x0_gauge = self.x0_gauge.sel(id=da_gauge.id.data)

    def update_block_(self, da_rad, discretization, da_cml=None, da_gauge=None):
        """Update weights and x0 geometry for CML and gauge assuming block data

        This function uses the full CML geometry and makes the gauge geometry
        look like a rain gauge.

        Parameters
        ----------
        da_rad: xarray.DataArray
            Gridded radar data. Must contain the x and y meshgrid
        da_cml: xarray.DataArray
            CML observations.
        da_gauge: xarray.DataArray
            gauge observations. Must contain the transformed coordinates y, x.
        """
        # Check that there is radar or gauge data, if not raise an error
        if (da_cml is None) and (da_gauge is None):
            msg = "Please provide cml or gauge data"
            raise ValueError(msg)

        # If CML is present
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

                # CML coordinates along all links
                self.x0_cml = merge_functions.calculate_cml_geometry(
                    da_cml, discretization=discretization
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

                    # Slice the stored intersect weights, keeping only new ones
                    self.intersect_weights = self.intersect_weights.sel(
                        cml_id=cml_id_keep
                    )

                    # Slice stored CML midpoints, keeping only new ones
                    self.x0_cml = self.x0_cml.sel(cml_id=cml_id_keep)

                    # Identify cml_id not in the new
                    cml_id_not_in_old = np.setdiff1d(cml_id_new, cml_id_old)

                    # Slice da_cml to get only missing coords
                    da_cml_add = da_cml.sel(cml_id=cml_id_not_in_old)

                    # Intersect, weights of CMLs to add
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
                        [self.intersect_weights, intersect_weights_add], dim="cml_id"
                    )

                    # Calculate CML geometry for new links
                    x0_cml_add = merge_functions.calculate_cml_geometry(
                        da_cml_add, discretization=discretization
                    )

                    # Add new x0 to self.x0_cml
                    self.x0_cml = xr.concat([self.x0_cml, x0_cml_add], dim="cml_id")

            # Sort x0_cml so it follows the same order as da_cml
            self.x0_cml = self.x0_cml.sel(cml_id=da_cml.cml_id.data)

        # If gauge data is present
        if da_gauge is not None:
            # If this is the first update
            if self.x0_gauge is None:
                # Calculate gridpoints for gauges
                self.get_grid_at_points = plg.spatial.GridAtPoints(
                    da_gridded_data=da_rad,
                    da_point_data=da_gauge,
                    nnear=1,
                    stat="best",
                )

                # Calculate gauge coordinates
                x0_gauge = merge_functions.calculate_gauge_midpoint(da_gauge)

                # Repeat the same coordinates so that the array gets the same
                # shape as x0_cml, used for block kriging
                self.x0_gauge = x0_gauge.expand_dims(
                    disc=range(discretization + 1)
                ).transpose("id", "yx", "disc")

            else:
                # Get names of new gauges
                gauge_id_new = da_gauge.id.data

                # Get names of gauges in previous update
                gauge_id_old = self.x0_gauge.id.data

                # Check that equal, element order is important
                if not np.array_equal(gauge_id_new, gauge_id_old):
                    # Calculate new gauge positions
                    self.get_grid_at_points = plg.spatial.GridAtPoints(
                        da_gridded_data=da_rad,
                        da_point_data=da_gauge,
                        nnear=1,
                        stat="best",
                    )

                    # Calculate gauge coordinates
                    x0_gauge = merge_functions.calculate_gauge_midpoint(da_gauge)

                    # As the gauge is just a point, repeat the gauge coord
                    self.x0_gauge = x0_gauge.expand_dims(
                        disc=range(discretization + 1)
                    ).transpose("id", "yx", "disc")

            # Sort x0_gauge so it follows the same order as da_gauge
            self.x0_gauge = self.x0_gauge.sel(id=da_gauge.id.data)

    def radar_at_ground_(self, da_rad, da_cml=None, da_gauge=None):
        """Evaluate radar at cml and rain gauge ground positions

        Evaluates weather radar along cml and at rain gauge positions. Assumes
        That gauge and CML weights are updated.

        Parameters
        ----------
        da_rad: xarray.DataArray
            Gridded radar data. Must contain the xs and ys meshgrid.
        da_cml: xarray.DataArray
            CML observations.
        da_gauge: xarray.DataArray
            gauge observations. Must contain the transformed coordinates y, x.
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

            # get x0
            x0 = np.vstack([self.x0_cml.data, self.x0_gauge.data])

        # When only CML
        elif da_cml is not None:
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

            # get x0
            x0 = self.x0_cml.data

        # When only gauge
        else:
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

            # get x0
            x0 = self.x0_gauge.data

        # Return radar at ground observations and corresponding x0
        return observations_radar, observations_ground, x0


class MergeAdditiveIDW(Merge):
    """Merge CML and radar using an additive IDW (CML midpoint).

    Marges the provided radar field in ds_rad to CML observations by
    interpolating the difference between the CML and radar observations using
    IDW.
    """

    def __init__(
        self,
        grid_location_radar="center",
        min_obs=5,
    ):
        Merge.__init__(self, grid_location_radar, min_obs)

    def update(self, da_rad, da_cml=None, da_gauge=None):
        """Update weights and x0 geometry for CML and gauge

        This function uses the midpoint of the CML as CML reference.

        Parameters
        ----------
        da_rad: xarray.DataArray
            Gridded radar data.
        da_cml: xarray.DataArray
            CML observations.
        da_gauge: xarray.DataArray
            gauge observations.
        """
        self.update_(da_rad, da_cml=da_cml, da_gauge=da_gauge)

    def adjust(self, da_rad, da_cml=None, da_gauge=None):
        """Adjust radar field for one time step.

        Evaluates if we have enough observations to adjust the radar field.

        Parameters
        ----------
        da_rad: xarray.DataArray
            Gridded radar data.
        da_cml: xarray.DataArray
            CML observations.
        da_gauge: xarray.DataArray
            gauge observations.

        Returns
        -------
        da_rad_out: xarray.DataArray
            DataArray with the same structure as the ds_rad but with the CML
            adjusted radar field.
        """
        # Evaluate radar at cml and gauge ground positions
        rad, obs, x0 = self.radar_at_ground_(da_rad, da_cml=da_cml, da_gauge=da_gauge)

        # Calculate radar-instrument difference if radar has observation
        diff = np.where(rad > 0, obs - rad, np.nan)

        # Get index of not-nan obs
        keep = np.where(~np.isnan(diff))[0]

        # Check that that there is enough observations
        if keep.size > self.min_obs_:
            # get timestamp
            time = da_rad.time.data[0]

            # Remove radar time dimension
            da_rad_t = da_rad.sel(time=time)

            # do addtitive IDW merging
            adjusted = merge_functions.merge_additive_idw(
                xr.where(da_rad_t > 0, da_rad_t, np.nan),  # function skips nan
                diff[keep],
                x0[keep, :],
            )

            # Replace nan with original radar data (so that da_rad nan is kept)
            adjusted = xr.where(np.isnan(adjusted), da_rad_t, adjusted)

            # Re-assign timestamp and return
            return adjusted.assign_coords(time=time)

        # Else return the unadjusted radar
        return da_rad


class MergeAdditiveBlockKriging(Merge):
    """Merge CML and radar using an additive block kriging.

    Marges the provided radar field in ds_rad to CML observations by
    interpolating the difference between the CML and radar observations using
    block kriging.
    """

    def __init__(
        self,
        grid_location_radar="center",
        min_obs=5,
        discretization=8,
    ):
        Merge.__init__(self, grid_location_radar, min_obs)

        # Number of discretization points along CML
        self.discretization = discretization

    def update(self, da_rad, da_cml=None, da_gauge=None):
        """Update weights and x0 geometry for CML and gauge assuming block data

        This function uses the full CML geometry and makes the gauge geometry
        look like a rain gauge.

        Parameters
        ----------
        da_rad: xarray.DataArray
            Gridded radar data.
            and ys.
        da_cml: xarray.DataArray
            CML observations.
        da_gauge: xarray.DataArray
            gauge observations.
        """
        self.update_block_(
            da_rad, self.discretization, da_cml=da_cml, da_gauge=da_gauge
        )

    def adjust(self, da_rad, da_cml=None, da_gauge=None, variogram=None):
        """Adjust radar field for one time step.

        Evaluates if we have enough observations to adjust the radar field.
        Then adjust radar field to observations using additive IDW.

        Parameters
        ----------
        da_rad: xarray.DataArray
            Gridded radar data.
        da_cml: xarray.DataArray
            CML observations.
        da_gauge: xarray.DataArray
            gauge observations.
        variogram: function
            Function returning expected variance given distance as input.

        Returns
        -------
        da_rad_out: xarray.DataArray
            DataArray with the same structure as the ds_rad but with the CML
            adjusted radar field.
        """
        if variogram is None:
            merge_functions.estimate_variogram()

        # Evaluate radar at cml and gauge ground positions
        rad, obs, x0 = self.radar_at_ground_(da_rad, da_cml=da_cml, da_gauge=da_gauge)

        # Calculate radar-instrument difference if radar has observation
        diff = np.where(rad > 0, obs - rad, np.nan)

        # Get index of not-nan obs
        keep = np.where(~np.isnan(diff))[0]

        # Check that that there is enough observations
        if keep.size > self.min_obs_:
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
            )

            # Replace nan with original radar data (so that da_rad nan is kept)
            adjusted = xr.where(np.isnan(adjusted), da_rad_t, adjusted)

            # Re-assign timestamp and return
            return adjusted.assign_coords(time=time)

        # Else return the unadjusted radar
        return da_rad


class MergeBlockKrigingExternalDrift(Merge):
    """Merge CML and radar using block-kriging with external drift.

    Marges the provided radar field in ds_rad to CML observations by
    interpolating the difference between the CML and radar observations using
    block kriging.

    Parameters
    ----------
    grid_location_radar: str
        String indicating the grid location of the radar.
    min_obs: int
        Minimum number of unique observations needed in order to do adjustment.
    discretization: int
        Number of points to discretize the CML into.
    """

    def __init__(
        self,
        grid_location_radar="center",
        min_obs=5,
        discretization=8,
    ):
        Merge.__init__(self, grid_location_radar, min_obs)

        # Number of discretization points along CML
        self.discretization = discretization

    def update(self, da_rad, da_cml=None, da_gauge=None):
        """Update weights and x0 geometry for CML and gauge assuming block data

        This function uses the full CML geometry and makes the gauge geometry
        look like a rain gauge.

        Parameters
        ----------
        da_rad: xarray.DataArray
            Gridded radar data.
        da_cml: xarray.DataArray
            CML observations.
        da_gauge: xarray.DataArray
            gauge observations.
        """
        self.update_block_(
            da_rad, self.discretization, da_cml=da_cml, da_gauge=da_gauge
        )

    def adjust(
        self,
        da_rad,
        da_cml=None,
        da_gauge=None,
        variogram=None,
        transform=None,
        backtransform=None,
    ):
        """Adjust radar field for one time step.

        Evaluates if we have enough observations to adjust the radar field.
        Then adjust radar field to observations using additive IDW.

        Parameters
        ----------
        da_rad: xarray.DataArray
            Gridded radar data.
        da_cml: xarray.DataArray
            CML observations.
        da_gauge: xarray.DataArray
            gauge observations.
            Function returning expected variance given distance as input.
        transform: function
            Function transforming the data to follow a normal distribution.
        backtransform: function
            Function backtransforming the data from normal distribution.

        Returns
        -------
        da_rad_out: xarray.DataArray
            DataArray with the same structure as the ds_rad but with the CML
            adjusted radar field.
        """
        if variogram is None:
            merge_functions.estimate_variogram()

        if (transform is None) | (backtransform is None):
            merge_functions.estimate_transformation()

        # Evaluate radar at cml and gauge ground positions
        rad, obs, x0 = self.radar_at_ground_(da_rad, da_cml=da_cml, da_gauge=da_gauge)

        # Get index of not-nan obs
        keep = np.where(~np.isnan(obs) & ~np.isnan(rad) & (obs > 0) & (rad > 0))[0]

        # Check that that there is enough observations
        if keep.size > self.min_obs_:
            # get timestamp
            time = da_rad.time.data[0]

            # Remove radar time dimension
            da_rad_t = da_rad.sel(time=time)

            # do addtitive IDW merging
            adjusted = merge_functions.merge_ked_blockkriging(
                xr.where(da_rad_t > 0, da_rad_t, np.nan),  # function skips nan
                rad[keep],
                transform(obs)[keep],
                x0[keep],
                variogram,
            )

            # Replace nan with original radar data (so that da_rad nan is kept)
            adjusted = xr.where(np.isnan(adjusted), da_rad_t, backtransform(adjusted))

            # Re-assign timestamp and return
            return adjusted.assign_coords(time=time)

        # Else return the unadjusted radar
        return da_rad
