"""
Created on Fri Oct 18 20:21:53 2024

@author: erlend
"""

import numpy as np
import pykrige
import xarray as xr
import scipy
import poligrain as plg

class OBKrigTree:
    """ Ordinary block kriging interpolation using KDTree. 

    Interpolate CML and rain gauge data using an modified neighbourhood 
    based implementation of block kriging. See Journel and Huijbregts
    (1978) Mining Geostatistics. 
    """
    def __init__(
            self, 
            variogram,
            ds_cmls=None, 
            ds_gauges=None, 
            discretization=8,
            nnear=8,
            max_distance=60000,
            ):

        """ Construct kriging matrices and block geometry.

        Relies on xarray datasets for CML and rain gauge data following
        the OpenSense naming convention.

        Parameters
        ----------
        variogram: function
            A user defined function defining the variogram. Input 
            distance, returns the expected variance.
        ds_cmls: xarray.Dataset
            CML dataset or data array. Must contain the projected coordinates 
            of the CML (site_0_x, site_0_y, site_1_x, da_cml.site_1_y).
        ds_gauges: xarray.Dataset
            Gauge dataset or data array. Must contain the projected 
            coordinates of the rain gauges (x, y).
        discretization: int
            Number of intervals to discretize the CML into.
        nnear: int
            Number of neighbors to include in neighbourhood.
        max_distance: float
            Max distance for including observation in neighbourhood.
        """
        if ds_cmls and ds_gauges:
            # Get structured coordinates of CML and rain gauge
            x0_cml = calculate_cml_line(ds_cmls).data
            x0_gauge = calculate_gauge_midpoint(ds_gauges)
            
            # Create block geometry for rain gauge
            x0_gauge = x0_gauge.expand_dims(
                disc=range(discretization + 1)
            ).transpose("id", "yx", "disc").data

            # Store cml and gauge block geometry
            x0 = np.vstack([x0_cml, x0_gauge])

        elif ds_cmls:
            # Get structured coordinates of CML and store
            x0 = calculate_cml_line(ds_cmls).data
        
        elif ds_gauges:
            # Get structured coordinates of rain gauge
            x0_gauge = calculate_gauge_midpoint(ds_gauges)
            
            # Create block geometry for rain gauge and store
            x0 = x0_gauge.expand_dims(
                disc=range(discretization + 1)
            ).transpose("id", "yx", "disc").data
        
        # Calculate lengths within all CMLs
        lengths_within_l = within_block_l(x0)

        # Estimate variance within blocks
        var_within = variogram(lengths_within_l).mean(axis=(1, 2))

        # Mean variance within block pairs
        var_within = 0.5 * (var_within + var_within.reshape(-1, 1))

        # Calculate lengths between all points along all blocks
        lengths_block_l = block_points_to_lengths(x0)

        # Average variance across blocks
        var_block = variogram(lengths_block_l).mean(axis=(2, 3))

        # Subtract within from block and turn into covariance
        cov_mat = var_within - var_block

        # Add nugget value to diagonal 
        nugget = variogram(np.array([0.0]),)
        np.fill_diagonal(cov_mat, nugget)

        # Create Kriging matrix
        mat = np.zeros([cov_mat.shape[0] + 1, cov_mat.shape[1] + 1])
        mat[: cov_mat.shape[0], : cov_mat.shape[1]] = cov_mat
        mat[-1, :-1] = np.ones(cov_mat.shape[1])  # non-bias condition
        mat[:-1, -1] = np.ones(cov_mat.shape[0])  # lagrange multipliers

        # Store data to self
        self.mat = mat
        self.x0 = x0
        self.var_within = np.diag(var_within)
        self.n_obs = self.var_within.size
        self.variogram = variogram
        self.nnear = nnear
        self.max_distance = max_distance
    
    def __call__(self, points, da_cmls=None, da_gauges=None):
        """ Construct kriging matrices and block geometry.

        Parameters
        ----------
        points numpy.array
            2D array containing the coordaintes as [y x]. 
        da_cmls: xarray.DataArray
            CML data array. Must contain observations for one time step
            and the projected coordinates of the CML 
            (site_0_x, site_0_y, site_1_x, da_cml.site_1_y).
        da_gauges: xarray.Dataset
            Gauge data array. Must contain the observations for one time 
            step and the projected coordinates of the rain gauges (x, y).

        Returns
        -------
        estimate: numpy.array
            1D array with the same length as the number of coordinates
            (points.shape[0]) containing the interpolated field.
        """
        # Get observations
        if (da_cmls is not None) and (da_gauges is not None): 
            obs = np.concatenate([da_cmls, da_gauges])

        elif (da_cmls is not None):
            obs = da_cmls.data.flatten()
        
        elif (da_gauges is not None):
            obs = da_gauges.data.flatten()
        
        else:
            msg = "provide da_cmls or da_gauges"
            raise ValueError(msg)
        
        # Test that number of observations match init
        if obs.size > self.n_obs:
            msg = "numer of observations changed, reinitialize"
            raise ValueError(msg)

        # Coordinates of neighbour
        x_neighbours = self.x0[:, 0, int(self.x0.shape[2] / 2)]
        y_neighbours = self.x0[:, 1, int(self.x0.shape[2] / 2)]
        
        # Get neighbourhood, links represented by midpoint
        xgrid, ygrid = points[:, 0], points[:, 1]
        tree_neighbors = scipy.spatial.KDTree(
            data=list(zip(x_neighbours, y_neighbours))
        )
        distances, ixs = tree_neighbors.query(
            list(zip(xgrid, ygrid)),
            k=self.nnear,
            distance_upper_bound=self.max_distance,
        )

        # Vectorized difference estimate
        y_reshaped = points[:, 0, np.newaxis, np.newaxis]
        x_reshaped = points[:, 1, np.newaxis, np.newaxis]
        delta_y = self.x0[:, 0, :] - y_reshaped
        delta_x = self.x0[:, 1, :] - x_reshaped
        lengths = np.sqrt(delta_x**2 + delta_y**2)

        # Estimate expected variance for all links
        var_line_point = self.variogram(lengths).mean(axis = 2)
        
        # Array for storing estimate
        estimate = np.zeros(xgrid.size)
   
        # Compute the contributions from nearby CMLs to points in grid
        for i in range(xgrid.size):
            # kdtree sets missing neighbours to len(neighbourhood)
            ind = ixs[i][ixs[i] < self.n_obs]
            
            # Append the non-bias indices to krigin matrix lookup
            i_mat = np.append(ind, self.n_obs)
            
            # Subtract withinblock covariance of the blocks
            target = -1 * (var_line_point[i, ind] - self.var_within[ind])

            # Add non bias condition
            target = np.append(target, 1)
            
            # Solve the kriging system
            w = np.linalg.solve(self.mat[np.ix_(i_mat, i_mat)], target)[:-1]

            # Estimate rainfall amounts at location i
            estimate[i] = obs[ind] @ w

        # Return dataset with interpolated values
        return estimate

class BKEDTree:
    """ Block kriging with external drift merging using KDTree. 

    Merge radar and CML and rain gauge data using an modified 
    neighbourhood based implementation of block kriging with 
    external drift. See Journel and Huijbregts (1978) 
    Mining Geostatistics. 
    """
    def __init__(
            self, 
            variogram,
            rad_cmls=None,
            rad_gauges=None,
            ds_cmls=None, 
            ds_gauges=None,
            discretization=8,
            nnear=8,
            max_distance=60000,
            ):

        """ Construct kriging matrices and block geometry.

        Relies on xarray datasets for CML and rain gauge data following
        the OpenSense naming convention.

        Parameters
        ----------
        variogram: function
            A user defined function defining the variogram. Input 
            distance, returns the expected variance.
        rad_cmls: numpy.array or xr.DataArray
            Radar observations at CML positions.
        rad_gauges: numpy.array or xr.DataArray
            Radar observations at rain gauge positions.
        ds_cmls: xarray.Dataset
            CML dataset or data array. Must contain the projected coordinates 
            of the CML (site_0_x, site_0_y, site_1_x, da_cml.site_1_y).
        ds_gauges: xarray.Dataset
            Gauge dataset or data array. Must contain the projected 
            coordinates of the rain gauges (x, y).
        discretization: int
            Number of intervals to discretize the CML into.
        nnear: int
            Number of neighbors to include in neighbourhood.
        max_distance: float
            Max distance for including observation in neighbourhood.
        """
        if ds_cmls and ds_gauges:
            # Get structured coordinates of CML and rain gauge
            x0_cml = calculate_cml_line(ds_cmls).data
            x0_gauge = calculate_gauge_midpoint(ds_gauges)
            
            # Create block geometry for rain gauge
            x0_gauge = x0_gauge.expand_dims(
                disc=range(discretization + 1)
            ).transpose("id", "yx", "disc").data

            # Store cml and gauge block geometry
            x0 = np.vstack([x0_cml, x0_gauge])

            # Radar observations
            rad_obs = np.concatenate([rad_cmls, rad_gauges])

        elif ds_cmls:
            # Get structured coordinates of CML and store
            x0 = calculate_cml_line(ds_cmls).data
            
            # Radar observations
            rad_obs = np.array(rad_cmls)
        
        elif ds_gauges:
            # Get structured coordinates of rain gauge
            x0_gauge = calculate_gauge_midpoint(ds_gauges)
            
            # Create block geometry for rain gauge and store
            x0 = x0_gauge.expand_dims(
                disc=range(discretization + 1)
            ).transpose("id", "yx", "disc").data
        
            # Radar observations
            rad_obs = np.array(rad_gauges)

        # Calculate lengths within all CMLs
        lengths_within_l = within_block_l(x0)

        # Estimate variance within blocks
        var_within = variogram(lengths_within_l).mean(axis=(1, 2))

        # Mean variance within block pairs
        var_within = 0.5 * (var_within + var_within.reshape(-1, 1))

        # Calculate lengths between all points along all blocks
        lengths_block_l = block_points_to_lengths(x0)

        # Average variance across blocks
        var_block = variogram(lengths_block_l).mean(axis=(2, 3))

        # Subtract within from block and turn into covariance
        cov_mat = var_within - var_block

        # Add nugget value to diagonal 
        nugget = variogram(np.array([0.0]),)
        np.fill_diagonal(cov_mat, nugget)

        # Create Kriging matrix
        mat = np.zeros([cov_mat.shape[0] + 2, cov_mat.shape[1] + 2])
        mat[: cov_mat.shape[0], : cov_mat.shape[1]] = cov_mat
        mat[-2, :-2] = np.ones(cov_mat.shape[1])  # non-bias condition
        mat[-1, :-2] = rad_obs  # Radar drift
        mat[:-2, -2] = np.ones(cov_mat.shape[0])  # non-bias condition
        mat[:-2, -1] = rad_obs  # Radar drift

        # Store data to self
        self.mat = mat
        self.x0 = x0
        self.var_within = np.diag(var_within)
        self.n_obs = self.var_within.size
        self.variogram = variogram
        self.nnear = nnear
        self.max_distance = max_distance
    
    def __call__(self, points, rad_field, da_cmls=None, da_gauges=None):
        """ Construct kriging matrices and block geometry.

        Parameters
        ----------
        points: numpy.array
            2D array containing the coordaintes as [y x]. 
        rad_field: numpy.array
            1D array containing the radar observation at the coordinates
            stored in points.
        da_cmls: xarray.DataArray
            CML data array. Must contain observations for one time step
            and the projected coordinates of the CML 
            (site_0_x, site_0_y, site_1_x, da_cml.site_1_y).
        da_gauges: xarray.Dataset
            Gauge data array. Must contain the observations for one time 
            step and the projected coordinates of the rain gauges (x, y).

        Returns
        -------
        estimate: numpy.array
            1D array with the same length as the number of coordinates
            (points.shape[0]) containing the interpolated field.
        """
        # Get observations
        if (da_cmls is not None) and (da_gauges is not None): 
            obs = np.concatenate([da_cmls, da_gauges])

        elif (da_cmls is not None):
            obs = da_cmls.data.flatten()
        
        elif (da_gauges is not None):
            obs = da_gauges.data.flatten()
        
        else:
            msg = "provide da_cmls or da_gauges"
            raise ValueError(msg)
        
        # Test that number of observations match init
        if obs.size > self.n_obs:
            msg = "observations does not match init, re-initialize"
            raise ValueError(msg)

        # Coordinates of neighbour
        x_neighbours = self.x0[:, 0, int(self.x0.shape[2] / 2)]
        y_neighbours = self.x0[:, 1, int(self.x0.shape[2] / 2)]
        
        # Get neighbourhood, links represented by midpoint
        xgrid, ygrid = points[:, 0], points[:, 1]
        tree_neighbors = scipy.spatial.KDTree(
            data=list(zip(x_neighbours, y_neighbours))
        )
        distances, ixs = tree_neighbors.query(
            list(zip(xgrid, ygrid)),
            k=self.nnear,
            distance_upper_bound=self.max_distance,
        )

        # Vectorized difference estimate
        y_reshaped = points[:, 0, np.newaxis, np.newaxis]
        x_reshaped = points[:, 1, np.newaxis, np.newaxis]
        delta_y = self.x0[:, 0, :] - y_reshaped
        delta_x = self.x0[:, 1, :] - x_reshaped
        lengths = np.sqrt(delta_x**2 + delta_y**2)

        # Estimate expected variance for all links
        var_line_point = self.variogram(lengths).mean(axis = 2)
        
        # Array for storing estimate
        estimate = np.zeros(xgrid.size)
   
        # Compute the contributions from nearby CMLs to points in grid
        for i in range(xgrid.size):
            # kdtree sets missing neighbours to len(neighbourhood)
            ind = ixs[i][ixs[i] < self.n_obs]
            
            # Append the non-bias and rad indices to krigin matrix lookup
            i_mat = np.append(ind, [self.n_obs - 1, self.n_obs])
            
            # Subtract withinblock covariance of the blocks
            target = -1 * (var_line_point[i, ind] - self.var_within[ind])

            # Add non bias condition and rad obs
            target = np.append(target, [1, rad_field[i]])
            
            # Solve the kriging system
            w = np.linalg.solve(self.mat[np.ix_(i_mat, i_mat)], target)[:-2]

            # Estimate rainfall amounts at location i
            estimate[i] = obs[ind] @ w

        # Return dataset with interpolated values
        return estimate

def within_block_l(x0):
    """Calculate the lengths within all CMLs.

    Given the numpy array x0 created by the function 'calculate_cml_geometry'
    this function calculates the length between all points within all CMLs.

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
    lengths_withinblock_l: np.array
        Array with lengths between all points along within the CMLs. The array
        is organized into a 4D matrix with the following structure:
            (cml_i: [0, ..., number of cmls],
             cml_i: [0, ..., number of cmls],
             length_i: [0, ..., number of points along cml].
             length_i: [0, ..., number of points along cml]).

        Accessing the length between point 0 along cml 0 and point 0 along
        cml 1 can then be done by lengths_point_l[0, 1, 0, 0]. The mean length
        can be calculated by lengths_point_l.mean(axis = (2, 3)).
    """
    # Estimate delta x within all blocks
    delta_x = np.array([x0[i][1] - x0[i][1].reshape(-1, 1) for i in range(x0.shape[0])])
    delta_y = np.array([x0[i][0] - x0[i][0].reshape(-1, 1) for i in range(x0.shape[0])])

    # Lengths within all blocks
    return np.sqrt(delta_x**2 + delta_y**2)


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


def construct_variogram(
    obs,
    x0,
    variogram_parameters,
    variogram_model,
):
    """Construct variogram

    Construct the variogram using pykrige variogram model and parameters
    provided by user.

    Returns
    -------
    variogram: function
        Variogram function that returns the expected variance given the
        distance between observations.

    """
    # If x0 contains block data, get approximate midpoints
    if len(x0.shape) > 2:
        x0 = x0[:, :, int(x0.shape[2] / 2)]

    return pykrige.OrdinaryKriging(
        x0[:, 1],  # x-midpoint coordinate
        x0[:, 0],  # y-midpoint coordinate
        obs,
        variogram_parameters=variogram_parameters,
        variogram_model=variogram_model,
    )

# Functions for setting up x0 for gauges and CMLs
def calculate_cml_line(ds_cmls, discretization=8):
    """Calculate the position of points along CMLs.

    Calculates the discretized CML line coordinates by dividing the CMLs into
    discretization-number of intervals. The ds_cmls xarray object must contain the
    projected coordinates (site_0_x, site_0_y, site_1_x site_1_y) defining
    the start and end point of the CML.

    Parameters
    ----------
    ds_cmls: xarray.Dataset
        CML geometry as a xarray object. Must contain the coordinates
        (site_0_x, site_0_y, site_1_x site_1_y)
    discretization: int
        Number of intervals to discretize lines into.

    Returns
    -------
    x0: xr.DataArray
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
    x0_cml = np.array([ypos, xpos]).transpose([1, 0, 2])

    # Turn into xarray dataarray and return
    return xr.DataArray(
        x0_cml,
        coords={
            "cml_id": ds_cmls.cml_id.data,
            "yx": ["y", "x"],
            "discretization": np.arange(discretization + 1),
        },
    )


def calculate_cml_midpoint(da_cml):
    """Calculate DataArray with midpoints of CMLs

    Calculates the CML midpoints and stores the results in an xr.DataArray.
    The da_cml xarray object must contain the projected coordinates (site_0_x,
    site_0_y, site_1_x site_1_y) defining the start and end point of the CML.

    Parameters
    ----------
    da_cml: xarray.DataArray
        CML geometry as a xarray object. Must contain the coordinates
        (site_0_x, site_0_y, site_1_x site_1_y)

    Returns
    -------
    x0: xr.DataArray
        Array with midpoints for all CMLs. The array is organized into a 2D
        matrix with the following structure:
            (number of n CMLs [0, ..., n],
             y/x-cooridnate [y, x],
    """
    # CML midpoint coordinates as columns
    x = ((da_cml.site_0_x + da_cml.site_1_x) / 2).data
    y = ((da_cml.site_0_y + da_cml.site_1_y) / 2).data

    # CML midpoint coordinates as columns
    x0_cml = np.hstack([y.reshape(-1, 1), x.reshape(-1, 1)])

    # Create dataarray and return
    return xr.DataArray(
        x0_cml,
        coords={
            "cml_id": da_cml.cml_id.data,
            "yx": ["y", "x"],
        },
    )


def calculate_gauge_midpoint(da_gauge):
    """Calculate DataArray with coordinates of raingauge

    Calculates the gauge coordinates and stores the results in an xr.DataArray.
    The da_gauge xarray object must contain the projected coordinates (y, x)
    defining the position of the raingauge.

    Parameters
    ----------
    da_gauge: xarray.DataArray
        Gauge coordinate as a xarray object. Must contain the coordinates (y, x)

    Returns
    -------
    x0: xr.DataArray
        Array with coordinates for all gauges. The array is organized into a 2D
        matrix with the following structure:
            (number of n gauges [0, ..., n],
             y/x-cooridnate [y, x],
    """
    x0_gauge = np.hstack(
        [
            da_gauge.y.data.reshape(-1, 1),
            da_gauge.x.data.reshape(-1, 1),
        ]
    )

    # Create dataarray return
    return xr.DataArray(
        x0_gauge,
        coords={
            "id": da_gauge.id.data,
            "yx": ["y", "x"],
        },
    )
