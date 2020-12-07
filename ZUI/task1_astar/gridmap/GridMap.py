#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import math
import numpy as np
import matplotlib.pyplot as plt


class GridMap:

    def __init__(self, width=0, height=0, scale=0.1):
        """
        Parameters
        ----------
        width : int, optional
            width of the grid map
        height : int, optional
            height of the grid map
        scale : double, optional
            the scale of the map 
        """
        if width > 0 and height > 0:
            self.grid = np.zeros((width, height), dtype=[('p', np.dtype(float)), ('free', np.dtype(bool))])
            self.grid['p'] = 0.5     # set all probabilities to 0.5
            self.grid['free'] = True  # set all cells to passable
            self.width = width
            self.height = height
            self.scale = scale
        else:
            self.grid = None
            self.width = None
            self.height = None
            self.scale = None

        self.expanded_nodes = 0

    def get_expanded_nodes(self):
        return self.expanded_nodes

    def load_map(self, filename, scale=0.1):
        """ 
        Load map from file 
        
        Parameters
        ----------
        filename : str 
            path to the map file
        scale : double (optional)
            the scale of the map
        """
        grid = np.genfromtxt(filename, delimiter=',')
        grid = grid*0.95 + 0.025
        grid = np.flip(grid,0) #to cope with the vrep scene

        self.grid = np.zeros(grid.shape, dtype=[('p',np.dtype(float)),('free',np.dtype(bool))])
        self.grid['p'] = grid   #populate the occupancy grid
        self.grid['free'] = True    #assign passable and impassable cells
        self.grid['free'][grid > 0.5] = False

        self.width = self.grid.shape[0]
        self.height = self.grid.shape[1]
        self.scale = scale

    ##################################################
    # Planning helper functions
    ##################################################

    def in_bounds(self, coord):
        """
        Check the boundaries of the map

        Parameters
        ----------
        coord : (int, int)
            map coordinate

        Returns
        -------
        bool
            if the value is inside or outside the grid map dimensions
        """
        (x,y) = coord
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, coord):
        """
        Check the passability of the given cell

        Parameters
        ----------
        coord : (int, int)
            map coordinate

        Returns
        -------
        bool
            if the grid map cell is occupied or not
        """
        ret = False
        if self.in_bounds(coord):
            (x,y) = coord
            ret = self.grid[x,y]['free']
        return ret

    def neighbors(self, coord, neigh):
        self.expanded_nodes += 1
        n = []
        if neigh == 'N4':
            n = self.neighbors4(coord)
        elif neigh == 'N8':
            n = self.neighbors8(coord)
        return n

    def neighbors4(self, coord):
        """
        Returns coordinates of passable neighbors of the given cell in 4-neighborhood

        Parameters
        ----------
        coord : (int, int)
            map coordinate

        Returns
        -------
        list (int,int)
            list of neighbor coordinates 
        """
        (x,y) = coord
        results = [(x+1, y), (x, y-1), (x-1, y), (x, y+1)]
        results = list(filter(self.in_bounds, results))
        results = list(filter(self.passable, results))
        return results

    def neighbors8(self, coord):
        """
        Returns coordinates of passable neighbors of the given cell in 8-neighborhood

        Parameters
        ----------
        coord : (int, int)
            map coordinate

        Returns
        -------
        list (int,int)
            list of neighbor coordinates 
        """
        (x,y) = coord
        results = [(x+1, y+1), (x+1, y), (x+1, y-1),
                   (x,   y+1),           (x,   y-1),
                   (x-1, y+1), (x-1, y), (x-1, y-1)]
        results = list(filter(self.in_bounds, results))
        results = list(filter(self.passable, results))
        return results

    ##############################################
    ## Helper functions for coordinate calculation
    ##############################################

    def world_to_map(self, coord):
        """
        function to return the map coordinates given the world coordinates

        Parameters
        ----------
        coord: list(float, float)
            list of world coordinates

        Returns
        -------
        list(int, int)
            list of map coordinates using proper scale and closest neighbor
        """
        cc = np.array(coord)/self.scale
        ci = cc.astype(int)
        if len(ci) > 2:
            return tuple(map(tuple, ci))
        else:
            return tuple(ci)

    def map_to_world(self, coord):
        """
        function to return the world coordinates given the map coordinates

        Parameters
        ----------
        coord: list(int, int)
            list of map coordinates

        Returns
        -------
        list(float, float)
            list of world coordinates
        """
        ci = np.array(coord).astype(float)
        cc = ci*self.scale
        if len(cc) > 2:
            return tuple(map(tuple, cc))
        else:
            return tuple(cc)

##############################################
## Helper functions to plot the map
##############################################
def plot_map(grid_map, cmap="Greys", clf=True, data='p'):
    """
    Method to plot the occupany grid map
    
    Parameters
    ----------
    grid_map : GridMap
        the GridMap object
    cmap : String(optional)
        the color map to be used
    clf : bool (optional)
        switch to clear or not to clear the drawing canvas
    data : String (optional)
        The grid map data to be displayed, 'p' for occupancy grid, 'free' for obstacle map
    """
    if clf:  #clear the canvas
        plt.clf()
    if data == 'p':
        plt.imshow(grid_map.grid['p'].transpose(), cmap=cmap, interpolation='nearest', vmin=0, vmax=1, origin='lower')
        plt.xlabel('x [grid cells]')
        plt.ylabel('y [grid cells]')
    elif data == 'free':
        grid = grid_map.grid['free'].transpose()
        grid = np.array(grid, dtype=int)
        plt.xlabel('x [grid cells]')
        plt.ylabel('y [grid cells]')
        plt.imshow(grid, cmap=cmap, interpolation='nearest', vmin=0, vmax=1, origin='lower')
    else:
        print('Unknown data entry')
    plt.draw()

def plot_path(path, color="red", label=None):
    """
    Method to plot the path
    
    Parameters
    ----------
    path: list(int,int)
        the list of individual waypoints
    """
    x_val = [x[0] for x in path]
    y_val = [x[1] for x in path]
    if label is None:
        plt.plot(x_val, y_val, '.-', color=color)
    else:
        plt.plot(x_val, y_val, '.-', color=color, label=label)
    plt.plot(x_val[0], y_val[0], 'og')
    plt.plot(x_val[-1], y_val[-1], 'sk')
    plt.draw()
