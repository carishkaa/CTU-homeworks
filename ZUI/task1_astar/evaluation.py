#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
import sys
import math
import time
import numpy as np
import matplotlib.pyplot as plt
 
sys.path.append('gridmap')
sys.path.append('gridplanner')
 
import GridMap as gmap
import GridPlanner as gplanner
 
PLOT_ENABLE = True
 
if __name__=="__main__":
 
    #define planning problems:
    #  map file 
    #  start position [m]
    #  goal position [m]
    #  execute flag
    scenarios = [
                ("maps/public_maze01.csv", (1, 3), (9, 3), 'N8')
                ,("maps/public_maze02.csv", (1, 9), (9, 3), 'N8')
                ,("maps/public_maze03.csv", (1, 7), (9, 12), 'N8')
                ,("maps/public_maze04.csv", (19, 16), (8.5, 8.5), 'N4')
                ,("maps/public_maze05.csv", (32, 44.5), (5, 8.5), 'N4')
                ]
 
    # fetch individual scenarios
    for scenario in scenarios:
        mapfile = scenario[0] # the name of the map
        start = scenario[1]   # start point
        goal = scenario[2]    # goal point
        neigh = scenario[3]   # neighborhood type
 
        # instantiate the map
        gridmap = gmap.GridMap()
 
        # load map from file
        gridmap.load_map(mapfile, 0.1)
 
        # plot the map with the start and goal positions
        if PLOT_ENABLE:
            gmap.plot_map(gridmap)
            gmap.plot_path(gridmap.world_to_map([start, goal]))
            plt.show()
 
        planner = gplanner.GridPlanner()
 
        # plan the route from start to goal
        path = planner.plan(gridmap, gridmap.world_to_map(start), gridmap.world_to_map(goal), neigh)
 
        if path == None:
            print("Destination unreachable")
            continue
 
        # show the planned path
        if PLOT_ENABLE:
            gmap.plot_map(gridmap)
            gmap.plot_path(path)
            plt.show()

        print('----------')
        print('maze : %s' % mapfile)
        print('expanded nodes : %d' % gridmap.get_expanded_nodes())
