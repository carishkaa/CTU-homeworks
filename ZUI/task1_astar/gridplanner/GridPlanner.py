#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from queue import PriorityQueue
import numpy as np

import GridMap as gmap


# heuristic function
def h(a, b, neigh):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])

    # Manhattan distance
    if neigh == 'N4':
        return dx + dy

    # Euclidean distance
    if neigh == 'N8':
        return np.sqrt(dx * dx + dy * dy)


# distance between two nodes
def d(node1, node2):
    dx = (node1[0] - node2[0])
    dy = (node1[1] - node2[1])
    return np.sqrt(dx * dx + dy * dy)


class GridPlanner:

    def __init__(self):
        pass

    def plan(self, gridmap, start, goal, neigh):
        """
        Method to plan the path

        Parameters
        ----------
        gridmap: GridMap
            gridmap of the environment
        start: (int,int)
            start coordinates
        goal:(int,int)
            goal coordinates
        neigh:string(optional)
            type of neighborhood for planning ('N4' - 4 cells, 'N8' - 8 cells)

        Returns
        -------
        list(int,int)
            the path between the start and the goal if there is one, None if there is no path
        """
        came_from = self.a_star_search(gridmap, start, goal, neigh)
        if goal not in came_from:
            return None
        path = self.reconstruct_path(came_from, start, goal)
        return path

    #########################################
    # A_STAR
    #########################################

    def a_star_search(self, graph, start, goal, neigh):
        """
        This is the function for you to implement.

        Parameters
        ----------
        graph: GridMap
            gridmap of the environment
        start: (int,int)
            start coordinates
        goal: (int,int)
            goal coordinates
        neigh:string(optional)
            type of neighborhood for planning ('N4' - 4 cells, 'N8' - 8 cells)

        Returns
        -------
        dict (int,int) -> (int, int)
            for every node in path give his predecessor.
        """

        open_list = PriorityQueue()
        open_list.put((0, start))

        came_from = {start: None}
        g = {start: 0}

        while not open_list.empty():
            current = open_list.get()[1]

            if current == goal:
                break

            for neighbor in graph.neighbors(current, neigh):
                new_g = g[current] + d(current, neighbor)
                if neighbor not in g or new_g < g[neighbor]:
                    g[neighbor] = new_g
                    came_from[neighbor] = current
                    f = g[neighbor] + h(neighbor, goal, neigh)
                    open_list.put((f, neighbor))
        return came_from

    #########################################
    # backtracking function
    #########################################

    def reconstruct_path(self, came_from, start, goal):
        current = goal
        path = [current]
        while current != start:
            current = came_from[current]
            path.append(current)
        path.append(start)  # optional
        path.reverse()  # optional
        return path
