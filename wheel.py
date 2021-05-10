# -*- coding: utf-8 -*-

"""
A Wheel Network Topology

This class implements a wheel topology. In this topology,
the particles are connected to a single common particle.
"""

# Import standard library
import logging

# Import modules
import numpy as np
from pyswarms.backend import operators as ops
from pyswarms.backend.handlers import BoundaryHandler, VelocityHandler
from pyswarms.backend.topology import Topology
from pyswarms.utils.reporter import Reporter


class Wheel(Topology):
    def __init__(self, static=False):
        """Initializes the class

        Parameters
        ----------
        static : bool (Default is :code:`False`)
            a boolean that decides whether the topology
            is static or dynamic
        """
        super(Wheel, self).__init__(static)
        self.rep = Reporter(logger=logging.getLogger(__name__))

    def compute_gbest(self, swarm, **kwargs):
        """Update the global best using a wheel-like neighborhood approach

        Parameters
        ----------
        swarm : pyswarms.backend.swarms.Swarm
            a Swarm instance

        Returns
        -------
        numpy.ndarray
            Best position of shape :code:`(n_dimensions, )`
        float
            Best cost
        """
        try:
            # Check if the topology is static or not and assign neighbors
            if (self.static and self.neighbor_idx is None) or not self.static:
                # Obtain the number of particles of the swam
                n_particles = swarm.n_particles
                # Obtain the common particle
                common_particle = np.random.randint(n_particles)
                # Define the neighbourhood
                self.neighbor_idx = np.repeat(common_particle, n_particles).reshape(-1, 1).tolist()
                self.neighbor_idx[common_particle] = np.delete(np.arange(start=0, stop=n_particles),
                                                               common_particle).tolist()

            idx_min = np.array(
                [
                    swarm.pbest_cost[self.neighbor_idx[i]].argmin()
                    for i in range(len(self.neighbor_idx))
                ]
            )

            best_neighbor = np.array(
                [
                    self.neighbor_idx[i][idx_min[i]]
                    for i in range(len(self.neighbor_idx))
                ]
            ).astype(int)

            # Obtain best cost and position
            best_cost = np.min(swarm.pbest_cost[best_neighbor])
            best_pos = swarm.pbest_pos[best_neighbor]
        except AttributeError:
            self.rep.logger.exception(
                "Please pass a Swarm class. You passed {}".format(type(swarm))
            )
            raise
        else:
            return (best_pos, best_cost)

    def compute_velocity(
            self,
            swarm,
            clamp=None,
            vh=VelocityHandler(strategy="unmodified"),
            bounds=None,
    ):
        """Compute the velocity matrix

        This method updates the velocity matrix using the best and current
        positions of the swarm. The velocity matrix is computed using the
        cognitive and social terms of the swarm.

        A sample usage can be seen with the following:

        .. code-block :: python

            import pyswarms.backend as P
            from pyswarms.backend.swarm import Swarm
            from pyswarms.backend.handlers import VelocityHandler
            from pyswarms.backend.topology import Ring

            my_swarm = P.create_swarm(n_particles, dimensions)
            my_topology = Ring(static=False)
            my_vh = VelocityHandler(strategy="invert")

            for i in range(iters):
                # Inside the for-loop
                my_swarm.velocity = my_topology.update_velocity(my_swarm, clamp, my_vh,
                bounds)

        Parameters
        ----------
        swarm : pyswarms.backend.swarms.Swarm
            a Swarm instance
        clamp : tuple of floats (default is :code:`None`)
            a tuple of size 2 where the first entry is the minimum velocity
            and the second entry is the maximum velocity. It
            sets the limits for velocity clamping.
        vh : pyswarms.backend.handlers.VelocityHandler
            a VelocityHandler instance
        bounds : tuple of :code:`np.ndarray` or list (default is :code:`None`)
            a tuple of size 2 where the first entry is the minimum bound while
            the second entry is the maximum bound. Each array must be of shape
            :code:`(dimensions,)`.

        Returns
        -------
        numpy.ndarray
            Updated velocity matrix
        """
        return ops.compute_velocity(swarm, clamp, vh, bounds)

    def compute_position(
            self, swarm, bounds=None, bh=BoundaryHandler(strategy="periodic")
    ):
        """Update the position matrix

        This method updates the position matrix given the current position and
        the velocity. If bounded, it waives updating the position.

        Parameters
        ----------
        swarm : pyswarms.backend.swarms.Swarm
            a Swarm instance
        bounds : tuple of :code:`np.ndarray` or list (default is :code:`None`)
            a tuple of size 2 where the first entry is the minimum bound while
            the second entry is the maximum bound. Each array must be of shape
            :code:`(dimensions,)`.
        bh : pyswarms.backend.handlers.BoundaryHandler
            a BoundaryHandler instance

        Returns
        -------
        numpy.ndarray
            New position-matrix
        """
        return ops.compute_position(swarm, bounds, bh)
