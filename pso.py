import pyswarms.backend as p

from custom_functions import *


def run_pso(function, topology, no_particles, dimensions, max_no_iterations, max_no_times_no_improve):
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    # Extract function details
    function_name = function['Name']
    f = function['Expression']
    # Extract topology details
    topology_name = topology['Name']
    t = topology['Architecture']()
    # Create the swarm
    swarm = p.create_swarm(n_particles=no_particles, dimensions=dimensions, options=options)
    # Initialise the initial pbest costs
    swarm.pbest_cost = np.full(no_particles, np.inf)
    # Initialise number of iterations
    i = 0
    # Initialise stop criteria
    stop = False
    no_times_no_improve = 0
    previous_best_cost = np.inf
    why_stop = ''
    # Until the stop criteria is meet
    while not stop:
        # Compute cost for current position and personal best
        swarm.current_cost = abs(f(swarm.position))
        swarm.pbest_pos, swarm.pbest_cost = p.compute_pbest(swarm)
        # Update gbest from neighborhood
        swarm.best_pos, swarm.best_cost = t.compute_gbest(swarm)
        # Perform position velocity update
        swarm.velocity = t.compute_velocity(swarm)
        swarm.position = t.compute_position(swarm)
        # Update number of iterations
        i = i + 1
        # Check stop criteria
        # Maximum number of iterations
        if i == max_no_iterations:
            stop = True
            why_stop = 'Max. iterations'
        # Solutions close
        elif np.isclose(swarm.best_cost, previous_best_cost):
            no_times_no_improve = no_times_no_improve + 1
            if no_times_no_improve == max_no_times_no_improve:
                stop = True
                why_stop = 'No improvement'
        else:
            # Reset the number of times with no improvement
            no_times_no_improve = 0
        # Get the current best cost and proceed
        previous_best_cost = swarm.best_cost
    # Obtain the final best_cost and the final best_position
    final_best_cost = swarm.best_cost
    final_best_pos = swarm.pbest_pos[swarm.pbest_cost.argmin()]
    # Prepare array with results
    results = {'Test function': function_name,
               'Number of dimensions': dimensions,
               'Topology': topology_name,
               'Number of iterations': i,
               'Why stop?': why_stop,
               'Best score': final_best_cost,
               'Best position': final_best_pos}
    return results
