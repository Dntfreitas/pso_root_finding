import pyswarms.backend.topology as a
import pyswarms.utils.functions.single_obj as f

from custom_functions import *
from pso import run_pso

# number of dimensions to try
test_dimensions = [10,
                   30,
                   50,
                   100]

# test functions to try
test_functions = [{'Name': 'Ackley', 'Expression': f.ackley},
                  {'Name': 'Rastrigin', 'Expression': f.rastrigin},
                  {'Name': 'Rosenbrock', 'Expression': f.rosenbrock},
                  {'Name': 'Schaffer2', 'Expression': f.schaffer2},
                  {'Name': 'Sphere', 'Expression': f.sphere},
                  {'Name': 'Griewank', 'Expression': griewank},
                  {'Name': 'Zakharov', 'Expression': zakharov},
                  {'Name': 'Cigar', 'Expression': cigar},
                  {'Name': 'Schwefel', 'Expression': schwefel},
                  {'Name': 'Salomon', 'Expression': salomon}]

# architectures
test_architectures = [{'Name': 'Star', 'Architecture': a.Star},
                      {'Name': 'Ring', 'Architecture': a.Ring},
                      {'Name': 'Pyramid', 'Architecture': a.Pyramid},
                      {'Name': 'Random', 'Architecture': a.Random},
                      {'Name': 'VonNeuman', 'Architecture': a.VonNeumann}]

# population sizes
test_no_particles = [24,
                     36,
                     48,
                     60]

function = test_functions[0]
topology = test_architectures[0]
no_particles = test_no_particles[0]
dimensions = test_dimensions[0]
max_no_iterations = 5000
max_no_times_no_improve = 5000

results = run_pso(function, topology, no_particles, dimensions, max_no_iterations, max_no_times_no_improve)
print(results)
