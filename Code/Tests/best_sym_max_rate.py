"""This is the file I used to generate the two figures in my report.
Must be run as a module if PYTHONPATH is not set appropriately.
This is true for all of these test scripts.
"""

import itertools as it

import graphs
import network
import numpy as np
import Simulation.ondemand as ond
import Tests.speedup as speedup
import utils

# L = 20  # Uniform
L = [20, 11, 15, 50, 30, 10]  # Varied/longer
# L = [20, 19, 18, 20, 18, 20]  # Similar/closer
min_rate = 1
max_rate = 9
steps = 8
n_peers = 6
runtime = 50  # sec
seed = None
algorithms = [
    ond.Smart,
    ond.Parallel,
    ond.Sequential,
    ond.ParallelQoS,
    ond.SequentialQoS,
]


names = [utils.canonical_name(c) for c in algorithms]
h = [f"{x}hz" for x in np.round(np.linspace(min_rate, max_rate, num=steps), 3)]
sims = speedup.run(
    [
        s(
            network.curry(L, utils.convert_to_hz(hz), n_peers=n_peers, seed=seed)(),
            runtime,
            extra_name=hz,
        )
        for s, hz in it.product(
            algorithms,
            h,
        )
    ]
)
for s in sims:
    s.run()

graphs.plot_lines(sims, h, names)
