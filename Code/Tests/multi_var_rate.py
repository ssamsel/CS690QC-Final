import itertools as it

import graphs
import network
import Simulation.multi as multi
import Tests.speedup as speedup
import utils

L = [20, 10, 15, 50]
n_peers = len(L)
rates1 = utils.convert_to_hz("55hz")
rates2 = utils.convert_to_hz(
    [
        [0, "200hz", "100hz", "35hz"],
        [0, 0, "175hz", "100hz"],
        [0, 0, 0, "50hz"],
        [0, 0, 0, 0],
    ]
)
runtime = 100  # sec
seed = 0


n1 = network.curry(L, rates1, n_peers=n_peers, seed=seed)
n2 = network.curry(L, rates2, n_peers=n_peers, seed=seed)


sims = speedup.run(
    [
        s(net[0](), runtime, extra_name=net[1])
        for net, s in it.product(
            [(n1, "-fixed_rates"), (n2, "-mixed_rates")],
            [
                multi.RandomMatchSimulation,
                multi.YoungestEPRFirstSimulation,
                multi.RandomMatchSimulationQoS,
                multi.YoungestEPRFirstSimulationQoS,
            ],
        )
    ]
)
for s in sims:
    s.run()
graphs.plot_simulation_means(sims, title="Performance of multi Variations")
graphs.plot_latency_vs_fidelity(sims)
graphs.plot_simulation_edgewise_means(
    sims, title="Edgewise Performances of multi Variations"
)
