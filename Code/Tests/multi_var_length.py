import itertools as it

import graphs
import network
import Simulation.multi as multi
import Tests.speedup as speedup
import utils

seed = 0
L1 = [20, 10, 15, 50, 30, 10]
L2 = [20, 19, 18, 20, 18, 20]
n_peers = len(L1)
rates = utils.convert_to_hz("55hz")
rates = utils.convert_to_hz("30hz")
runtime = 100  # sec


n1 = network.curry(L1, rates, n_peers=n_peers, seed=seed)
n2 = network.curry(L2, rates, n_peers=n_peers, seed=seed)


sims = speedup.run(
    [
        s(net[0](), runtime, extra_name=net[1])
        for net, s in it.product(
            [(n1, "-diff_links"), (n2, "-sim_links")],
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
