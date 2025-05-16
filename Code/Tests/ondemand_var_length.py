import itertools as it

import graphs
import network
import Simulation.ondemand as ond
import Tests.speedup as speedup

L1 = [20, 11, 15, 50, 30, 10]  # Varied/longer
L2 = [20, 19, 18, 20, 18, 20]  # Similar/closer
n_peers = len(L1)
rates = (55 * 2 * 6) / 2 / ((n_peers) * (n_peers - 1) / 2)  # 55hz for 4 peers scaled
runtime = 50  # sec
seed = 0


n1 = network.curry(L1, rates, n_peers=n_peers, seed=seed)
n2 = network.curry(L2, rates, n_peers=n_peers, seed=seed)


sims = speedup.run(
    [
        s(net[0](), runtime, extra_name=net[1])
        for net, s in it.product(
            [(n1, "-diff_links"), (n2, "-sim_links")],
            [
                ond.Parallel,
                ond.Sequential,
                ond.Smart,
                ond.ParallelQoS,
                ond.SequentialQoS,
            ],
        )
    ]
)
for s in sims:
    s.run()
graphs.plot_simulation_means(
    sims, title="Performance of OnDemand Variations with Varying Link Distances"
)
graphs.plot_latency_vs_fidelity(sims)
graphs.plot_simulation_edgewise_means(
    sims, title="Edgewise Performances of Single OnDemand Variations"
)
