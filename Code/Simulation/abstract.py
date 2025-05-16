"""General methods and definitions for Simulations/Algorithms"""

import warnings
from abc import ABC, abstractmethod

import numpy as np
import utils
from network import Network
from tqdm import TqdmWarning, tqdm

warnings.filterwarnings("ignore", category=TqdmWarning)


class Simulation(ABC):
    """Abstract class for a simulation run and switch algorithm."""

    def __init__(self, network: Network, runtime: int, extra_name="", *args, **kwargs):
        """Initialize the simulation.

        Args:
            network (Network): the network to use
            runtime (int): how long to run the simulation for (lower bound)
            extra_name (str, optional): string to append to classname for graphing purposes. Defaults to "".
        """
        self.runtime = runtime
        self.network = network
        reverse_mapping = [0] * (self.network.n_peers * (self.network.n_peers - 1) // 2)
        for k, v in self.network.mapping.items():
            t = tuple(k)
            reverse_mapping[v] = f"{t[0]}⟷{t[1]}"
        self.reverse_mapping = reverse_mapping
        self.name = utils.canonical_name(self) + extra_name
        self.data = None

    def run(self, idx=None):
        """Run the simulation."""
        if self.network.time >= self.runtime:
            return
        prog_bar = tqdm(total=self.runtime, position=idx)
        while self.network.time < self.runtime:
            self.matching_decision()
            self.entanglement_decision()
            idle_time = self.idle_decision()
            if idle_time == 0:
                idle_time += self.network.idle_until_next_request()
            prog_bar.update(idle_time)
        prog_bar.close()

    @abstractmethod
    def matching_decision(self):
        """Perform BSMs (if at all)."""
        raise NotImplementedError()

    @abstractmethod
    def entanglement_decision(self):
        """Decide which peers to entangle, (if any)."""
        raise NotImplementedError()

    @abstractmethod
    def idle_decision(self):
        """Decide how long to idle for."""
        raise NotImplementedError()

    def parse_results(self):
        """Calculate performance metrics from the simulation run.
        Utilizes memoization for efficiency and since the request
        queue is destroyed upon first calculation.
        Returns:
            dict[str, NDArray]: aggregated data
        """
        if self.data is not None:
            return self.data

        remaining = np.zeros(self.network.edge_count)
        for i, q in enumerate(self.network.request_queues):
            while not q.empty():
                if q.get() >= self.network.time:
                    break
                remaining[i] += 1

        edgewise_tput = self.network.tputs / self.network.time
        edgewise_rel_tput = self.network.tputs / (self.network.tputs + remaining)

        total_served_requests = np.sum(self.network.tputs)
        tput = total_served_requests / self.network.time
        rel_tput = total_served_requests / (total_served_requests + np.sum(remaining))
        edgewise_fidelity_stats = np.array(
            [
                (
                    np.mean(
                        edge_fidelities,
                    ),
                    np.var(edge_fidelities),
                )
                for edge_fidelities in self.network.fidelities
            ]
        )
        _flat_fidelities = utils.flatten2d(self.network.fidelities)
        fidelity_stats = np.array((np.mean(_flat_fidelities), np.var(_flat_fidelities)))

        edgewise_latency_stats = np.array(
            [
                (np.mean(edge_latencies), np.var(edge_latencies))
                for edge_latencies in self.network.latencies
            ]
        )
        _flat_latencies = utils.flatten2d(self.network.latencies)
        latency_stats = np.array((np.mean(_flat_latencies), np.var(_flat_latencies)))

        peerwise_age_stats = np.array(
            [(np.mean(peer_ages), np.var(peer_ages)) for peer_ages in self.network.ages]
        )
        _flat_ages = utils.flatten2d(self.network.ages)
        age_stats = np.array((np.mean(_flat_ages), np.var(_flat_ages)))
        self.data = {
            "edge_tput": edgewise_tput,
            "edge_rel_tput": edgewise_rel_tput,
            "edge_fidelity": edgewise_fidelity_stats,
            "edge_latency": edgewise_latency_stats,
            "fidelity": fidelity_stats,
            "latency": latency_stats,
            "tput": tput,
            "rel_tput": rel_tput,
            "peer_age": peerwise_age_stats,
            "age": age_stats,
        }
        return self.data


class BasicSimulation(Simulation):
    """Abstract simulation that entangles all unentangled peers and idles until another entanglement succeeds."""

    def entanglement_decision(self):
        self.network.entangle_unentangled()

    def idle_decision(self):
        return self.network.idle_until_next_entanglement()


class OnDemandSimulation(Simulation):
    """Version os switch/algorithm that only has 2 communications qubits."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min = None
        self.busy = 0

    def idle_decision(self):
        assert (
            len([x for x in self.network.bell_states if x != -1]) <= 2
        ), "Only 2 comm qubits allowed!"
        self.busy += (t := self.network.idle_until_next_entanglement())
        return t

    def util(self):
        """Calculate empirical utilization."""
        return self.busy / self.network.time

    def util_exp(self):
        """Calculate expected utilization.
        Tot always possible so defaults to 0 if unimplemented."""
        return 0


class _LocalClasses:
    """Fix weird pickling errors when multiprocessing.
    It is a modification of (Top Answer) --> Method 3
    from https://stackoverflow.com/questions/72766345/attributeerror-cant-pickle-local-object-in-multiprocessing
    """

    @classmethod
    def add_class(cls, *args):
        for c in args:
            setattr(cls, c.__name__, c)
            c.__qualname__ = cls.__qualname__ + "." + c.__name__


def buildQoSFidelitySimulation(parent: type[Simulation]):
    """Create a Simulation class that has a fidelity QoS guarantee.

    Args:
        parent (type[Simulation]): the base algorithm/simulation type

    Returns:
        type[QoS[type[parent]]]: QoS version of passed simulation type.
    """

    class QoS(parent):
        def __init__(self, *args, fidelity_guarantee=0.8, **kwargs):
            self.fidelity_guarantee = fidelity_guarantee
            super().__init__(*args, **kwargs)

        def matching_decision(self):
            matchings = self.network.current_valid_matchings()
            if len(matchings) > 0:
                discard_set = set()
                for matching in matchings:
                    for edge in matching:
                        if (
                            self.network.calc_post_swap_fidelity(*edge)
                            < self.fidelity_guarantee
                        ):
                            discard_set |= edge
                self.network.discard([*discard_set])
            return super().matching_decision()

    QoS.__name__ = utils.canonical_name(parent) + utils.canonical_name(QoS)
    QoS.__qualname__ = utils.canonical_name(parent) + utils.canonical_name(QoS)
    _LocalClasses.add_class(QoS)
    return QoS
