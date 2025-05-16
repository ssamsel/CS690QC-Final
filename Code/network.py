import itertools as it
from collections.abc import Iterable
from functools import wraps
from typing import overload

import constants
import numpy as np
import utils


def check_bell_states(func):
    """Decorator that validates bell states entanglements are not in the future."""

    @wraps(func)
    def outer(self, *peers):
        assert -1 < self.bell_states[peers[0]] <= self.time, "Peer 1 bell state invalid"
        assert -1 < self.bell_states[peers[1]] <= self.time, "Peer 2 bell state invalid"
        return func(self, *peers)

    return outer


class Network:
    """Class that contains the state of the network/switch.
    Is managed by a Simulation class."""

    @utils.wrap_to_np
    def __init__(self, L, req_rates, n_peers=None, seed=None):
        """Initialize Network Parameters.

        Args:
            L (list[float] | float): Link lengths in km.
            req_rates (list[list[float]] | float): NxN matrix of requests rates in req/sec.
            n_peers (int, optional): Number of peers. Must be set if L and req_rates are scalar.
            seed (int, optional): RNG seed. Defaults to None.
        """

        # Check parameters
        if n_peers is None:
            for arg in (L, req_rates):
                if isinstance(arg, Iterable):
                    n_peers = arg.shape[0]
                    break
            else:
                assert False, "Must have size set if L and request_rates is 1d"
        # Map scalars to vector, is a noop is already not a scalar
        L = L * np.ones(n_peers)
        req_rates = req_rates * np.ones([n_peers, n_peers])

        self.rng = np.random.default_rng(seed=seed)

        self.time = 0
        self.n_peers = n_peers

        self.entangle_probs = constants.calc_epr_prob_bk(L)
        self.tau = constants.tau(L)
        self.request_rates = (req_rates + req_rates.T)[
            np.triu_indices(self.n_peers, 1)
        ]  # Add complimentary request rates then flatten to 1d array

        self.bell_states = -np.ones(self.n_peers)

        self.edge_count = (
            self.n_peers * (self.n_peers - 1) // 2
        )  # Number of edges in an interconnected graph with self.size vertices
        self.request_queues = [utils.Queue() for _ in range(self.edge_count)]

        self.fidelities = [list() for _ in range(self.edge_count)]
        self.latencies = [list() for _ in range(self.edge_count)]
        self.tputs = np.zeros(self.edge_count)
        self.ages = [list() for _ in range(self.n_peers)]

        self.mapping = {
            frozenset(edge): idx
            for idx, edge in enumerate(it.combinations(range(self.n_peers), 2))
        }
        self.reverse_mapping = {v: k for k, v in self.mapping.items()}
        self.matchings = utils.all_matchings(self.n_peers)

    def entangle(self, peers):
        """Simulate how long it takes until the switch is entangled with peer(s).

        Args:
            peers (Iterable[int] | int): peer or list of peers

        Returns:
            Iterable[float] | float: future time(s) of entablement
        """
        assert (
            self.bell_states[peers] == -1
        ).all(), "Peer must not have an ongoing attempt"
        self.bell_states[peers] = self.time + 2 * self.tau[peers] * self.rng.geometric(
            self.entangle_probs[peers]
        )

        return self.bell_states[peers]

    def entangle_unentangled(self):
        """Simulate entanglement times for peers with no ongoing entanglement attempts."""
        self.entangle([peer for peer, time in enumerate(self.bell_states) if time < 0])

    @overload
    def empty_queue(self, q: int) -> bool:
        pass

    @overload
    def empty_queue(self, q: utils.Queue) -> bool:
        pass

    @overload
    def empty_queue(self, q: list[int]) -> list[bool]:
        pass

    @overload
    def empty_queue(self, q: list[utils.Queue]) -> list[bool]:
        pass

    def empty_queue(self, q):
        def _single(q):
            if not isinstance(q, utils.Queue):
                q = self.request_queues[q]
            return q.empty() or q.peek() > self.time

        if isinstance(q, Iterable):
            return list(_single(qq) for qq in q)
        return _single(q)

    @check_bell_states
    def calc_post_swap_fidelity(self, *peers):
        peers = np.array(peers)
        return constants.calculate_swapped_fidelity(
            *(-self.bell_states[peers] + self.time)
        )

    @check_bell_states
    def swap(self, *peers):
        """Perform a BSM on peer1 and peer2

        Args:_summary_
            peers (iterable[int]): peer indices
        """
        peers = np.array(peers)
        idx = self.mapping[frozenset(peers)]
        assert not self.empty_queue(idx), "No request in queue"
        assert self.request_queues[idx].peek() <= self.time, "Request is in the future"

        self.fidelities[idx].append(self.calc_post_swap_fidelity(*peers))
        self.tputs[idx] += 1
        t = self.time - self.request_queues[idx].get()
        self.latencies[idx].append(t)
        self.ages[peers[0]].append(self.time - self.bell_states[peers[0]])
        self.ages[peers[1]].append(self.time - self.bell_states[peers[1]])

        self.discard(peers)

    def discard(self, peers) -> None:
        """Discard entanglement between the switch and peer(s).

        Args:
            peers (int | Iterable[int]): peer(s) index(es)
        """
        self.bell_states[peers] = -1

    def _simulate_requests(self, idle_time):
        """Simulate how many requests enter the system during an interval.
        Samples number of events from Poisson and then samples interevent times
        via uniform.

        Args:
            idle_time (float): interval
        """
        num_events = self.rng.poisson(self.request_rates * idle_time)
        for idx, n in enumerate(num_events):
            for arrival_time in sorted(self.rng.uniform(0, idle_time, size=n)):
                self.request_queues[idx].put(arrival_time + self.time)

    def idle(self, idle_time) -> None:
        """Simulate requests in interval and increment time.

        Args:
            idle_time (float): interval

        Returns:
            float: time idled (idle_time param)
        """
        self._simulate_requests(idle_time)
        self.time += idle_time
        return idle_time

    def idle_until_next_entanglement(self):
        """Idle until the next entanglement generation succeeds.

        Returns:
            float: time idled
        """
        entanglements = [t for t in self.bell_states if t != -1 and t > self.time]
        if len(entanglements) == 0:
            return 0
        future_time = min(entanglements)
        return self.idle(future_time - self.time)

    def idle_until_all_entangled(self):
        """Idle until all entanglement generation succeeds.

        Returns:
            float: time idled
        """
        future_time = np.max(self.bell_states)
        if future_time < self.time:
            return 0
        return self.idle(future_time - self.time)

    def idle_until_next_request(self):
        """Idle for sampled time until next request.

        Returns:
            float: time idled
        """
        return self.idle(self.rng.exponential(1 / np.sum(self.request_rates)))

    def non_empty_queues(self):
        """Get list of queues with requests.

        Returns:
            frozenset[int]: set of queue indices
        """
        return frozenset(
            edge_idx
            for edge_idx, q in enumerate(self.request_queues)
            if not self.empty_queue(q)
        )

    def entangled_peers(self):
        """Get list of peers that are currently entangled with the switch.

        Returns:
            frozenset[int]: set of peer indices
        """
        return frozenset(
            peer for peer, time in enumerate(self.bell_states) if -1 < time <= self.time
        )

    def current_valid_matchings(self):
        """Get list of currently possible parallel BSMs.

        Returns:
            frozenset[frozenset[frozenset[int]]]: set of set of edges
        """
        non_empty_queues = self.non_empty_queues()
        entangled_peers = self.entangled_peers()
        valid_matchings = list(
            frozenset(
                edge
                for edge in matching
                if self.mapping[edge]
                in non_empty_queues  # there is a corresponding request waiting
                and edge
                <= entangled_peers  # peers in edge make up a subset of the entangled peers
            )
            for matching in self.matchings
        )

        # Remove empty matchings and remove duplicates, if any
        return set(matching for matching in valid_matchings if len(matching) > 0)


def curry(L, req_rates, n_peers=None, seed=None):
    """Create a function that creates a new Network object according to parameters.

    Returns:
        callable: zero-argument function
    """
    return lambda: Network(L, req_rates, n_peers=n_peers, seed=seed)
