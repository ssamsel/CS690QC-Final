"""These are versions of switches with 1 communication qubit per peer.
Not tested in report.
"""

import itertools as it

import numpy as np
import utils
from Simulation.abstract import BasicSimulation, Simulation, buildQoSFidelitySimulation


class RandomMatchSimulation(BasicSimulation):
    """Make a random set of valid BSMs"""

    def matching_decision(self):
        matchings = self.network.current_valid_matchings()
        if len(matchings) == 0:
            return
        for edge in np.random.choice(list(matchings)):
            self.network.swap(*edge)


class YoungestEPRFirstSimulation(BasicSimulation):
    """Greedily choose the matching set containing the youngest entangled qubit."""

    def matching_decision(self):
        matchings = self.network.current_valid_matchings()
        if len(matchings) == 0:
            return
        self.network.swap(
            *max(
                set(utils.flatten2d(matchings)),
                key=lambda edge: self.network.calc_post_swap_fidelity(*edge),
            )
        )
        return self.matching_decision()


class BlindEqualTputSimulation(BasicSimulation):
    """Try to balance throughput equally to each peer without accounting for request rates."""

    def matching_decision(self):
        matchings = self.network.current_valid_matchings()
        if len(matchings) == 0:
            return
        for idx, _ in sorted(
            ((i, t) for i, t in enumerate(self.network.tputs)), key=lambda x: x[1]
        ):
            tmp = [
                matching
                for matching in matchings
                if any(self.network.reverse_mapping[idx] == edge for edge in matching)
            ]
            if len(tmp) > 0:
                matchings = tmp
        for edge in matchings[0]:
            self.network.swap(*edge)

    def idle_decision(self):
        return self.network.idle_until_all_entangled()


class RoundRobinSimulation(Simulation):
    """Just like the name says :)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        order = list(self.network.mapping.items())
        self.network.rng.shuffle(order)
        self.iter = it.cycle(order)
        self.current = next(self.iter)
        self.next = next(self.iter)

    def matching_decision(self):
        if self.network.empty_queue(self.current[1]):
            return
        if any(
            t > self.network.time or t < 0
            for t in self.network.bell_states[[*self.current[0]]]
        ):
            return
        self.network.swap(*self.current[0])
        self.current = self.next
        self.next = next(self.iter)
        self.matching_decision()

    def entanglement_decision(self):
        l_current = list(self.current[0])
        l_next = list(self.next[0])
        if self.network.bell_states[l_current[0]] == -1:
            self.network.entangle(l_current[0])
        if self.network.bell_states[l_current[1]] == -1:
            self.network.entangle(l_current[1])
        if self.network.bell_states[l_next[0]] == -1:
            self.network.entangle(l_next[0])
        if self.network.bell_states[l_next[1]] == -1:
            self.network.entangle(l_next[1])

    def idle_decision(self):
        if not self.network.empty_queue(self.current[1]):
            return self.network.idle_until_next_entanglement()
        return self.network.idle_until_next_request()


RandomMatchSimulationQoS = buildQoSFidelitySimulation(RandomMatchSimulation)
YoungestEPRFirstSimulationQoS = buildQoSFidelitySimulation(YoungestEPRFirstSimulation)
BlindEqualTputSimulationQoS = buildQoSFidelitySimulation(BlindEqualTputSimulation)
RoundRobinSimulationQoS = buildQoSFidelitySimulation(RoundRobinSimulation)
