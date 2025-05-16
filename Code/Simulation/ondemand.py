"""This is where the algorithms tested in my report are located"""

import itertools as it

import network
import numpy as np
import scipy.stats as stats
from Simulation.abstract import OnDemandSimulation, buildQoSFidelitySimulation


def calc_load(net: network.Network, EX_i):
    """Calculate server utilization (in expectation).

    Args:
        net (network.Network): Network class to extract parameters from
        EX_i (callable): function that calculates expected service time of a class for a simulation algorithm

    Returns:
        float: calculated expected load.
    """
    lambdas = net.request_rates
    p = net.entangle_probs
    tau = net.tau
    reverse_mapping = {
        idx: list(edge)
        for idx, edge in enumerate(it.combinations(range(net.n_peers), 2))
    }

    return sum(
        lambda_i * EX_i(reverse_mapping[i], tau, p)
        for i, lambda_i in enumerate(lambdas)
    )


def est_p_seq_gr_par(ps, pf, ts, tf, trials=100):
    """Estimate the probability that parallel performs better than sequential for a given class"""

    def _est(n):
        return (1 - stats.geom.cdf(n * (ts / (2 * tf)), pf)) * stats.geom.pmf(n, ps)

    return np.sum(_est(np.arange(trials) + 1))


def Seq_EX_i(peers, tau, p):
    """Expected value of class i service time in Sequential"""
    return 2 * np.sum(tau[peers] / p[peers])


def Par_EX_i(peers, tau, p):
    """Expected value of class i service time in Parallel"""
    p_1, p_2 = p[peers]
    tau_1, tau_2 = tau[peers]

    def sum_func(n):
        return (1 - stats.geom.cdf(n * (tau_2 / tau_1), p_1)) * stats.geom.pmf(n, p_2)

    return 2 * (
        (tau_1 / p_1 - tau_2 / p_2) * np.sum(sum_func(np.arange(1, 101))) + tau_2 / p_2
    )


def Smt_EX_i(peers, tau, p):
    """Expected value of class i service time in Smart"""
    p_1, p_2 = p[peers]
    tau_1, tau_2 = tau[peers]
    fast = 0 if tau_1 / p_1 <= tau_2 / p_2 else 1
    slow = (fast + 1) % 2
    p_smt = est_p_seq_gr_par(*p[[slow, fast]], *tau[[slow, fast]])
    return Par_EX_i(peers, tau, p) if p_smt > 0.5 else Seq_EX_i(peers, tau, p)


class Parallel(OnDemandSimulation):

    def _update_min(self):
        heads = list(
            (iq[0], iq[1].peek())
            for iq in ((i, q) for i, q in enumerate(self.network.request_queues))
            if not self.network.empty_queue(iq[1])
        )
        if len(heads) == 0:
            self.min = None
            return
        self.min, _ = min(heads, key=lambda x: x[1])

    def matching_decision(self):
        self._update_min()
        if self.min is not None:
            if any(
                self.network.reverse_mapping[self.min] in matching
                for matching in self.network.current_valid_matchings()
            ):
                self.network.swap(*self.network.reverse_mapping[self.min])
                self._update_min()

    def entanglement_decision(self):
        if self.min is not None:
            peers = list(self.network.reverse_mapping[self.min])
            if (self.network.bell_states[peers] == -1).all():
                self.network.entangle(peers)

    def util_exp(self):
        return calc_load(self.network, Par_EX_i)


class Smart(Parallel):
    def __init__(self, *args, threshold=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold

    def entanglement_decision(self):
        if self.min is not None:
            peers = list(self.network.reverse_mapping[self.min])
            if (self.network.bell_states[peers] == -1).all():
                slow_peer = np.argmin(self.network.entangle_probs[peers])
                fast_peer = (slow_peer + 1) % 2
                prob_parallel_better = est_p_seq_gr_par(
                    self.network.entangle_probs[peers[slow_peer]],
                    self.network.entangle_probs[peers[fast_peer]],
                    self.network.tau[peers[slow_peer]],
                    self.network.tau[peers[fast_peer]],
                )
                if prob_parallel_better > self.threshold:
                    self.network.entangle(peers)
                else:
                    self.network.entangle(peers[slow_peer])
            elif (self.network.bell_states == -1).any():
                self.network.entangle(
                    list(peer for peer in peers if self.network.bell_states[peer] == -1)
                )
        pass

    def util_exp(self):
        return calc_load(self.network, Smt_EX_i)


class Sequential(Smart):
    def entanglement_decision(self):
        if self.min is not None:
            peers = list(self.network.reverse_mapping[self.min])
            if (self.network.bell_states[peers] == -1).all():
                slow_peer = np.argmin(self.network.entangle_probs[peers])
                self.network.entangle(peers[slow_peer])
            elif (self.network.bell_states == -1).any():
                self.network.entangle(
                    list(peer for peer in peers if self.network.bell_states[peer] == -1)
                )
        pass

    def util_exp(self):
        return calc_load(self.network, Seq_EX_i)


ParallelQoS = buildQoSFidelitySimulation(Parallel)
SequentialQoS = buildQoSFidelitySimulation(Sequential)
