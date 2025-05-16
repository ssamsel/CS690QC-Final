"""This file contains methods and constants that do not change across any switch algorithm nor network topology."""

import numpy as np
import utils

Phi_p = np.array(
    [[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]],
)

Gamma = 0.2  # 1/T_1 in msec
L_att = 22  # km
c = 2 * 10**8  # m/s


@utils.wrap_to_np
def tau(L):
    """Calculate one-way propagation delay(s)

    Args:
        L (Iterable[float]|float): Link length(s) in km

    Returns:
        Iterable[float] | float: propagation delay(s)
    """
    return (L * 1000) / c


@utils.wrap_to_np
def eta(L):
    """Calculate link transmissivity(s)

    Args:
        L (Iterable[float] | float): Link length(s) in km

    Returns:
        Iterable[float] | float: link transmissivity(s)
    """
    return np.exp(-(L / 2) / L_att)


@utils.wrap_to_np
def calc_fidelity_to_phi_p(rho):
    """Calculate fidelity between Phi+ and rho

    Args:
        rho (NDarray): density matrix

    Returns:
        float: fidelity (no sqrt)
    """
    return (rho[0, 0] + rho[3, 0] + rho[0, 3] + rho[3, 3]) / 2


@utils.wrap_to_np
def calc_epr_prob_bk(L):
    """Calculate success probability of Barret-Kok entanglement generation.

    Args:
        L (Iterable[float] | float): Link length(s) in km

    Returns:
        Iterable[float] | float: success probability(s)
    """
    return eta(L) ** 2 / 2


@utils.wrap_to_np
def calc_epr_prob_sc(L, alpha):
    """Calculate success probability of Single-Click entanglement generation.

    Args:
        L (Iterable[float] | float): Link length(s) in km
        L (Iterable[float] | float): alpha parameter of single-click

    Returns:
        Iterable[float] | float: success probability(s)
    """
    return 2 * eta(L) * alpha


@utils.wrap_to_np
def calc_rho(w):
    """Calculate werner state density matrix.

    Args:
        w (Iterable[float] | float): werner parameter/probability

    Returns:
        NDArray: density matrix
    """
    return w * Phi_p + (1 - w) * np.identity(4) / 4


def calculate_swapped_fidelity(*deltas):
    """Calculate the fidelity of the final bell state after a BSM.

    Args:
        deltas (iterable[float]): ages of peer bell states

    Returns:
        float: fidelity of swapped state
    """
    deltas = np.array(deltas)
    assert (deltas >= 0).all()
    w = np.prod(np.exp(-Gamma * deltas * 1000))
    return w + (1 - w) / 4
