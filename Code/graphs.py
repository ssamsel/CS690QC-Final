"""General graphing methods for creating plots.
Not relevant to understanding the simulation so is not documented.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from Simulation.abstract import Simulation


def _simulation_bar_stats(data, labels, ax: Axes, width=0.25, padding=3):
    x = np.arange(len(data))
    ax.bar_label(ax.bar(x, data[:, 0], width, label="Mean"), padding=padding)
    ax.bar_label(
        ax.bar(x + width, data[:, 1], width, label="Variance"), padding=padding
    )
    ax.set_xticks(x + width / 2, labels, fontsize="small")
    ax.legend(loc="upper left")


def _simulation_x_y(x_vals: list[float], y_vals, stat_name: str, ax: Axes, names):
    groups = []
    for i in range(0, len(y_vals), len(x_vals)):
        groups.append(y_vals[i : i + len(x_vals)])
    for g, name in zip(groups, names):
        ax.plot(x_vals, g, label=name)
    ax.set_xlabel("Arrival Rate")
    ax.set_ylabel(stat_name)
    ax.legend(loc="upper left")


def plot_simulation_means(
    simulations: list[Simulation], width=0.17, padding=3, title=None
):
    data = [sim.parse_results() for sim in simulations]
    simwise_data = {
        data_type: np.array([sim_results[data_type] for sim_results in data])
        for data_type in ("tput", "rel_tput", "fidelity", "latency")
    }
    factor = np.ceil(np.log10(np.max(simwise_data["latency"][:, 0])))
    simwise_data[f"latency E-{factor}"] = simwise_data.pop("latency")[:, 0] / 10**factor
    factor = np.ceil(np.log10(np.max(simwise_data["tput"])))
    simwise_data[f"tput E-{factor}"] = simwise_data.pop("tput") / 10**factor
    simwise_data["fidelity"] = simwise_data["fidelity"][:, 0]
    x = np.arange(len(data))
    _, ax = plt.subplots(1, 1)
    for index, data_type, values in (
        (i, *kv) for i, kv in enumerate(simwise_data.items())
    ):
        ax.bar_label(
            ax.bar(x + index * width, values, width, label=data_type), padding=padding
        )
    ax.set_xticks(x + 1.5 * width, [sim.name for sim in simulations], fontsize="small")
    ax.set_ylim(top=1)
    ax.legend(loc="upper left")
    plt.setp(ax.get_xticklabels(), rotation=20, horizontalalignment="right")
    if title:
        plt.title(title, y=1.08)
    plt.tight_layout()
    plt.show()
    plt.clf()
    plt.close()


def _plot_simulation_edgewise_means(
    simulation: Simulation, title, ax: Axes, width=0.17, padding=3
):
    data = simulation.parse_results()
    edgewise_data = {
        data_type: data[data_type]
        for data_type in ("edge_tput", "edge_rel_tput", "edge_fidelity", "edge_latency")
    }

    factor = np.ceil(np.log10(np.max(edgewise_data["edge_latency"][:, 0])))
    edgewise_data[f"edge_latency E-{factor}"] = (
        edgewise_data.pop("edge_latency")[:, 0] / 10**factor
    )
    factor = np.ceil(np.log10(np.max(edgewise_data["edge_tput"])))
    edgewise_data[f"edge_tput E-{factor}"] = edgewise_data.pop("edge_tput") / 10**factor
    edgewise_data["edge_fidelity"] = edgewise_data["edge_fidelity"][:, 0]
    x = np.arange(len(simulation.reverse_mapping))
    for index, data_type, values in (
        (i, *kv) for i, kv in enumerate(edgewise_data.items())
    ):
        ax.bar_label(
            ax.bar(x + index * width, values, width, label=data_type), padding=padding
        )
    ax.set_title(title)
    ax.set_xticks(x + 1.5 * width, simulation.reverse_mapping, fontsize="small")
    ax.set_ylim(top=1)
    ax.legend(loc="upper left")


def plot_simulation_edgewise_means(simulations: list[Simulation], title=None):
    _, axes = plt.subplots(int(np.ceil(len(simulations) / 2)), 2)
    for ax, sim in zip(axes.flat[: len(simulations)], simulations):
        _plot_simulation_edgewise_means(sim, sim.name, ax)
    plt.tight_layout()
    if title:
        plt.title(title)
    plt.show()
    plt.clf()
    plt.close()


def plot_fidelity_stats(simulations: list[Simulation], title=None):
    _, ax = plt.subplots(1, 1)
    fidelity_data = np.array([sim.parse_results()["fidelity"] for sim in simulations])
    _simulation_bar_stats(
        fidelity_data,
        [sim.name for sim in simulations],
        ax,
    )
    ax.set_ylim(top=1)
    ax.set_title("Fidelity Mean and Variance Across Simulations")
    plt.setp(ax.get_xticklabels(), rotation=20, horizontalalignment="right")
    if title:
        plt.title(title)
    plt.show()
    plt.clf()
    plt.close()


def plot_latency_stats(simulations: list[Simulation], title=None):
    _, ax = plt.subplots(1, 1)
    latency_data = np.array([sim.parse_results()["latency"] for sim in simulations])
    _simulation_bar_stats(latency_data, [sim.name for sim in simulations], ax)
    plt.setp(ax.get_xticklabels(), rotation=20, horizontalalignment="right")
    ax.set_title("Latency Mean and Variance Across Simulations" if not title else title)
    plt.show()
    plt.clf()
    plt.close()


def _plot_2d_distribution(data1, data2, label1, label2, title, ax: Axes, bins=[40, 40]):
    combined = [list(), list()]
    for x, y in zip(data1, data2):
        combined[0].extend(x)
        combined[1].extend(y)
    a = ax.hist2d(combined[0], combined[1], cmap=plt.cm.jet, bins=bins)
    ax.set_xlabel(label1)
    ax.set_ylabel(label2)
    ax.set_title(title)
    plt.colorbar(a[3], ax=ax)


def plot_latency_vs_fidelity(simulations: list[Simulation], title=None):
    _, axes = plt.subplots(int(np.ceil(len(simulations) / 2)), 2)
    for ax, sim in zip(axes.flat[: len(simulations)], simulations):
        _plot_2d_distribution(
            sim.network.fidelities,
            sim.network.latencies,
            "fidelity",
            "latency",
            sim.name,
            ax,
        )
    plt.show()
    if title:
        plt.title(title)
    plt.clf()
    plt.close()


def plot_lines(simulations: list[Simulation], x_vals, names):
    _, axes = plt.subplots(2, 2)
    data = [sim.parse_results() for sim in simulations]
    for ax, stat_name in zip(axes.flat, ("fidelity", "latency", "tput", "rel_tput")):
        _simulation_x_y(
            x_vals,
            np.array(
                [
                    (
                        sim_data[stat_name][0]
                        if stat_name in ("fidelity", "latency")
                        else sim_data[stat_name]
                    )
                    for sim_data in data
                ],
            ),
            stat_name,
            ax,
            names,
        )
    plt.show()
    plt.clf()
    plt.close()
