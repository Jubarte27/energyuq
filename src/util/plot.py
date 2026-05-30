from math import floor, ceil, sqrt
from typing import Sequence

from matplotlib.figure import SubFigure
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

from pandas import DataFrame

from .. import machines
from .data import *
machine = machines.Glados

def pad_to_even_and_split(arr: np.ndarray, value=None) -> np.ndarray:
    pad_by = arr.shape[-1] % 2
    new_shape = [*arr.shape[:-1], 2, -1]
    return np.pad(arr, (0, pad_by), mode='constant', constant_values=value).reshape(new_shape)

def mostly_square_grid(blocks: int, total_width: float, min_block_width: float):
    max_col = floor(total_width / min_block_width)
    cols = min(ceil(sqrt(blocks)), max_col)
    rows = ceil(blocks / cols)

    total_height = rows * total_width
    return (cols, rows), (total_width, total_height)

limits={
    "N_THREADS": limit(1, 28),
    "CLK": limit(0, len(machine.freq) - 1)
}
labels = np.array(list(limits.keys()), dtype=str)
values = np.array(list(limits.values()), dtype=limit)

L = (labels.size+1)//2
(C, R), grid_fig_size = mostly_square_grid(L, 6, 2)
full_rows = L // C

nd_values = pad_to_even_and_split(values)
nd_labels = pad_to_even_and_split(labels)


more_red = Line2D([0], [0], color='red', lw=2, marker="o", linestyle='')
more_blue = Line2D([0], [0], color='blue', lw=2, marker="o", linestyle='')

legend_handles = [more_red, more_blue]

def colors_for(qoi) -> dict[str, tuple[str,str,str,str]]:
    return {
        'high_low': ('#0000ff', f'lower {qoi}', '#ff0000', f'higher {qoi}'),
        'highest_lowest': ( '#0000ff00', f'lowest {qoi}', '#0000ff', f'highest {qoi}'),
    }

def key_for(result, qoi) -> str | Sequence[str]:
    if isinstance(result, EasyResult):
        return (qoi, 0) # why??
    else:
        return qoi



def plot_grid_2D(result: EasyResult):
    analysis = result.analysis
    sampler = result.sampler

    fig = plt.figure(figsize=grid_fig_size, layout="constrained")
    fig.supylabel("Configurations chosen")

    ax=[]
    i = 0
    cols = C
    index = lambda: i + 1
    for _ in range(full_rows):
        if R > full_rows:
            cols = L % C
            index = lambda: R * cols - (R * C - i)
        for _ in range(cols):
            xd = nd_values[0, i].upper - nd_values[0, i].lower
            yd = nd_values[1, i].upper - nd_values[1, i].lower
            ax.append(fig.add_subplot(R, cols, index(),
                                    xlim=[nd_values[0, i].lower - xd/10, nd_values[0, i].upper + xd/10],
                                    ylim=[nd_values[1, i].lower - yd/10, nd_values[1, i].upper + yd/10], 
                                    xlabel=nd_labels[0, i], ylabel=nd_labels[1, i])
                    )
            i += 1

    accepted_grid = pad_to_even_and_split(sampler.generate_grid(analysis.l_norm).astype(object))
    ic=0
    for i in range(L):
        ax[i].plot(accepted_grid[:,0, ic], accepted_grid[:,1,ic], 'o', alpha=0.25)
        ic += 1
    # plt.tight_layout()
    return fig


def plot_grid_2D_best(result: Result, qoi=None, order_focus=False, subfig=None):
    if qoi is None:
        qoi = result.qois[0]
    key = key_for(result, qoi)
    df = result.df

    pretty_colors = colors_for(qoi)

    if subfig is None:
        fig = plt.figure(figsize=grid_fig_size, layout="constrained")
        fig.supylabel(f"Configurations evaluated by {qoi}")
    else:
        fig = subfig
    
    ax=[]
    i = 0
    cols = C
    index = lambda: i + 1
    ax: list[Axes]=[]
    for _ in range(full_rows):
        if R > full_rows:
            cols = L % C
            index = lambda: R * cols - (R * C - i)
        for _ in range(cols):
            xd = nd_values[0, i].upper - nd_values[0, i].lower
            yd = nd_values[1, i].upper - nd_values[1, i].lower
            ax.append(fig.add_subplot(R, cols, index(),
                                    xlim=[nd_values[0, i].lower - xd/10, nd_values[0, i].upper + xd/10],
                                    ylim=[nd_values[1, i].lower - yd/10, nd_values[1, i].upper + yd/10], 
                                    xlabel=nd_labels[0, i], ylabel=nd_labels[1, i])
                    )
            ax[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
            ax[-1].yaxis.set_major_locator(MaxNLocator(integer=True))
            ax[-1].set_box_aspect(1)
            ax[-1].set_anchor('N')
            i += 1

    dataframe: DataFrame = df.sort_values(by=key, ascending=False)
    dataframe[f"{qoi}_norm"] = ((column := dataframe[qoi]) - column.min()) / (column.max() - column.min())
    s = dataframe[f"{qoi}_norm"].size

    if order_focus:
        cmap = mcolors.LinearSegmentedColormap.from_list("ba", list(pretty_colors['highest_lowest'][::2]))
        legend_labels = ["Lower ranked", "Higher ranked"]
        colors = [ cmap(i/s) for i, _ in enumerate(dataframe[f"{qoi}_norm"].to_numpy())]
    else:
        legend_labels = [f"Higher {qoi}", f"Lesser {qoi}"]
        cmap = mcolors.LinearSegmentedColormap.from_list("ba", list(pretty_colors['high_low'][::2]))
        colors = [ cmap(y) for y in dataframe[f"{qoi}_norm"].to_numpy()]
    
    

    for i in range(L):
        xs = dataframe[labels[i*2]].to_numpy()
        ys = dataframe[labels[i*2 + 1]].to_numpy()
        ax[i].legend(handles=legend_handles, labels=legend_labels, draggable=True, fontsize='x-small', ncols=2, bbox_to_anchor=(1, 1.1), loc='upper right')
        ax[i].scatter(xs, ys, c=colors)
    
    # plt.tight_layout()
    return fig


def plot_sobols1(result: EasyResult, qoi = None):
    results = result.results
    if qoi is None:
        qoi = result.qois[0]
    d = len(labels)
    
    fig = plt.figure(layout="constrained")
    ax = fig.add_subplot(title=r'First-order Sobol indices', ylim=[0,1])
    ax.set_ylabel(r'$S_i$', fontsize=14)
    
    sobols_first = np.array(list(results.sobols_first(qoi).values()))
    ax.bar(0, np.sum(sobols_first), color='salmon')
    ax.bar(np.arange(1, d+1), sobols_first.flatten(), color='dodgerblue')

    ax.set_xticks(np.arange(d+1))
    ax.set_xticklabels(['Total first order', *labels], rotation=90)
    # plt.tight_layout()
    return fig


def draw_gradients(*color_list):
    n_items = len(color_list)
    
    fig, ax = plt.subplots(figsize=(6, 0.9 * n_items + 0.4))
    
    ax.set_facecolor('#ffffff')
    for spine in ax.spines.values():
        spine.set_color('#cccccc')
        spine.set_linewidth(1.0)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.2, n_items)

    gradient_base = np.linspace(0, 1, 256).reshape(1, -1)

    for i, (color_a, label_a, color_b, label_b) in enumerate(reversed(color_list)):
        y_bottom = i * 1.0
        y_top = y_bottom + 0.28
    
        cmap = mcolors.LinearSegmentedColormap.from_list(f"legend_cmap_{i}", [color_a, color_b])
        
        ax.imshow(gradient_base, aspect='auto', cmap=cmap, extent=(0, 1, y_bottom, y_top))
        ax.text(0.0, y_top + 0.05, label_a, ha='left', va='bottom', fontsize=10, color='#333333')
        ax.text(1.0, y_top + 0.05, label_b, ha='right', va='bottom', fontsize=10, color='#333333')
        
    # plt.tight_layout()
    return fig

def plot_sorted(result: Result, qoi=None, subfig: SubFigure | None=None):
    if qoi is None:
        qoi = result.qois[0]
    key = key_for(result, qoi)
    dataframe = result.df
    
    if not subfig:
        fig, ax = plt.subplots(figsize=(12,12), layout="constrained")
    else:
        fig = subfig
        ax = fig.subplots()

    sorted_dataframe: DataFrame = dataframe.sort_values(by=key, ascending=False)

    ax.semilogy(sorted_dataframe[key].to_numpy())
    ax.set_box_aspect(1)
    ax.set_anchor('N')
    if not subfig:
        ax.set_title(f"{qoi} sorted")
    # plt.tight_layout()
    return fig


def get_confidence_intervals(samples, conf=0.9):
    """
    Compute the confidence intervals given an array of samples

    Parameters
    ----------
    samples : array
        Samples on which to compute the intervals.
    conf : float, optional, must be in [0, 1].
        The confidence interval percentage. The default is 0.9.

    Returns
    -------
    lower : array
        The lower confidence bound..
    upper : array
        The upper confidence bound.

    """

    # ake sure conf is in [0, 1]
    if conf < 0.0 or conf > 1.0:
        print('conf must be specified within [0, 1]')
        return

    # lower bound = alpha, upper bound = 1 - alpha
    alpha = 0.5 * (1.0 - conf)

    # arrays for lower and upper bound of the interval
    n_samples = samples.shape[0]
    N_qoi = samples.shape[1]
    lower = np.zeros(N_qoi)
    upper = np.zeros(N_qoi)

    # the probabilities of the ecdf
    prob = np.linspace(0, 1, n_samples)
    # the closest locations in prob that correspond to the interval bounds
    idx0 = np.where(prob <= alpha)[0][-1]
    idx1 = np.where(prob <= 1.0 - alpha)[0][-1]

    # for every location of qoi compute the ecdf-based confidence interval
    for i in range(N_qoi):
        # the sorted surrogate samples at the current location
        samples_sorted = np.sort(samples[:, i])
        # the corresponding confidence interval
        lower[i] = samples_sorted[idx0]
        upper[i] = samples_sorted[idx1]

    return lower, upper

def plot_2D_single_dimension(result: Result, qoi=None, subfig=None):
    if qoi is None:
        qoi = result.qois[0]
    key = key_for(result, qoi)
    df = result.df

    L = len(labels)
    C = 2
    R = int(np.ceil(L / C))

    if subfig:
        fig = subfig
    else:
        fig = plt.figure(figsize=(12,12/C*R), layout="constrained")
        fig.supylabel(f"For {qoi}")

    ax: list[Axes]=[]
    i=0
    
    ax = fig.subplots(R, C, sharey=False)

    for i in range(L):
        xd = values[i].upper - values[i].lower
        ax[i].set_xlim((values[i].lower - xd/10, values[i].upper + xd/10))
        ax[i].set_xlabel(xlabel=labels[i])
        ax[i].set_ylabel(qoi)
        ax[i].xaxis.set_major_locator(MaxNLocator(integer=values[i]))
        ax[i].yaxis.set_major_locator(MaxNLocator(integer=True))
        ax[i].set_box_aspect(1)
        ax[i].set_anchor('N')

    for i in range(L):
        ax[i].scatter(df[labels[i]], df[key])
    
    # plt.tight_layout()
    return fig

def plot_boxplot(result: Result, qoi=None):
    if qoi is None:
        qoi = result.qois[0]
    key = key_for(result, qoi)
    df = result.df

    L = len(labels)
    C = int(np.ceil(np.sqrt((10+1)//2)))
    R = int(np.ceil(L / C))

    fig = plt.figure(figsize=(12,12/C*R), layout="constrained")
    fig.supylabel(f"For {qoi}")
    
    ax: list[Axes]=[]
    for i in range(L):
        xd = values[i].upper - values[i].lower
        ax.append(fig.add_subplot(R, C, i+1,
                                  xlim=[values[i].lower - xd/10, values[i].upper + xd/10],
                                  xlabel=labels[i], ylabel=qoi
                    )
                 )
        ax[-1].xaxis.set_major_locator(MaxNLocator(integer=values[i]))
        ax[-1].yaxis.set_major_locator(MaxNLocator(integer=True))

    def col_to_numpy(x: DataFrame):
        return x[key].to_numpy()

    for i in range(L):
        label_c = (labels[i], 0) if isinstance(key, tuple) else labels[i]
        box_frame = df[[key, label_c]].groupby(by=label_c)[[key]].apply(col_to_numpy) # pyright: ignore[reportArgumentType, reportCallIssue]

        VP = ax[i].boxplot(box_frame.to_numpy(),
                            positions=box_frame.index.to_numpy(),
                            patch_artist=True,
                            showmeans=False, showfliers=False, manage_ticks = False,
                            medianprops={"color": "white", "linewidth": 0.5},
                            boxprops={"facecolor": "C0", "edgecolor": "white", "linewidth": 0.5},
                            whiskerprops={"color": "C0", "linewidth": 1.5},
                            capprops={"color": "C0", "linewidth": 1.5}
                        )

    # plt.tight_layout()
    return fig
