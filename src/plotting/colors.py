"""
Color utilities for EnergyUQ plotting.

Provides functions to generate visually distinct color palettes for plots,
categorical data, and stacked sensitivity indices.
"""

from __future__ import annotations

import colorsys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def _rgb_to_oklab(r: float, g: float, b: float) -> np.ndarray:
    """
    Convert sRGB [0, 1] to Oklab color space (L, a, b).
    
    Oklab is a perceptually uniform color space designed to closely mimic
    human visual difference perception.
    """
    def _to_linear(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    lr, lg, lb = _to_linear(r), _to_linear(g), _to_linear(b)
    l_ = (0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb) ** (1.0 / 3.0)
    m_ = (0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb) ** (1.0 / 3.0)
    s_ = (0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb) ** (1.0 / 3.0)

    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return np.array([L, a, b], dtype=float)


def get_distinct_colors(
    n: int,
    as_hex: bool = True,
    palette: str | None = None,
    method: str = "auto",
) -> list[str] | list[tuple[float, float, float]]:
    """
    Generate n visually distinct colors.

    Parameters
    ----------
    n : int
        Number of visually distinct colors to produce.
    as_hex : bool, default True
        If True, return colors as '#rrggbb' hex strings.
        If False, return colors as (r, g, b) float tuples in [0, 1].
    palette : str or None, default None
        Optional matplotlib colormap name to sample from (e.g. 'tab10', 'Set2').
        If None, an optimal set of visually distinct colors is generated.
    method : str, default 'auto'
        Method to generate distinct colors when palette is None:
        - 'auto': Uses Tableau 10 for n <= 10, interleaved Tableau 20 for n <= 20,
                  and greedy furthest-point sampling in Oklab space for n > 20.
        - 'oklab': Uses greedy furthest-point sampling in Oklab perceptual space.

    Returns
    -------
    list[str] | list[tuple[float, float, float]]
        List of n distinct colors.
    """
    if n <= 0:
        return []

    # Custom palette request
    if palette is not None:
        cmap = plt.get_cmap(palette)
        if hasattr(cmap, "colors") and len(cmap.colors) >= n:
            raw_colors = [mcolors.to_rgb(c) for c in cmap.colors[:n]]
        else:
            # Sample evenly across the colormap
            positions = np.linspace(0.0, 1.0, n, endpoint=(n > 1))
            raw_colors = [cmap(p)[:3] for p in positions]

        if as_hex:
            return [mcolors.to_hex(c) for c in raw_colors]
        return raw_colors

    # Method: auto
    if method == "auto":
        # For n <= 10: tab10 is standard and distinct
        tab10_rgb = [mcolors.to_rgb(c) for c in plt.get_cmap("tab10").colors]
        if n <= 10:
            selected_rgb = tab10_rgb[:n]
            return [mcolors.to_hex(c) for c in selected_rgb] if as_hex else selected_rgb

        # For 10 < n <= 20: tab20 reordered (darks first, then lights)
        tab20_rgb = [mcolors.to_rgb(c) for c in plt.get_cmap("tab20").colors]
        reordered_20 = [tab20_rgb[2 * i] for i in range(10)] + [tab20_rgb[2 * i + 1] for i in range(10)]
        if n <= 20:
            selected_rgb = reordered_20[:n]
            return [mcolors.to_hex(c) for c in selected_rgb] if as_hex else selected_rgb

        # Seed with reordered 20 and continue with greedy furthest-point in Oklab
        seed_rgb = reordered_20
    else:
        seed_rgb = [mcolors.to_rgb(plt.get_cmap("tab10").colors[0])]

    # Greedy furthest-point sampling in Oklab space
    selected_rgb = list(seed_rgb)
    if len(selected_rgb) >= n:
        selected_rgb = selected_rgb[:n]
        return [mcolors.to_hex(c) for c in selected_rgb] if as_hex else selected_rgb

    selected_oklab = [_rgb_to_oklab(*c) for c in selected_rgb]

    # Generate candidate pool in HLS space (excluding extreme lightness/darkness)
    num_hues = max(72, n * 3)
    hues = np.linspace(0, 1, num_hues, endpoint=False)
    lightnesses = [0.35, 0.48, 0.62, 0.76]
    saturations = [0.65, 0.85, 0.98]

    candidates: list[tuple[float, float, float]] = []
    cand_oklab_list: list[np.ndarray] = []
    for h in hues:
        for l in lightnesses:
            for s in saturations:
                rgb = colorsys.hls_to_rgb(h, l, s)
                candidates.append(rgb)
                cand_oklab_list.append(_rgb_to_oklab(*rgb))

    cand_oklab = np.array(cand_oklab_list)

    # Initial minimum distance to all already selected colors
    min_dists = np.min(
        [np.linalg.norm(cand_oklab - s_ok, axis=1) for s_ok in selected_oklab],
        axis=0,
    )

    needed = n - len(selected_rgb)
    for _ in range(needed):
        best_idx = int(np.argmax(min_dists))
        best_rgb = candidates[best_idx]
        best_ok = cand_oklab[best_idx]

        selected_rgb.append(best_rgb)
        new_dists = np.linalg.norm(cand_oklab - best_ok, axis=1)
        min_dists = np.minimum(min_dists, new_dists)
        min_dists[best_idx] = -1.0  # Prevent re-selection

    if as_hex:
        return [mcolors.to_hex(c) for c in selected_rgb]
    return selected_rgb


# Convenient aliases
distinct_colors = get_distinct_colors
get_n_distinct_colors = get_distinct_colors
n_distinct_colors = get_distinct_colors

__all__ = [
    "distinct_colors",
    "get_distinct_colors",
    "get_n_distinct_colors",
    "n_distinct_colors",
]

