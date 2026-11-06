"""Static figures: the word grid and the SOM heatmaps.

Every heatmap uses the same grid orientation as the word grid (row 0 at the
top), so a region can be found in every figure at the same place.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib.patches  # noqa: E402,F401
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8985"
SURFACE = "#fcfcfb"
GRID_LINE = "#e4e3df"

SEQ = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
     "#184f95", "#0d366b"],
)
GREY = LinearSegmentedColormap.from_list(
    "seq_grey", ["#fcfcfb", "#e9e8e4", "#c9c8c2", "#9a9993", "#5f5e5a", "#2d2d2b"]
)
DIV = LinearSegmentedColormap.from_list(
    "div_blue_red",
    ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f2a7a6", "#e34948", "#a32a2a"],
)
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a"]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "text.color": INK,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK_2,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }
)


def _save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _frame(ax, rows, cols):
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def region_boundaries(ax, regions, color=INK, lw=1.6):
    """Draw a line wherever two neighbouring cells belong to different regions."""
    rows, cols = regions.shape
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols and regions[r, c] != regions[r, c + 1]:
                ax.plot([c + 0.5, c + 0.5], [r - 0.5, r + 0.5], color=color, lw=lw,
                        solid_capstyle="round")
            if r + 1 < rows and regions[r, c] != regions[r + 1, c]:
                ax.plot([c - 0.5, c + 0.5], [r + 0.5, r + 0.5], color=color, lw=lw,
                        solid_capstyle="round")


def heatmap(ax, values, cmap=SEQ, norm=None, regions=None, title=None,
            cbar_label=None, fig=None, vmin=None, vmax=None):
    rows, cols = values.shape
    masked = np.ma.masked_invalid(values)
    cmap = cmap.copy()
    cmap.set_bad("#ffffff")
    im = ax.imshow(masked, cmap=cmap, norm=norm, vmin=None if norm else vmin,
                   vmax=None if norm else vmax, interpolation="nearest")
    _frame(ax, rows, cols)
    # Cells without enough data are hatched, so they never read as "average".
    for r, c in zip(*np.nonzero(~np.isfinite(values))):
        ax.add_patch(matplotlib.patches.Rectangle(
            (c - 0.5, r - 0.5), 1, 1, fill=False, hatch="////", lw=0,
            edgecolor="#c9c8c2"))
    if regions is not None:
        region_boundaries(ax, regions, color=INK, lw=0.6)
    if title:
        ax.set_title(title, loc="left", fontsize=10, color=INK, pad=6)
    if fig is not None:
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.015)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7, color=MUTED)
        if cbar_label:
            cb.set_label(cbar_label, fontsize=8, color=INK_2)
    return im


def word_grid(path, cells, rows, cols, umatrix, regions, title, max_words=6,
              cell_inches=0.95):
    """The "galaxy of concepts": one square per prototype, its words inside.

    ``cells`` maps (row, col) to a list of labels sorted by frequency. The
    cell shading is the U-matrix (darker = the prototype is far from its
    neighbours, i.e. a border zone), thick lines separate the k-means regions,
    and the small grey ``nXeY`` tag follows the book's notation: node X with
    Y elements.
    """
    fig, ax = plt.subplots(figsize=(cols * cell_inches, rows * cell_inches * 0.8))
    norm_u = (umatrix - umatrix.min()) / (np.ptp(umatrix) or 1)
    ax.imshow(norm_u * 0.45, cmap=GREY, vmin=0, vmax=1, interpolation="nearest",
              extent=(-0.5, cols - 0.5, rows - 0.5, -0.5), aspect="auto")
    for r in range(rows + 1):
        ax.axhline(r - 0.5, color=GRID_LINE, lw=0.5)
    for c in range(cols + 1):
        ax.axvline(c - 0.5, color=GRID_LINE, lw=0.5)
    region_boundaries(ax, regions, color=INK, lw=1.8)
    for r in range(rows):
        for c in range(cols):
            words = cells.get((r, c), [])
            node = r * cols + c + 1
            ax.text(c - 0.45, r - 0.42, f"n{node}e{len(words)}", fontsize=3.2,
                    color=MUTED, va="top", ha="left")
            if not words:
                continue
            shown = words[:max_words]
            more = len(words) - len(shown)
            text = "\n".join(shown) + (f"\n+{more}" if more else "")
            ax.text(c, r + 0.06, text, fontsize=4.6, color=INK, va="center",
                    ha="center", linespacing=1.05)
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="left", fontsize=14, color=INK, pad=10)
    _save(fig, path)


def small_multiples(path, maps, titles, suptitle, cmap=DIV, symmetric=True,
                    regions=None, cbar_label=None, ncols=3):
    n = len(maps)
    nrows = int(np.ceil(n / ncols))
    rows, cols = maps[0].shape
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * cols * 0.16, nrows * rows * 0.16 + 0.6),
                             squeeze=False)
    finite = np.concatenate([m[np.isfinite(m)] for m in maps])
    if symmetric:
        lim = np.nanpercentile(np.abs(finite), 98) or 1
        norm = TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim)
    else:
        norm = None
    vmin, vmax = (None, None) if symmetric else (np.nanmin(finite),
                                                np.nanpercentile(finite, 99))
    im = None
    for ax, values, t in zip(axes.ravel(), maps, titles):
        im = heatmap(ax, values, cmap=cmap, norm=norm, regions=regions, title=t,
                     vmin=vmin, vmax=vmax)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7)
    if cbar_label:
        cb.set_label(cbar_label, fontsize=8, color=INK_2)
    fig.suptitle(suptitle, x=0.01, ha="left", fontsize=12, color=INK)
    _save(fig, path)


def single(path, values, title, cbar_label, cmap=SEQ, regions=None, norm=None,
           labels=None):
    rows, cols = values.shape
    fig, ax = plt.subplots(figsize=(cols * 0.28 + 1.2, rows * 0.28 + 0.6))
    heatmap(ax, values, cmap=cmap, norm=norm, regions=regions, title=title,
            cbar_label=cbar_label, fig=fig)
    if labels:
        for (r, c), text in labels:
            ax.text(c, r, text, fontsize=5.5, ha="center", va="center", color=INK,
                    bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none",
                              alpha=0.85))
    _save(fig, path)


def category_map(path, codes, names, title, regions=None, share=None):
    """Majority category per cell (torchsom's classification map).

    Lighter cells are less pure: the alpha is the majority's share.
    """
    rows, cols = codes.shape
    fig, ax = plt.subplots(figsize=(cols * 0.28 + 2.2, rows * 0.28 + 0.6))
    rgba = np.ones((rows, cols, 4))
    for k, name in enumerate(names):
        rgb = matplotlib.colors.to_rgb(CATEGORICAL[k])
        mask = codes == k
        rgba[mask, :3] = rgb
    if share is not None:
        alpha = np.clip(share, 0, 1)
        rgba[..., :3] = rgba[..., :3] * alpha[..., None] + (1 - alpha[..., None])
    rgba[codes < 0] = (1, 1, 1, 1)
    ax.imshow(rgba, interpolation="nearest")
    _frame(ax, rows, cols)
    if regions is not None:
        region_boundaries(ax, regions, color=INK, lw=0.6)
    handles = [matplotlib.patches.Patch(color=CATEGORICAL[k], label=n)
               for k, n in enumerate(names)]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1),
              frameon=False, fontsize=8)
    ax.set_title(title, loc="left", fontsize=10, pad=6)
    _save(fig, path)


def training_curves(path, qe, te, title):
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.6))
    for ax, values, label in ((axes[0], qe, "Quantisation error"),
                              (axes[1], te, "Topographic error (%)")):
        ax.plot(np.arange(1, len(values) + 1), values, color=CATEGORICAL[0], lw=2)
        ax.set_title(label, loc="left", fontsize=9)
        ax.set_xlabel("Epoch")
        ax.grid(axis="y", color=GRID_LINE, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    _save(fig, path)
