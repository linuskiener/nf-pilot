"""Train a SOM with torchsom and collect the arrays the plots need."""

from dataclasses import dataclass, field

import numpy as np
import torch
from torchsom import SOM


@dataclass
class TrainedMap:
    rows: int
    cols: int
    weights: np.ndarray  # rows x cols x dims
    bmu: np.ndarray  # n_samples x 2 (row, col)
    hits: np.ndarray  # rows x cols
    umatrix: np.ndarray  # rows x cols, mean distance to the 8 neighbours
    qe: list = field(default_factory=list)  # quantisation error per epoch
    te: list = field(default_factory=list)  # topographic error (%) per epoch
    clusters: np.ndarray | None = None  # rows x cols region id

    @property
    def node(self):
        """Flat node index of every sample's best-matching unit."""
        return self.bmu[:, 0] * self.cols + self.bmu[:, 1]


def umatrix(weights):
    """Mean Euclidean distance of each prototype to its 8 grid neighbours."""
    rows, cols, _ = weights.shape
    out = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            d = [
                np.linalg.norm(weights[r, c] - weights[rr, cc])
                for rr in range(max(0, r - 1), min(rows, r + 2))
                for cc in range(max(0, c - 1), min(cols, c + 2))
                if (rr, cc) != (r, c)
            ]
            out[r, c] = np.mean(d)
    return out


def train_som(
    vectors,
    rows,
    cols,
    epochs=60,
    batch_size=64,
    learning_rate=0.5,
    sigma=None,
    seed=42,
    n_regions=None,
):
    """Fit a rectangular SOM with PCA initialisation and Gaussian neighbourhood.

    ``sigma`` defaults to a quarter of the longer grid side (as in Bungum &
    Gambäck) and decays towards 1, so the map first unfolds globally and then
    fine-tunes locally. ``n_regions`` additionally groups the prototypes with
    k-means (torchsom's ``cluster``) to outline larger regions on the plots.
    """
    torch.manual_seed(seed)
    data = torch.from_numpy(np.ascontiguousarray(vectors))
    som = SOM(
        x=rows,
        y=cols,
        num_features=data.shape[1],
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        sigma=sigma or max(rows, cols) / 4,
        topology="rectangular",
        neighborhood_function="gaussian",
        distance_function="euclidean",
        lr_decay_function="asymptotic_decay",
        sigma_decay_function="sig_linear_decay_to_one",
        initialization_mode="pca",
        search_backend="torch",
        device="cpu",
        random_seed=seed,
    )
    som.initialize_weights(data, mode="pca")
    qe, te = som.fit(data, verbose=False)
    weights = som.weights.detach().cpu().numpy()
    bmu = som.identify_bmus(data).cpu().numpy()
    hits = np.zeros((rows, cols), dtype=int)
    np.add.at(hits, (bmu[:, 0], bmu[:, 1]), 1)
    trained = TrainedMap(
        rows=rows,
        cols=cols,
        weights=weights,
        bmu=bmu,
        hits=hits,
        umatrix=umatrix(weights),
        qe=[float(v) for v in qe],
        te=[float(v) for v in te],
    )
    if n_regions:
        result = som.cluster(
            method="kmeans", n_clusters=n_regions, feature_space="weights"
        )
        labels = np.asarray(result["labels"]).reshape(rows, cols)
        trained.clusters = labels
    trained.final_qe = float(som.quantization_error(data))
    trained.final_te = 100 * float(som.topographic_error(data))
    return trained
