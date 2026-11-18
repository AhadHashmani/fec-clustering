from __future__ import annotations
from typing import Optional
import numpy as np


class FluxEquilibriumClustering:
    """Flux Equilibrium Clustering (FEC) with improved defaults.

    Parameters
    ----------
    k : int, default=10
        Number of nearest neighbours when building the k-NN graph.
        For better clustering, consider k in range [8, 15].
    sigma : Optional[float], default=None
        Gaussian kernel bandwidth for the initial edge weights. If ``None``,
        infer from data. Recommended: use a fraction of median pairwise distance
        (e.g., 0.2 * median_distance) for more discriminative weights.
    alpha : float, default=0.5
        Fraction of stored flux that can be redistributed at each iteration.
        Higher values (0.5-0.8) promote more aggressive clustering.
    T : int, default=25
        Number of transport iterations. Consider 30-50 for better convergence.
    epsilon : float, default=1e-4
        Threshold on the *outgoing* flux below which a node is considered a sink.
        Lower values (1e-4 to 1e-6) help create fewer, larger clusters.
    beta : Optional[float], default=None
        Controls the non-linear reinforcement when re-weighting edges.

        * ``beta is None``  –  use ``phi(f) = log(1+f)`` (recommended).
        * ``beta = 0``      –  recovers the *original* FEC behaviour.
        * any other value   –  use ``phi(f) = f**beta``.

    Notes
    -----
    Implementation hints for better clustering:
    - Handle small datasets: ensure k <= n-1 to avoid boundary errors
    - Consider relaxed downhill flow: allow flux to similar-distance neighbors
    - Add cycle detection in path-following to prevent infinite loops
    - Post-process: merge very small clusters with nearest larger ones
    """

    def __init__(
        self,
        k: int = 10,
        sigma: Optional[float] = None,
        alpha: float = 0.5,
        T: int = 25,
        epsilon: float = 1e-4,
        beta: Optional[float] = None,
    ) -> None:
        self.k = k
        self.sigma = sigma
        self.alpha = alpha
        self.T = T
        self.epsilon = epsilon
        self.beta = beta

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _phi(self, f: np.ndarray, t: int) -> np.ndarray:
        """Flux-aware reinforcement function phi(f) at iteration t."""
        if t == 0 or self.beta == 0:
            return np.ones_like(f)
        if self.beta is None:
            return np.log1p(f)
        return f ** self.beta

    def _outflow(self, f, w, nbr, down, t):
        """Compute delta_ij(t): flux sent along each k-NN edge."""
        # flux-aware edge weights, restricted to downhill edges
        w_tilde = w * self._phi(f, t)[nbr] * down
        denom = w_tilde.sum(axis=1)  # sum over downhill neighbours m of i
        scale = np.zeros_like(denom)
        np.divide(self.alpha * f, denom, out=scale, where=denom > 0)
        return w_tilde * scale[:, None]

    # ------------------------------------------------------------------ #
    # main routine
    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray) -> "FluxEquilibriumClustering":
        """Compute cluster labels for *X*.

        Implementation steps:
        1. Compute global centroid and distances to centroid
        2. Build symmetric k-NN graph with Gaussian edge weights
           - Handle small datasets: use effective_k = min(k, n-1)
           - If sigma is None, infer from data (e.g., fraction of median distance)
        3. Initialize flux: f_i = 1 for all nodes
        4. Run T flux-transport iterations:
           - Apply flux-aware reweighting (beta parameter)
           - Use relaxed downhill criterion (optional improvement)
           - Update flux: f_i(t+1) = f_i(t) - outgoing + incoming
        5. Identify sinks: nodes with outgoing flux < epsilon
        6. Assign clusters via steepest downhill paths
           - Add cycle detection to prevent infinite loops
           - Consider post-processing to merge small clusters

        The final labels must be stored in ``self.labels``.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("X must be a non-empty 2-D array of shape (n, m)")
        n = X.shape[0]

        # ---- Step 1: pre-compute ----------------------------------------
        centroid = X.mean(axis=0)
        d = np.linalg.norm(X - centroid, axis=1)  # distance to centroid

        diff = X[:, None, :] - X[None, :, :]
        D2 = (diff ** 2).sum(axis=2)  # pairwise squared distances
        D = np.sqrt(D2)

        if self.sigma is not None:
            sigma = float(self.sigma)
        elif n > 1:
            sigma = 0.2 * float(np.median(D[np.triu_indices(n, k=1)]))
        else:
            sigma = 1.0
        if not sigma > 0:  # all points identical (or user gave sigma <= 0)
            sigma = 1.0
        tau = 0.1 * sigma

        k_eff = min(self.k, n - 1)

        # k nearest neighbours of each node (self excluded), nearest first
        D2_nosel = D2.copy()
        np.fill_diagonal(D2_nosel, np.inf)
        nbr = np.argsort(D2_nosel, axis=1, kind="stable")[:, :k_eff]
        rows = np.arange(n)[:, None]
        w_ij = np.exp(-D2[rows, nbr] / sigma ** 2)  # shape (n, k_eff)

        # relaxed downhill mask for every edge (i -> nbr[i, j])
        down = d[nbr] <= d[:, None] + tau

        # ---- Step 2: initialise flux ------------------------------------
        flux = np.ones(n)

        # ---- Step 3: simulate flow --------------------------------------
        delta = np.zeros((n, k_eff))
        for t in range(self.T):
            delta = self._outflow(flux, w_ij, nbr, down, t)
            incoming = np.zeros(n)
            np.add.at(incoming, nbr.ravel(), delta.ravel())
            flux = flux - delta.sum(axis=1) + incoming

        # ---- Step 4: detect sinks ---------------------------------------
        out_final = delta.sum(axis=1)
        sinks = out_final < self.epsilon
        if not sinks.any():
            sinks[np.argmin(d)] = True

        # ---- Step 5: assign clusters with cycle detection ---------------
        if k_eff > 0:
            steepest = nbr[np.arange(n), np.argmax(delta, axis=1)]
            steepest = np.where(delta.max(axis=1) > 0, steepest, -1)
        else:
            steepest = -np.ones(n, dtype=int)

        labels = -np.ones(n, dtype=int)
        sink_idx = np.flatnonzero(sinks)
        labels[sink_idx] = np.arange(len(sink_idx))
        next_label = len(sink_idx)

        for i in range(n):
            if labels[i] != -1:
                continue
            path, on_path = [], set()
            cur = i
            while labels[cur] == -1:
                if cur in on_path:  # closed cycle -> break it with a new sink
                    labels[cur] = next_label
                    next_label += 1
                    break
                on_path.add(cur)
                path.append(cur)
                nxt = steepest[cur]
                if nxt < 0:  # no outgoing flux at all: treat as its own sink
                    labels[cur] = next_label
                    next_label += 1
                    break
                cur = nxt
            labels[path] = labels[cur]

        # ---- Step 6: merge small clusters -------------------------------
        s_min = max(2, n // 20)
        uniq, counts = np.unique(labels, return_counts=True)
        large = uniq[counts >= s_min]
        small = uniq[counts < s_min]
        if len(large) > 0 and len(small) > 0:
            cents = {lab: X[labels == lab].mean(axis=0) for lab in uniq}
            merged = labels.copy()
            for lab in small:
                dists = [np.linalg.norm(cents[lab] - cents[big]) for big in large]
                merged[labels == lab] = large[int(np.argmin(dists))]
            labels = merged

        # ---- DO NOT CHANGE THE REST OF THIS METHOD ---- #
        # We require you maintain these attributes for testing purpose.
        self.w_ij = w_ij
        self.flux = flux
        self.sinks = sinks
        self.labels = labels

        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Convenience method that returns the cluster labels."""
        self.fit(X)
        return self.labels

    def __repr__(self) -> str:
        params = (
            f"k={getattr(self, 'k', '?')}",
            f"sigma={getattr(self, 'sigma', '?')}",
            f"alpha={getattr(self, 'alpha', '?')}",
            f"T={getattr(self, 'T', '?')}",
            f"epsilon={getattr(self, 'epsilon', '?')}",
            f"beta={getattr(self, 'beta', '?')}",
        )
        return f"{self.__class__.__name__}({', '.join(params)})"