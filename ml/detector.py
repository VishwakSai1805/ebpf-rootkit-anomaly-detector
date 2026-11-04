"""Unsupervised anomaly detector: Isolation Forest + numpy Autoencoder, trained on benign data only.

Training uses scikit-learn for the forest; the fitted trees are exported to flat arrays so that
inference is a vectorised numpy walk (sklearn's ``predict`` costs several ms per call and would
break the <1 ms budget). Models persist as ``.npz`` (no pickle).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest

from .features import FEATURE_DIM, N_UNI, S, SC_OFF

EULER = 0.5772156649
# The forest sees unigram + behavioural-scalar columns only: with ~500 mostly-zero n-gram columns
# random splits almost never pick an informative one. The autoencoder sees everything.
IF_COLS = np.r_[0:N_UNI, SC_OFF:FEATURE_DIM]
# Scalars get extra weight in the autoencoder loss/score so one rare flag is not drowned out by 480 n-gram dims.
SCALAR_WEIGHT = 12.0
AE_W = np.ones(FEATURE_DIM, dtype=np.float32)
AE_W[SC_OFF:] = SCALAR_WEIGHT


def _c(n: np.ndarray | float) -> np.ndarray:
    """Average path length of an unsuccessful BST search (Isolation Forest normaliser)."""
    n = np.asarray(n, dtype=np.float64)
    out = np.zeros_like(n)
    big = n > 2
    out[big] = 2.0 * (np.log(n[big] - 1.0) + EULER) - 2.0 * (n[big] - 1.0) / n[big]
    out[n == 2] = 1.0
    return out


@dataclass
class Detection:
    anomaly: bool
    score: float            # fused: max(if/thr_if, ae/thr_ae); >= 1.0 means anomalous
    if_score: float
    ae_error: float
    categories: list[str] = field(default_factory=list)


def categorize(x: np.ndarray) -> list[str]:
    """Human-readable attack categories from the raw behavioural scalars of a feature vector."""
    v = x[SC_OFF:]
    cats = []
    l2 = np.log1p(2) - 1e-6
    l3 = np.log1p(3) - 1e-6
    if v[S["unexplained_root"]] > 0 or (v[S["root_transition"]] > 0 and v[S["exec_tmp"]] > 0):
        cats.append("privilege_escalation")
    if v[S["revshell_chain"]] > 0 or v[S["sock_dup"]] >= l2:
        cats.append("reverse_shell")
    if v[S["preload_env"]] > 0 or v[S["preload_open"]] > 0:
        cats.append("ld_preload_injection")
    if v[S["ptrace_poke"]] >= l3 or (v[S["ptrace_foreign"]] > 0 and (v[S["ptrace_poke"]] > 0 or v[S["ptrace_regs"]] > 0)):
        cats.append("ptrace_injection")
    if v[S["hidden_pids"]] > 0:
        cats.append("hidden_process")
    return cats


class AnomalyDetector:
    def __init__(self, n_trees: int = 100, max_samples: int = 256, ae_hidden: tuple[int, int] = (64, 16),
                 epochs: int = 60, percentile: float = 99.7, seed: int = 0, z_clip: float = 8.0,
                 min_std: float = 0.05):
        self.n_trees, self.max_samples, self.ae_hidden = n_trees, max_samples, ae_hidden
        self.epochs, self.percentile, self.seed = epochs, percentile, seed
        self.z_clip, self.min_std = z_clip, min_std
        self.fitted = False

    # ------------------------------------------------------------------ training
    def fit(self, X: np.ndarray) -> "AnomalyDetector":
        X = np.asarray(X, dtype=np.float32)
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(X))
        n_val = max(len(X) // 5, 1)
        val, tr = X[idx[:n_val]], X[idx[n_val:]]
        self.mean = tr.mean(0)
        self.std = np.maximum(tr.std(0), self.min_std).astype(np.float32)

        forest = IsolationForest(n_estimators=self.n_trees, max_samples=min(self.max_samples, len(tr)),
                                 random_state=self.seed, n_jobs=1).fit(tr[:, IF_COLS])
        self._sk_forest = forest  # kept in memory only (parity tests); never persisted
        self._export_forest(forest, min(self.max_samples, len(tr)))
        self._fit_autoencoder(self._z(tr), rng)

        self.thr_if = float(np.percentile(self.if_scores(val), self.percentile))
        self.thr_ae = float(np.percentile(self.ae_errors(val), self.percentile))
        self.fitted = True
        return self

    def _export_forest(self, forest: IsolationForest, psi: int) -> None:
        trees = [e.tree_ for e in forest.estimators_]
        feats = forest.estimators_features_
        nmax = max(t.node_count for t in trees)
        T = len(trees)
        self.t_feat = np.zeros((T, nmax), dtype=np.int32)
        self.t_thr = np.zeros((T, nmax), dtype=np.float32)
        self.t_left = np.tile(np.arange(nmax, dtype=np.int32), (T, 1))
        self.t_right = self.t_left.copy()
        self.t_leaf = np.zeros((T, nmax), dtype=np.float32)
        depth_max = 0
        for i, t in enumerate(trees):
            depth = np.zeros(t.node_count, dtype=np.int32)
            for node in range(t.node_count):  # children always have larger ids than parents
                l, r = t.children_left[node], t.children_right[node]
                if l != -1:
                    depth[l] = depth[r] = depth[node] + 1
                    self.t_feat[i, node] = IF_COLS[feats[i][t.feature[node]]]
                    self.t_thr[i, node] = t.threshold[node]
                    self.t_left[i, node], self.t_right[i, node] = l, r
                else:
                    self.t_leaf[i, node] = depth[node] + _c(t.n_node_samples[node])
            depth_max = max(depth_max, int(depth.max()))
        self.depth_max = depth_max
        self.c_psi = float(_c(psi))

    def _fit_autoencoder(self, Z: np.ndarray, rng: np.random.Generator) -> None:
        h1, h2 = self.ae_hidden
        dims = [FEATURE_DIM, h1, h2, h1, FEATURE_DIM]
        W = [rng.normal(0, np.sqrt(2.0 / a), (a, b)).astype(np.float32) for a, b in zip(dims[:-1], dims[1:])]
        B = [np.zeros(b, dtype=np.float32) for b in dims[1:]]
        m = [np.zeros_like(p) for p in W + B]
        v = [np.zeros_like(p) for p in W + B]
        step, lr, b1, b2 = 0, 2e-3, 0.9, 0.999
        bs = 128
        for _ in range(self.epochs):
            order = rng.permutation(len(Z))
            for s in range(0, len(Z), bs):
                x = Z[order[s:s + bs]]
                acts = [x]
                for li in range(4):
                    a = acts[-1] @ W[li] + B[li]
                    acts.append(np.maximum(a, 0) if li < 3 else a)
                g = 2.0 * AE_W * (acts[-1] - x) / x.shape[0]
                gW, gB = [None] * 4, [None] * 4
                for li in range(3, -1, -1):
                    gW[li] = acts[li].T @ g
                    gB[li] = g.sum(0)
                    if li:
                        g = (g @ W[li].T) * (acts[li] > 0)
                step += 1
                for j, (p, gr) in enumerate(zip(W + B, gW + gB)):
                    m[j] = b1 * m[j] + (1 - b1) * gr
                    v[j] = b2 * v[j] + (1 - b2) * gr * gr
                    p -= lr * (m[j] / (1 - b1 ** step)) / (np.sqrt(v[j] / (1 - b2 ** step)) + 1e-8)
        self.ae_W, self.ae_B = W, B

    # ------------------------------------------------------------------ inference
    def _z(self, X: np.ndarray) -> np.ndarray:
        return np.clip((X - self.mean) / self.std, -self.z_clip, self.z_clip).astype(np.float32)

    def if_scores(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X).astype(np.float32, copy=False)
        N, T = len(X), self.t_feat.shape[0]
        tr = np.arange(T)[None, :]
        rows = np.arange(N)[:, None]
        node = np.zeros((N, T), dtype=np.int32)
        for _ in range(self.depth_max):
            go_left = X[rows, self.t_feat[tr, node]] <= self.t_thr[tr, node]
            node = np.where(go_left, self.t_left[tr, node], self.t_right[tr, node])
        h = self.t_leaf[tr, node].mean(1)
        return np.power(2.0, -h / self.c_psi)

    def ae_errors(self, X: np.ndarray) -> np.ndarray:
        x = self._z(np.atleast_2d(X))
        a = x
        for li in range(4):
            a = a @ self.ae_W[li] + self.ae_B[li]
            if li < 3:
                a = np.maximum(a, 0)
        return (AE_W * (a - x) ** 2).mean(1)

    def score_batch(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        s_if, s_ae = self.if_scores(X), self.ae_errors(X)
        return np.maximum(s_if / self.thr_if, s_ae / self.thr_ae), s_if, s_ae

    def detect(self, x: np.ndarray) -> Detection:
        fused, s_if, s_ae = self.score_batch(x)
        anomaly = bool(fused[0] >= 1.0)
        return Detection(anomaly, float(fused[0]), float(s_if[0]), float(s_ae[0]),
                         categorize(x) or ["behavioral_anomaly"] if anomaly else [])

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrs = {"mean": self.mean, "std": self.std, "t_feat": self.t_feat, "t_thr": self.t_thr,
                "t_left": self.t_left, "t_right": self.t_right, "t_leaf": self.t_leaf}
        for i, (w, b) in enumerate(zip(self.ae_W, self.ae_B)):
            arrs[f"W{i}"], arrs[f"B{i}"] = w, b
        meta = {"depth_max": self.depth_max, "c_psi": self.c_psi, "thr_if": self.thr_if, "thr_ae": self.thr_ae,
                "z_clip": self.z_clip, "min_std": self.min_std, "feature_dim": FEATURE_DIM,
                "percentile": self.percentile, "ae_hidden": list(self.ae_hidden)}
        np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrs)

    @classmethod
    def load(cls, path: str | Path) -> "AnomalyDetector":
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            if meta["feature_dim"] != FEATURE_DIM:
                raise ValueError("model was trained with a different feature set; retrain")
            d = cls(ae_hidden=tuple(meta["ae_hidden"]), percentile=meta["percentile"],
                    z_clip=meta["z_clip"], min_std=meta["min_std"])
            for k in ("mean", "std", "t_feat", "t_thr", "t_left", "t_right", "t_leaf"):
                setattr(d, k, z[k])
            d.ae_W = [z[f"W{i}"] for i in range(4)]
            d.ae_B = [z[f"B{i}"] for i in range(4)]
        d.depth_max, d.c_psi = meta["depth_max"], meta["c_psi"]
        d.thr_if, d.thr_ae = meta["thr_if"], meta["thr_ae"]
        d.n_trees = d.t_feat.shape[0]
        d.fitted = True
        return d
