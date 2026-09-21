"""Neural network with learned team (and optionally QB) embeddings.

Phase 3 of the plan. Deliberately small: ~3,000 training games is a tiny
dataset for a neural network, so every knob here is set for regularisation
rather than capacity - 8-dimensional embeddings, one hidden layer of 32 units,
heavy dropout, weight decay, and early stopping on a chronological tail of the
TRAINING fold (never the test fold).

Architecture is a shared trunk with one head per target. Sharing the trunk is
the main thing a neural network can do that a gradient-boosted tree cannot:
the team embeddings are learned jointly across win/spread/total rather than
refit independently for each.

Fair-comparison note: ``NeuralModel`` sees exactly the numeric features the
tree models see, plus team identity. The QB variant sees more, and is reported
separately for that reason - see ``use_qb``.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from nflpred.models.registry import BaseModel, Predictions, _usable
from nflpred.models.targets import BINARY_FROM_MARGIN

log = logging.getLogger(__name__)

UNKNOWN = 0  # embedding index reserved for identities unseen in training


_THREADS_SET = False


def _torch():
    """Import torch, pinning thread count once.

    Left at its default, torch oversubscribes cores and thrashes against any
    other work in the process - which made an identical fit measure 40s in one
    run and 2.1s in another. Pinning it keeps fold timings meaningful and the
    walk-forward reproducible.
    """
    global _THREADS_SET
    import torch

    if not _THREADS_SET:
        torch.set_num_threads(2)
        _THREADS_SET = True
    return torch


class _Net:
    """Lazily-built torch module (kept out of import time so torch stays optional)."""

    def __init__(self, n_teams, n_qbs, n_numeric, target_specs, emb_dim, hidden, dropout, seed):
        torch = _torch()
        import torch.nn as nn

        torch.manual_seed(seed)
        self.torch = torch
        self.specs = target_specs
        self.use_qb = n_qbs > 1

        class Module(nn.Module):
            def __init__(self):
                super().__init__()
                self.team_emb = nn.Embedding(n_teams, emb_dim)
                nn.init.normal_(self.team_emb.weight, std=0.05)
                in_dim = n_numeric + 2 * emb_dim
                if n_qbs > 1:
                    self.qb_emb = nn.Embedding(n_qbs, emb_dim)
                    nn.init.normal_(self.qb_emb.weight, std=0.05)
                    in_dim += 2 * emb_dim
                else:
                    self.qb_emb = None
                self.trunk = nn.Sequential(
                    nn.Linear(in_dim, hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, hidden // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                self.heads = nn.ModuleDict(
                    {s.name: nn.Linear(hidden // 2, 1) for s in target_specs}
                )

            def forward(self, x_num, home_idx, away_idx, home_qb, away_qb):
                parts = [x_num, self.team_emb(home_idx), self.team_emb(away_idx)]
                if self.qb_emb is not None:
                    parts += [self.qb_emb(home_qb), self.qb_emb(away_qb)]
                z = self.trunk(self.torch_cat(parts))
                return {name: head(z).squeeze(-1) for name, head in self.heads.items()}

            @staticmethod
            def torch_cat(parts):
                import torch

                return torch.cat(parts, dim=1)

        self.module = Module()


class NeuralModel(BaseModel):
    """Small embedding MLP, multi-head over the requested targets."""

    name = "neural"

    def __init__(self, groups: list[str] | None = None, targets: list[str] | None = None,
                 name: str | None = None, seed: int = 1729, emb_dim: int = 8,
                 hidden: int = 32, dropout: float = 0.4, weight_decay: float = 1e-3,
                 lr: float = 3e-3, max_epochs: int = 400, patience: int = 30,
                 val_fraction: float = 0.15, batch_size: int = 256,
                 use_qb: bool = False, derive_binary_from_margin: bool = True):
        super().__init__(targets)
        self.groups = groups or ["context", "form", "adjusted", "matchup", "elo"]
        if name:
            self.name = name
        self.seed, self.emb_dim, self.hidden = seed, emb_dim, hidden
        self.dropout, self.weight_decay, self.lr = dropout, weight_decay, lr
        self.max_epochs, self.patience = max_epochs, patience
        self.val_fraction, self.batch_size = val_fraction, batch_size
        self.use_qb = use_qb
        self.derive_binary_from_margin = derive_binary_from_margin

    # --- encoding -------------------------------------------------------
    def _fit_vocab(self, train: pd.DataFrame) -> None:
        teams = sorted(set(train["home_team"]) | set(train["away_team"]))
        self.team_ix_ = {t: i + 1 for i, t in enumerate(teams)}   # 0 = unknown
        if self.use_qb:
            qbs = pd.concat([train["home_qb_id"], train["away_qb_id"]]).dropna()
            # Only QBs with a real sample; the rest collapse to "unknown".
            counts = qbs.value_counts()
            keep = counts[counts >= 4].index
            self.qb_ix_ = {q: i + 1 for i, q in enumerate(sorted(keep))}
        else:
            self.qb_ix_ = {}

    def _encode(self, df: pd.DataFrame, features: list[str]):
        torch = _torch()
        X = df[features].to_numpy(dtype=float)
        X = np.where(np.isfinite(X), X, np.nan)
        X = np.where(np.isnan(X), self.median_, X)
        X = (X - self.mean_) / self.scale_
        home = df["home_team"].map(self.team_ix_).fillna(UNKNOWN).to_numpy(dtype=np.int64)
        away = df["away_team"].map(self.team_ix_).fillna(UNKNOWN).to_numpy(dtype=np.int64)
        if self.use_qb:
            hq = df["home_qb_id"].map(self.qb_ix_).fillna(UNKNOWN).to_numpy(dtype=np.int64)
            aq = df["away_qb_id"].map(self.qb_ix_).fillna(UNKNOWN).to_numpy(dtype=np.int64)
        else:
            hq = aq = np.zeros(len(df), dtype=np.int64)
        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(home), torch.tensor(away),
            torch.tensor(hq), torch.tensor(aq),
        )

    # --- fit ------------------------------------------------------------
    def fit(self, train: pd.DataFrame, features: list[str]) -> None:
        torch = _torch()
        import torch.nn.functional as F

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)

        self.features_ = features
        self._fit_vocab(train)

        # Impute/standardise using training statistics only.
        raw = train[features].to_numpy(dtype=float)
        self.median_ = np.nanmedian(np.where(np.isfinite(raw), raw, np.nan), axis=0)
        self.median_ = np.where(np.isfinite(self.median_), self.median_, 0.0)
        filled = np.where(np.isfinite(raw), raw, self.median_)
        self.mean_ = filled.mean(axis=0)
        self.scale_ = filled.std(axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0

        # Chronological validation tail for early stopping - NEVER the test fold.
        train = train.sort_values("kickoff")
        n_val = max(60, int(len(train) * self.val_fraction))
        tr, va = train.iloc[:-n_val], train.iloc[-n_val:]
        if len(tr) < 200:
            tr, va = train, train

        # Per-target normalisation of regression labels so no single scale dominates.
        self.y_mu_, self.y_sd_ = {}, {}
        fit_specs = [s for s in self.specs if not self._derives(s)]
        for spec in fit_specs:
            vals = pd.to_numeric(train[spec.column], errors="coerce").dropna()
            if spec.is_binary:
                self.y_mu_[spec.name], self.y_sd_[spec.name] = 0.0, 1.0
            else:
                self.y_mu_[spec.name] = float(vals.mean())
                self.y_sd_[spec.name] = float(vals.std()) or 1.0

        net = _Net(len(self.team_ix_) + 1, len(self.qb_ix_) + 1, len(features),
                   fit_specs, self.emb_dim, self.hidden, self.dropout, self.seed)
        model = net.module
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        def pack(frame):
            xs = self._encode(frame, features)
            ys, ms = {}, {}
            for spec in fit_specs:
                v = pd.to_numeric(frame[spec.column], errors="coerce")
                if spec.is_binary:
                    v = v.where(v.isin([0.0, 1.0]))
                mask = v.notna().to_numpy()
                arr = v.fillna(0.0).to_numpy(dtype=float)
                if not spec.is_binary:
                    arr = (arr - self.y_mu_[spec.name]) / self.y_sd_[spec.name]
                ys[spec.name] = torch.tensor(arr, dtype=torch.float32)
                ms[spec.name] = torch.tensor(mask)
            return xs, ys, ms

        (Xtr, htr, atr, qhtr, qatr), ytr, mtr = pack(tr)
        (Xva, hva, ava, qhva, qava), yva, mva = pack(va)

        def loss_of(out, ys, ms):
            total = 0.0
            for spec in fit_specs:
                m = ms[spec.name]
                if not m.any():
                    continue
                pred, target = out[spec.name][m], ys[spec.name][m]
                total = total + (
                    F.binary_cross_entropy_with_logits(pred, target)
                    if spec.is_binary else F.mse_loss(pred, target)
                )
            return total

        best, best_state, bad = float("inf"), None, 0
        n = len(tr)
        gen = torch.Generator().manual_seed(self.seed)
        for epoch in range(self.max_epochs):
            model.train()
            perm = torch.randperm(n, generator=gen)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                out = model(Xtr[idx], htr[idx], atr[idx], qhtr[idx], qatr[idx])
                loss = loss_of(out, {k: v[idx] for k, v in ytr.items()},
                               {k: v[idx] for k, v in mtr.items()})
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vloss = float(loss_of(model(Xva, hva, ava, qhva, qava), yva, mva))
            if vloss < best - 1e-5:
                best, bad = vloss, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self.model_, self.fit_specs_, self.epochs_run_ = model, fit_specs, epoch + 1

        # Coherent probabilities: map the predicted margin to a win probability.
        self.margin_maps_ = {}
        if self.derive_binary_from_margin:
            from sklearn.linear_model import LogisticRegression

            preds = self._raw_predict(train, features)
            for spec in self.specs:
                if not self._derives(spec):
                    continue
                margin_name = BINARY_FROM_MARGIN[spec.name]
                d = _usable(train, spec)
                m = train.index.isin(d.index)
                x = preds[margin_name][m].reshape(-1, 1)
                y = d[spec.column].to_numpy(dtype=int)
                if len(np.unique(y)) < 2:
                    continue
                self.margin_maps_[spec.name] = LogisticRegression(
                    fit_intercept=False, max_iter=1000
                ).fit(x, y)

    def _derives(self, spec) -> bool:
        names = {s.name for s in self.specs}
        return (
            self.derive_binary_from_margin
            and spec.is_binary
            and BINARY_FROM_MARGIN.get(spec.name) in names
        )

    def _raw_predict(self, df: pd.DataFrame, features: list[str]) -> dict[str, np.ndarray]:
        torch = _torch()
        X, h, a, qh, qa = self._encode(df, features)
        with torch.no_grad():
            out = self.model_(X, h, a, qh, qa)
        res = {}
        for spec in self.fit_specs_:
            v = out[spec.name].numpy()
            if spec.is_binary:
                res[spec.name] = 1.0 / (1.0 + np.exp(-v))
            else:
                res[spec.name] = v * self.y_sd_[spec.name] + self.y_mu_[spec.name]
        return res

    def predict(self, test: pd.DataFrame, features: list[str]) -> Predictions:
        raw = self._raw_predict(test, features)
        p = Predictions()
        for spec in self.specs:
            if spec.name in self.margin_maps_:
                margin = raw[BINARY_FROM_MARGIN[spec.name]]
                clf = self.margin_maps_[spec.name]
                p.set(spec, clf.predict_proba(margin.reshape(-1, 1))[:, 1])
            elif spec.name in raw:
                p.set(spec, raw[spec.name])
            else:
                margin = raw.get(BINARY_FROM_MARGIN.get(spec.name, ""), None)
                p.set(spec, (margin > 0).astype(float) if margin is not None
                      else np.full(len(test), 0.5))
        return p

    def team_embeddings(self) -> pd.DataFrame:
        """The learned team vectors - the thing only the neural net produces."""
        w = self.model_.team_emb.weight.detach().numpy()
        rows = {t: w[i] for t, i in self.team_ix_.items()}
        return pd.DataFrame(rows).T
