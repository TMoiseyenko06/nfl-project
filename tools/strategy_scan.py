"""Exhaustive strategy search with multiple-comparison correction.

Testing hundreds of strategies guarantees that some look profitable by chance:
at p<0.05, one in twenty null strategies passes. This scans a large grid and
then applies Benjamini-Hochberg FDR control, so what survives is what survives
*after* accounting for how many shots were taken.

It also splits every survivor in half chronologically. A real edge should
appear in both halves; a lucky one usually lives in a single stretch.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

VIG = 0.045
JUICE_DEC = 100 / 110 + 1  # -110 -> 1.909 decimal


def load(path="artifacts/backtest_qb.parquet"):
    p = pd.read_parquet(path)
    games = p.drop_duplicates("game_id").set_index("game_id")
    out = {}
    for col in ("p_home", "pred_spread", "pred_total", "p_home_h1", "pred_h1_total"):
        if col in p.columns:
            out[col] = p.pivot_table(index="game_id", columns="model", values=col)
    idx = out["p_home"].index
    return {k: v.loc[idx] for k, v in out.items()}, games.loc[idx]


def build_strategies(piv, g):
    """Yield (label, bet_mask, won, decimal_odds) for every strategy in the grid."""
    ok = g.home_win.isin([0.0, 1.0]).to_numpy()
    y = (g.home_win == 1).to_numpy()
    mk = piv["p_home"]["vegas"].to_numpy()
    vs = g.vegas_spread.to_numpy()
    vt = g.vegas_total.to_numpy()
    sa = g.spread_actual.to_numpy()
    ta = g.total_actual.to_numpy()
    week = g.week.to_numpy()
    div = g.get("div_game", pd.Series(np.zeros(len(g)))).to_numpy() if "div_game" in g else np.zeros(len(g))

    models = [m for m in piv["p_home"].columns if m not in ("home_team",)]
    contexts = [
        ("all games", np.ones(len(g), bool)),
        ("wk1-6", week <= 6),
        ("wk7-13", (week > 6) & (week <= 13)),
        ("wk14+", week >= 14),
        ("line<=3", np.abs(vs) <= 3),
        ("line 3-7", (np.abs(vs) > 3) & (np.abs(vs) <= 7)),
        ("line>7", np.abs(vs) > 7),
        ("total<=44", vt <= 44),
        ("total>=48", vt >= 48),
    ]

    for model, (cname, cmask) in itertools.product(models, contexts):
        mo = piv["p_home"][model].to_numpy()

        # --- moneyline, at market price ---
        for side, smask in [("ML any", np.ones(len(g), bool)),
                            ("ML dogs", (mo > 0.5) != (mk > 0.5)),
                            ("ML favs", (mo > 0.5) == (mk > 0.5))]:
            for thr in (0.0, 0.05, 0.10):
                sel = ok & cmask & smask & (np.abs(mo - mk) >= thr)
                if sel.sum() < 60:
                    continue
                pick_home = mo > 0.5
                q = np.where(pick_home, mk, 1 - mk)
                dec = (1 / np.clip(q, 1e-6, 1)) / (1 + VIG)
                won = pick_home == y
                yield (f"{model} | {cname} | {side} | edge>={thr:.2f}",
                       sel, won, dec)

        # --- against the spread, at -110 ---
        if "pred_spread" in piv:
            ps = piv["pred_spread"][model].to_numpy()
            valid = np.isfinite(ps) & np.isfinite(sa) & np.isfinite(vs) & (sa != vs)
            for side, smask in [("ATS any", np.ones(len(g), bool)),
                                ("ATS home", ps > vs), ("ATS away", ps <= vs),
                                ("ATS dogs", (ps > vs) != (vs > 0))]:
                for thr in (0.0, 1.0, 3.0):
                    sel = valid & cmask & smask & (np.abs(ps - vs) >= thr)
                    if sel.sum() < 60:
                        continue
                    won = (ps > vs) == (sa > vs)
                    yield (f"{model} | {cname} | {side} | edge>={thr:.0f}pt",
                           sel, won, np.full(len(g), JUICE_DEC))

        # --- totals, at -110 ---
        if "pred_total" in piv:
            pt = piv["pred_total"][model].to_numpy()
            valid = np.isfinite(pt) & np.isfinite(ta) & np.isfinite(vt) & (ta != vt)
            for side, smask in [("OU any", np.ones(len(g), bool)),
                                ("OU over", pt > vt), ("OU under", pt <= vt)]:
                for thr in (0.0, 1.5, 3.0):
                    sel = valid & cmask & smask & (np.abs(pt - vt) >= thr)
                    if sel.sum() < 60:
                        continue
                    won = (pt > vt) == (ta > vt)
                    yield (f"{model} | {cname} | {side} | edge>={thr:.1f}pt",
                           sel, won, np.full(len(g), JUICE_DEC))


def main():
    piv, g = load()
    seasons = g.season.to_numpy()
    mid = np.median(seasons)
    rows = []
    for label, sel, won, dec in build_strategies(piv, g):
        pr = np.where(won[sel], dec[sel] - 1.0, -1.0)
        if len(pr) < 60:
            continue
        t, p = stats.ttest_1samp(pr, 0.0)
        early, late = sel & (seasons <= mid), sel & (seasons > mid)
        roi_e = np.where(won[early], dec[early] - 1, -1).mean() if early.sum() > 20 else np.nan
        roi_l = np.where(won[late], dec[late] - 1, -1).mean() if late.sum() > 20 else np.nan
        rows.append(dict(strategy=label, n=len(pr), hit=won[sel].mean(),
                         roi=pr.mean(), p=p, roi_early=roi_e, roi_late=roi_l))
    r = pd.DataFrame(rows).sort_values("roi", ascending=False)

    # Benjamini-Hochberg FDR across every strategy tested.
    r = r.sort_values("p").reset_index(drop=True)
    mtests = len(r)
    r["bh_threshold"] = (r.index + 1) / mtests * 0.05
    r["survives_fdr"] = r["p"] <= r["bh_threshold"]
    r = r.sort_values("roi", ascending=False)
    r.to_csv("artifacts/strategy_scan.csv", index=False)

    print("=" * 96)
    print(f"EXHAUSTIVE STRATEGY SCAN - {mtests} strategies tested")
    print("=" * 96)
    print(f"\nAt p<0.05 you expect ~{mtests * 0.05:.0f} false positives by chance alone.")
    print(f"Strategies with raw p<0.05      : {(r.p < 0.05).sum()}")
    print(f"Surviving Benjamini-Hochberg FDR: {r.survives_fdr.sum()}")
    print("\nTOP 15 BY ROI (raw, before correction):")
    print("%-58s %5s %6s %8s %7s %8s %8s" % ("strategy", "n", "hit", "ROI", "p", "early", "late"))
    print("-" * 104)
    for _, x in r.head(15).iterrows():
        print("%-58s %5d %5.1f%% %+7.1f%% %7.3f %+7.1f%% %+7.1f%%" % (
            x.strategy[:58], x.n, x.hit * 100, x.roi * 100, x.p,
            (x.roi_early or 0) * 100, (x.roi_late or 0) * 100))
    surv = r[r.survives_fdr]
    print(f"\nSURVIVORS AFTER MULTIPLE-COMPARISON CORRECTION: {len(surv)}")
    if len(surv):
        for _, x in surv.iterrows():
            print("  %-56s n=%d ROI %+.1f%% p=%.4f" % (x.strategy[:56], x.n, x.roi * 100, x.p))
    else:
        print("  NONE. Every apparently-profitable strategy is consistent with luck.")


if __name__ == "__main__":
    main()
