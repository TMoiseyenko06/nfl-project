"""Per-criterion accuracy report for the last two seasons."""
import sys
import pandas as pd, numpy as np
from scipy import stats

MODEL = sys.argv[1] if len(sys.argv) > 1 else "neural"
SEASONS = [2025, 2026]
p = pd.read_parquet('artifacts/backtest_predictions.parquet')
d = p[(p.model == MODEL) & (p.season.isin(SEASONS))].copy()
t = p[p.season.isin(SEASONS)].drop_duplicates('game_id').set_index('game_id')

print("=" * 78)
print(f"SCORECARD - model '{MODEL}', {SEASONS[0]}-{SEASONS[-1]}, blind walk-forward")
print(f"{len(d)} games. Every prediction made using only data from before that game.")
print("=" * 78)

def band(p_):
    return 'strong' if p_ >= .75 else ('likely' if p_ >= .65 else ('lean' if p_ >= .57 else 'coin flip'))

# ---- binary criteria -------------------------------------------------------
for label, pcol, acol, naive_label, naive_fn in [
    ("1. WHO WINS", "p_home", "home_win", "always pick home",
     lambda s: (s.home_win == 1).mean()),
    ("2. WHO LEADS AT HALFTIME", "p_home_h1", "h1_home_win", "always pick home",
     lambda s: (s.h1_home_win == 1).mean()),
]:
    s = d[d[acol].isin([0.0, 1.0])].copy()
    if s.empty:
        continue
    s['correct'] = (s[pcol] > 0.5) == (s[acol] == 1)
    n, hit = len(s), s.correct.mean()
    lo, hi = stats.binomtest(int(s.correct.sum()), n).proportion_ci(0.95)
    print(f"\n{label}")
    print(f"   RIGHT {hit:.1%} of the time   ({int(s.correct.sum())} of {n})")
    print(f"   95% range: {lo:.1%} - {hi:.1%}")
    print(f"   baseline ({naive_label}): {naive_fn(s):.1%}")
    s['b'] = np.maximum(s[pcol], 1 - s[pcol]).map(band)
    print("   by confidence:")
    for b in ['strong', 'likely', 'lean', 'coin flip']:
        g = s[s.b == b]
        if len(g) < 10:
            continue
        print(f"      {b:10s} n={len(g):3d}   right {g.correct.mean():.0%}")

# ---- numeric criteria ------------------------------------------------------
def numeric(label, pred, actual, naive, thresholds):
    err = (actual - pred).abs()
    nerr = (actual - naive).abs()
    print(f"\n{label}")
    print(f"   average miss: {err.mean():.1f} points   (guessing: {nerr.mean():.1f})")
    edge = (nerr.mean() - err.mean()) / nerr.mean() * 100
    verdict = "better than guessing" if edge > 1 else ("NO BETTER than guessing" if edge > -1 else "WORSE than guessing")
    print(f"   {edge:+.1f}% vs guessing  ->  {verdict}")
    print("   how often it lands within:")
    for th in thresholds:
        print(f"      {th:2d} points : {(err <= th).mean():.0%}   (guessing: {(nerr <= th).mean():.0%})")

g = d.dropna(subset=['pred_total', 'total_actual'])
numeric("3. TOTAL POINTS", g.pred_total, g.total_actual, g.total_actual.mean(), [3, 7, 10, 14])

g = d.dropna(subset=['pred_spread', 'spread_actual'])
numeric("4. MARGIN OF VICTORY", g.pred_spread, g.spread_actual, g.spread_actual.mean(), [3, 7, 10, 14])

g = d.dropna(subset=['pred_total', 'pred_spread', 'total_actual', 'spread_actual'])
hp, ap = (g.pred_total + g.pred_spread) / 2, (g.pred_total - g.pred_spread) / 2
ah, aa = (g.total_actual + g.spread_actual) / 2, (g.total_actual - g.spread_actual) / 2
both = pd.concat([(ah - hp).abs(), (aa - ap).abs()])
nboth = pd.concat([(ah - ah.mean()).abs(), (aa - aa.mean()).abs()])
print("\n5. EACH TEAM'S SCORE")
print(f"   average miss per team: {both.mean():.1f} points   (guessing: {nboth.mean():.1f})")
print("   how often a team's score lands within:")
for th in [3, 7, 10]:
    print(f"      {th:2d} points : {(both <= th).mean():.0%}   (guessing: {(nboth <= th).mean():.0%})")

g = d.dropna(subset=['pred_h1_total', 'h1_total_actual'])
if len(g):
    numeric("6. HALFTIME TOTAL POINTS", g.pred_h1_total, g.h1_total_actual,
            g.h1_total_actual.mean(), [3, 5, 7, 10])
