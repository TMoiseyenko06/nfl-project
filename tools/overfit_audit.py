"""Overfitting audit.

Four independent checks. Walk-forward already prevents training on the test
set, so these look for the subtler failures it does NOT catch.
"""
import logging; logging.basicConfig(level=logging.WARNING)
import numpy as np, pandas as pd
from nflpred.config import Config
from nflpred.features.build import build_features, select_features
from nflpred.models.registry import (DEFAULT_GROUPS, LinearModel, LightGBMModel,
                                     EloBaseline)
from nflpred.models.neural import NeuralModel
from nflpred.backtest.metrics import accuracy, log_loss

c = Config.load('configs/default.yaml')
df = build_features(c)
train = df[df.played & (df.season < 2025)]
test = df[df.played & (df.season >= 2025)]
lin = dict(C=c.get('models.logistic.C'), alpha=c.get('models.ridge.alpha'),
           max_iter=c.get('models.logistic.max_iter'))
def mk():
    return [('linear', LinearModel(**lin, groups=DEFAULT_GROUPS)),
            ('lightgbm', LightGBMModel(params=c.get('models.lightgbm'), seed=c.seed, groups=DEFAULT_GROUPS)),
            ('neural', NeuralModel(groups=DEFAULT_GROUPS, seed=c.seed))]

print("="*76); print("CHECK 1 - TRAIN vs TEST GAP"); print("="*76)
print("A model that memorises scores far better on data it has seen.")
print("A small gap means it is learning patterns, not memorising games.\n")
print("%-12s %10s %10s %9s | %10s %10s" % ("model","train acc","test acc","GAP","train LL","test LL"))
for name, m in mk():
    f = select_features(df, m.groups)
    m.fit(train, f)
    ptr, pte = m.predict(train, f), m.predict(test, f)
    tr = train[train.home_win.isin([0,1])]; te = test[test.home_win.isin([0,1])]
    itr = train.home_win.isin([0,1]).to_numpy(); ite = test.home_win.isin([0,1]).to_numpy()
    a_tr = accuracy(tr.home_win, ptr.values['p_home'][itr])
    a_te = accuracy(te.home_win, pte.values['p_home'][ite])
    l_tr = log_loss(tr.home_win, ptr.values['p_home'][itr])
    l_te = log_loss(te.home_win, pte.values['p_home'][ite])
    flag = "  <-- large" if (a_tr - a_te) > 0.08 else ""
    print("%-12s %10.4f %10.4f %+9.4f | %10.4f %10.4f%s" % (name, a_tr, a_te, a_tr-a_te, l_tr, l_te, flag))

print("\n" + "="*76); print("CHECK 2 - SHUFFLED LABELS (does it find signal that isn't there?)"); print("="*76)
print("Train on randomly shuffled outcomes. The right reference is NOT 50%:")
print("shuffling preserves the marginal home-win rate, so a model that learns")
print("nothing still predicts the base rate and scores ~53% (the home baseline).")
print("Anything meaningfully above THAT means the model is exploiting a bug.\n")
rng = np.random.default_rng(1729)
for name, m in mk():
    f = select_features(df, m.groups)
    sh = train.copy()
    idx = rng.permutation(len(sh))
    for col in ['home_win','spread_actual','total_actual','h1_home_win','h1_spread_actual','h1_total_actual']:
        sh[col] = sh[col].to_numpy()[idx]
    m.fit(sh, f)
    p = m.predict(test, f)
    te = test[test.home_win.isin([0,1])]; ite = test.home_win.isin([0,1]).to_numpy()
    a = accuracy(te.home_win, p.values['p_home'][ite])
    home_base = (te.home_win == 1).mean()
    se = np.sqrt(0.25 / len(te))
    flag = "  <-- SUSPICIOUS" if a > home_base + 2 * se else "  ok (within noise of baseline)"
    print("%-12s shuffled-label acc %.4f  vs home baseline %.4f (+/-%.3f)%s"
          % (name, a, home_base, 2 * se, flag))

print("\n" + "="*76); print("CHECK 3 - DOES A TINY MODEL DO JUST AS WELL?"); print("="*76)
print("83 features on ~3,000 games is a lot. If 6 features match the full set,")
print("the extra 77 are decoration and the model is fitting noise.\n")
tiny = ['elo_diff','d_adj_off_epa_play_net','d_net_epa_ewma','d_rest_days',
        's_expected_total','d_margin_r5']
tiny = [t for t in tiny if t in df.columns]
full = select_features(df, DEFAULT_GROUPS)
te = test[test.home_win.isin([0,1])]; ite = test.home_win.isin([0,1]).to_numpy()
print("%-28s %8s %10s %10s" % ("feature set","n feats","test acc","total MAE"))
for label, feats in [(f"tiny ({len(tiny)} features)", tiny), (f"full ({len(full)} features)", full)]:
    m = LinearModel(**lin, groups=DEFAULT_GROUPS)
    m.fit(train, feats); p = m.predict(test, feats)
    a = accuracy(te.home_win, p.values['p_home'][ite])
    mae = np.abs(test.total_actual.to_numpy() - p.values['pred_total']).mean()
    print("%-28s %8d %10.4f %10.3f" % (label, len(feats), a, mae))

print("\n" + "="*76); print("CHECK 4 - IS PERFORMANCE STABLE OVER TIME?"); print("="*76)
print("A model tuned to the past degrades as it moves away from it.\n")
p = pd.read_parquet('artifacts/backtest_predictions.parquet')
d = p[(p.model=='neural') & p.home_win.isin([0,1])].copy()
d['correct'] = (d.p_home>0.5)==(d.home_win==1)
g = d.groupby('season').correct.agg(['size','mean'])
for s, r in g.iterrows():
    bar = '#' * int(r['mean']*40)
    print("  %d  n=%4d  %.1f%%  %s" % (s, r['size'], r['mean']*100, bar))
early = d[d.season<=2021].correct.mean(); late = d[d.season>=2022].correct.mean()
print("\n  2017-2021: %.1f%%   2022-2026: %.1f%%   drift %+.1f pts" % (early*100, late*100, (late-early)*100))
