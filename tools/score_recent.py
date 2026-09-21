import logging; logging.basicConfig(level=logging.WARNING)
import pandas as pd, numpy as np
from nflpred.config import Config
from nflpred.features.build import build_features, select_features
from nflpred.models.registry import LinearModel

c=Config.load('configs/default.yaml')
df=build_features(c, force=True)
lin=dict(C=c.get('models.logistic.C'), alpha=c.get('models.ridge.alpha'),
         max_iter=c.get('models.logistic.max_iter'))
feats=select_features(df,['lean'])

def predict(week):
    tgt=df[(df.season==2026)&(df.week==week)]
    asof=tgt.kickoff.min()
    tr=df[df.played&(df.kickoff<asof)]
    m=LinearModel(**lin,groups=['lean'],derive_binary_from_margin=True)
    m.fit(tr,feats); p=m.predict(tgt,feats)
    o=tgt[['game_id','home_team','away_team','home_score','away_score',
           'total_actual','spread_actual','h1_home_win']].copy()
    for k,v in p.values.items(): o[k]=v
    o['n_train']=len(tr)
    return o

rows=[]
for wk in (1,2):
    rows.append(predict(wk).assign(week=wk))
done=pd.concat(rows,ignore_index=True)
done=done[done.home_score.notna()]

done['pick']=np.where(done.p_home>0.5, done.home_team, done.away_team)
done['winner']=np.where(done.spread_actual>0, done.home_team, done.away_team)
done['right']=done.pick==done.winner
done['conf']=np.maximum(done.p_home,1-done.p_home)
done['pred_tot']=done.pred_total.round(0)
done['tot_err']=(done.total_actual-done.pred_total).abs()
done['hp']=((done.pred_total+done.pred_spread)/2).round(0)
done['ap']=((done.pred_total-done.pred_spread)/2).round(0)

print("="*96)
print("HOW RIGHT WAS IT?  2026 weeks 1-2 - predicted using ONLY games before each kickoff")
print("="*96)
view=pd.DataFrame({
  'wk':done.week,
  'matchup':done.away_team+' @ '+done.home_team,
  'PICK':done.pick,
  'conf':done.conf.map('{:.0%}'.format),
  'ACTUAL':done.winner,
  'hit':np.where(done.right,'YES','no'),
  'proj score':[f'{h:.0f}-{a:.0f}' for h,a in zip(done.hp,done.ap)],
  'real score':[f'{h:.0f}-{a:.0f}' for h,a in zip(done.home_score,done.away_score)],
  'proj tot':done.pred_tot.map('{:.0f}'.format),
  'real tot':done.total_actual.map('{:.0f}'.format),
  'off by':done.tot_err.map('{:.0f}'.format),
})
print(view.to_string(index=False))
print()
print("="*96)
n=len(done); hits=int(done.right.sum())
print(f"  WINNER      : {hits} of {n} correct  =  {hits/n:.1%}")
print(f"  TOTAL POINTS: average miss {done.tot_err.mean():.1f} points")
print(f"                within 7 pts: {(done.tot_err<=7).mean():.0%}   within 14: {(done.tot_err<=14).mean():.0%}")
sc=pd.concat([(done.home_score-done.hp).abs(),(done.away_score-done.ap).abs()])
print(f"  TEAM SCORES : average miss {sc.mean():.1f} points per team")
print()
for b,lab in [(0.70,'confident picks (>=70%)'),(0.0,'all picks')]:
    s=done[done.conf>=b]
    if len(s): print(f"  {lab:26s} {int(s.right.sum())}/{len(s)} = {s.right.mean():.0%}")
