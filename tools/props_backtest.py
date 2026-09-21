import logging; logging.basicConfig(level=logging.WARNING)
import numpy as np, pandas as pd, lightgbm as lgb
from nflpred.config import Config
from nflpred.features.props import build_player_features
c=Config.load('configs/default.yaml')
d=build_player_features(c, list(range(2018,2027)))
d['season_week']=d.season*100+d.week
BASE=['targets_ewma','receptions_ewma','receiving_yards_ewma','carries_ewma','rushing_yards_ewma',
      'target_share_ewma','air_yards_share_ewma','offense_pct_ewma','targets_std','carries_std',
      'games_played','team_targets_ewma','targets_shrunk','carries_shrunk','pos_teammates_out']
VAC=['vacated_target_share','vacated_carries','vacated_x_role','vacated_car_x_role']
def run(tgt,pos,minroll,feats):
    s=d[d.position.isin(pos)&d[tgt].notna()].copy(); roll=f'{tgt}_ewma'
    s=s[s[roll].notna()&(s[roll]>=minroll)]
    o=[]
    for w in sorted(s[s.season>=2022].season_week.unique()):
        tr=s[s.season_week<w]; te=s[s.season_week==w]
        if len(tr)<3000 or te.empty: continue
        m=lgb.LGBMRegressor(n_estimators=250,learning_rate=0.05,num_leaves=15,min_child_samples=40,
            subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=5.0,verbosity=-1,
            random_state=c.seed,n_jobs=2)
        f=[x for x in feats if x in tr.columns]
        m.fit(tr[f].fillna(0),tr[tgt])
        o.append(pd.DataFrame({'actual':te[tgt].to_numpy(),'dumb':te[roll].to_numpy(),
            'model':m.predict(te[f].fillna(0)),'vac':te.vacated_target_share.to_numpy()}))
    return pd.concat(o,ignore_index=True)
def r2(a,p): return 1-((a-p)**2).sum()/((a-a.mean())**2).sum()
for tgt,pos,mr in [('targets',['WR','TE','RB'],2.0),('receptions',['WR','TE','RB'],1.0)]:
    a=run(tgt,pos,mr,BASE); b=run(tgt,pos,mr,BASE+VAC)
    print('='*74); print(tgt.upper()); print('='*74)
    print('  %-26s %10s %9s' % ('','MAE','r2'))
    print('  %-26s %10.3f %9.3f' % ('dumb rolling average',(a.actual-a.dumb).abs().mean(),r2(a.actual,a.dumb)))
    print('  %-26s %10.3f %9.3f' % ('model (no vacated feats)',(a.actual-a.model).abs().mean(),r2(a.actual,a.model)))
    print('  %-26s %10.3f %9.3f' % ('model + VACATED VOLUME',(b.actual-b.model).abs().mean(),r2(b.actual,b.model)))
    print()
    print('  edge over the dumb baseline, by how much volume was vacated:')
    for lo,hi,lab in [(0,0.001,'nothing vacated'),(0.001,0.16,'some'),(0.16,9,'LARGE (16%+)')]:
        x=b[(b.vac>=lo)&(b.vac<hi)]
        if len(x)<200: continue
        dm=(x.actual-x.dumb).abs().mean(); mm=(x.actual-x.model).abs().mean()
        print('    %-18s n=%5d  dumb %.3f  model %.3f   %+.1f%%' % (lab,len(x),dm,mm,(dm-mm)/dm*100))
    print()
