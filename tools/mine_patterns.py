"""Mine play-by-play for patterns that survive controlling for the obvious."""
import logging; logging.basicConfig(level=logging.WARNING)
import pandas as pd, numpy as np
from nflpred.config import Config
from nflpred.ingest.nflverse import load_games
from nflpred.analysis.patterns import load_pbp, add_outcomes, residual_mine

c=Config.load('configs/default.yaml')
DISC=list(range(2015,2023)); HOLD=list(range(2023,2027))
games=load_games(c)
print("loading play-by-play...", flush=True)
d=add_outcomes(load_pbp(c, DISC+HOLD), games)
d=d[d.play_type.isin(['pass','run'])]
disc=d[d.season.isin(DISC)]; hold=d[d.season.isin(HOLD)]
print(f"discovery {len(disc):,} plays | holdout {len(hold):,} plays\n")

DIMS=['quarter','score_state','down','distance','field_pos','time_left','formation','tempo']
# The obvious drivers. A pattern must beat these to count as a finding.
CONTROL=['score_state','time_left','field_pos','down','distance']

for outcome,label in [('posteam_wins','the team with the ball WINS'),
                      ('drive_ends_score','the drive ENDS IN A SCORE'),
                      ('next_play_explosive','the NEXT PLAY gains 15+')]:
    r=residual_mine(disc,hold,outcome,DIMS,CONTROL,min_n=400,max_depth=2)
    if r.empty: continue
    good=r[r.survives_fdr & r.replicated & (r.excess.abs()>=0.015)]
    print("="*98)
    print(f"BEYOND the obvious -> {label}")
    print("="*98)
    print(f"  tested {len(r):,} | survive FDR {int(r.survives_fdr.sum()):,} | AND replicate {len(good):,}")
    if len(good):
        print()
        print("  %-46s %8s %10s %10s" % ("condition","n","excess","holdout"))
        print("  "+"-"*78)
        for _,x in good.head(10).iterrows():
            print("  %-46s %8d %+9.1f%% %+9.1f%%" % (x.condition[:46],x.n,x.excess*100,x.holdout_excess*100))
    else:
        print("  Nothing beats the obvious variables and replicates out of sample.")
    print()
