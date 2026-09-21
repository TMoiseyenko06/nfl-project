"""Single CLI entrypoint.

    nflpred fetch            download nflverse data into the versioned cache
    nflpred build-features   assemble the leak-free modelling table
    nflpred train            fit production models on everything completed
    nflpred backtest         walk-forward evaluation + comparison table
    nflpred predict-week     predict the upcoming week and log it before kickoff
    nflpred score            score previously logged predictions
    nflpred audit            data/leakage/log health checks
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from nflpred.config import Config


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_fetch(args, cfg: Config) -> int:
    from nflpred.ingest.nflverse import cache_vintage, fetch

    manifest = fetch(cfg, force=args.force)
    print(f"cached {len(manifest)} file(s) in {cfg.raw_cache}")
    print(f"data vintage: {cache_vintage(cfg)}")
    return 0


def cmd_build_features(args, cfg: Config) -> int:
    from nflpred.features.build import build_features, feature_groups

    df = build_features(cfg, force=args.force)
    groups = feature_groups(df)
    print(f"\nfeature table: {len(df)} games x {df.shape[1]} columns")
    print(f"seasons {df.season.min()}-{df.season.max()}  "
          f"played={int(df.played.sum())}  unplayed={int((~df.played).sum())}")
    print("\nfeature groups:")
    for name, cols in groups.items():
        print(f"  {name:10s} {len(cols):3d}")
    print(f"  {'TOTAL':10s} {sum(len(c) for c in groups.values()):3d}")
    return 0


def cmd_backtest(args, cfg: Config) -> int:
    from nflpred.backtest.walkforward import (by_season, format_table,
                                              run_backtest, summarize)
    from nflpred.features.build import build_features
    from nflpred.models.registry import default_phase1_models

    df = build_features(cfg)
    models = default_phase1_models(cfg)
    if args.models:
        wanted = set(args.models.split(","))
        models = [m for m in models if m.name in wanted]
        if not models:
            print(f"no models matched {args.models!r}", file=sys.stderr)
            return 2

    targets = args.targets.split(",") if getattr(args, "targets", None) else None
    models = default_phase1_models(cfg, targets=targets) if targets else models
    preds = run_backtest(df, models, cfg)
    out = cfg.artifacts
    out.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out / "backtest_predictions.parquet", index=False)

    tab = summarize(preds, cfg, model_order=[m.name for m in models], targets=targets)
    tab.to_csv(out / "backtest_summary.csv", index=False)
    print("\n" + "=" * 108)
    print(f"WALK-FORWARD OUT-OF-SAMPLE  |  {preds.season.min()} wk{preds[preds.season==preds.season.min()].week.min()}"
          f" - {preds.season.max()} wk{preds[preds.season==preds.season.max()].week.max()}"
          f"  |  {preds.game_id.nunique()} games")
    print("=" * 108)
    print(format_table(tab, cfg.get("evaluation.ats_breakeven")))

    bs = by_season(preds, cfg)
    bs.to_csv(out / "backtest_by_season.csv", index=False)
    if args.by_season:
        win = bs[bs["target"] == "win"]
        if not win.empty:
            print("\nper-season accuracy (win):")
            print(win.pivot(index="season", columns="model", values="accuracy").round(4).to_string())
    print(f"\nartifacts written to {out}/")
    return 0


def cmd_train(args, cfg: Config) -> int:
    from nflpred.features.build import build_features
    from nflpred.predict.week import save_models, train_production_models

    df = build_features(cfg)
    asof = pd.Timestamp.max if args.asof is None else pd.Timestamp(args.asof)
    models = train_production_models(cfg, df, asof)
    save_models(models, cfg.artifacts / "production_models.pkl")
    print(f"trained {len(models)} model(s): {', '.join(m.name for m in models)}")
    return 0


def cmd_predict_week(args, cfg: Config) -> int:
    from nflpred.predict.week import format_week, predict_week

    preds = predict_week(
        cfg, season=args.season, week=args.week, write_log=not args.no_log
    )
    season = int(preds["season"].iloc[0])
    week = int(preds["week"].iloc[0])
    print(f"\n{'='*100}\nPREDICTIONS - {season} week {week}\n{'='*100}")
    print(f"\nmodel: {args.model}\n")
    print(format_week(preds, args.model))
    if not args.no_log:
        print(f"\nlogged to {cfg.get('paths.prediction_log')} "
              f"({preds['model'].nunique()} models x {preds['game_id'].nunique()} games)")
    return 0


def cmd_score(args, cfg: Config) -> int:
    from nflpred.features.build import build_features
    from nflpred.predict.log import score_logged_predictions, verify_log

    bad = verify_log(cfg.get("paths.prediction_log"))
    if len(bad):
        print(f"WARNING: {len(bad)} logged row(s) fail their content hash "
              f"(edited after logging)", file=sys.stderr)

    df = build_features(cfg)
    perf = score_logged_predictions(
        cfg.get("paths.prediction_log"), df, cfg.get("paths.performance_log"), cfg
    )
    if perf.empty:
        print("nothing to score yet")
        return 0
    cols = [c for c in ["season", "week", "target", "model", "n_games", "accuracy",
                        "log_loss", "brier", "mae", "ats_pct"] if c in perf.columns]
    print(perf[cols].to_string(index=False))
    return 0


def cmd_audit(args, cfg: Config) -> int:
    from nflpred.features.build import build_features, feature_groups
    from nflpred.ingest.nflverse import cache_vintage
    from nflpred.predict.log import verify_log

    print(f"data vintage      : {cache_vintage(cfg)}")
    df = build_features(cfg)
    feats = [c for g in feature_groups(df).values() for c in g]
    played = df[df["played"]]
    print(f"feature table     : {len(df)} games, {len(feats)} features")
    print(f"completed games   : {len(played)}  ({played.season.min()}-{played.season.max()})")
    nulls = played[feats].isna().mean().sort_values(ascending=False)
    high = nulls[nulls > 0.05]
    print(f"features >5% null : {len(high)}")
    for c, v in high.head(10).items():
        print(f"    {c:40s} {v:.3f}")
    bad = verify_log(cfg.get("paths.prediction_log"))
    print(f"prediction log    : {'OK' if len(bad) == 0 else f'{len(bad)} TAMPERED ROWS'}")
    print("\nrun `pytest tests/` for the full leakage audit")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="nflpred", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--config", default="configs/default.yaml")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("fetch", help="download nflverse data")
    sp.add_argument("--force", action="store_true", help="re-download even if cached")
    sp.set_defaults(func=cmd_fetch)

    sp = sub.add_parser("build-features", help="build the modelling table")
    sp.add_argument("--force", action="store_true", help="rebuild even if cached")
    sp.set_defaults(func=cmd_build_features)

    sp = sub.add_parser("backtest", help="walk-forward evaluation")
    sp.add_argument("--models", help="comma-separated subset of model names")
    sp.add_argument("--by-season", action="store_true", help="also print per-season accuracy")
    sp.add_argument("--targets", help="comma-separated targets (default: all)")
    sp.set_defaults(func=cmd_backtest)

    sp = sub.add_parser("train", help="fit production models on all completed games")
    sp.add_argument("--asof", help="train only on games before this date")
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("predict-week", help="predict a week and log it")
    sp.add_argument("--season", type=int)
    sp.add_argument("--week", type=int)
    sp.add_argument("--model", default="lean_qb",
                    help="model to display (default: lean_qb, the best non-market "
                         "model on log loss, Brier and spread MAE)")
    sp.add_argument("--no-log", action="store_true", help="print without writing the log")
    sp.set_defaults(func=cmd_predict_week)

    sp = sub.add_parser("score", help="score previously logged predictions")
    sp.set_defaults(func=cmd_score)

    sp = sub.add_parser("audit", help="data and log health checks")
    sp.set_defaults(func=cmd_audit)

    args = p.parse_args(argv)
    _setup_logging(args.verbose)
    cfg = Config.load(args.config)
    return args.func(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
