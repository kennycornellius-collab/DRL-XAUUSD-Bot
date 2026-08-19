"""
Seed-ensembling evaluation (Next Priority #4): replays the walk-forward backtest driving
several already-trained seeds of ONE config at once, instead of trusting a single seed.
No training happens here - every seed's checkpoints already exist from a prior
run_sweep.py + run_seed_sweep.py run. At each step, every seed's deterministic action on
the current observation is combined (mean or median) into one action, which is what
actually steps the shared environment - so the two combine methods produce genuinely
different trajectories (not just different post-hoc aggregates of the same rollout) and
each needs its own full walk-forward pass.

Checkpoint alignment matters: run_wfo_pipeline (xau.py) processes week w by (1) rolling
out w with the CURRENT policy, (2) training on w's data, (3) saving that post-training
state as sac_xauusd_week_<w>.zip. So replaying week w must load the checkpoint saved at
the end of week w's *predecessor* (or sac_xauusd_pretrained.zip for the first
walk-forward week) - using week w's own checkpoint would replay w with a policy already
trained on w's own data (lookahead bias).

Reporting mirrors every prior era's Phase 1/Phase 2 protocol: validation-only ranking
between combine methods first, test revealed only for the validation-ranked winner.

--seed-sweep-dir must point at the exact timestamped run_seed_sweep.py output directory
being evaluated (as printed by that script, e.g. sweep_runs_seeds/20260819_195954) -
there is no bare-folder default, since a run_id-less "sweep_runs_seeds" no longer
identifies a single run. This script's own output (<winner-name>_ensemble_summary.json)
is written into --seed-sweep-dir alongside the seeds it evaluated, not a fresh
timestamp, since it isn't producing new trained artifacts.

--phase1-sweep-dir is only needed if run_seed_sweep.py reused an existing Phase 1 seed
42 (its default behavior) rather than training all seeds fresh via --no-reuse-seed42 -
in that case it must point at that Phase 1 run's exact directory, e.g.
sweep_runs/20260819_195817. Omit it when every seed (including 42) was trained by the
same run_seed_sweep.py invocation.

Usage:
    # Full 5-seed ensemble, both combine methods - seed 42 reused from Phase 1
    python ensemble_eval.py --winner-name 150k_500_256 \
        --phase1-sweep-dir sweep_runs/20260819_195817 \
        --seed-sweep-dir sweep_runs_seeds/20260819_195954

    # Same, but all 5 seeds (including 42) were trained together via
    # run_seed_sweep.py --no-reuse-seed42 - no --phase1-sweep-dir needed
    python ensemble_eval.py --winner-name 150k_500_256 \
        --seed-sweep-dir sweep_runs_seeds/20260820_090000

    # Single-seed replay check (see plan.md verification step 2) - a 1-seed "ensemble"
    # must reproduce that seed's already-recorded results.json numbers almost exactly
    python ensemble_eval.py --winner-name 150k_500_256 --seeds 43 --combine mean \
        --seed-sweep-dir sweep_runs_seeds/20260819_195954
"""
import argparse
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm

from xau import prepare_data, XAUEnv, rollout_week, build_segment_metrics, print_backtest_summary

REAL_CSV = "data/data.csv"


def seed_model_dir(winner_name: str, seed: int, phase1_sweep_dir: Optional[str], seed_sweep_dir: str) -> str:
    """run_seed_sweep.py produces checkpoints in one of two layouts depending on whether
    it reused seed 42 from an existing Phase 1 run (the default) or trained every seed
    fresh via --no-reuse-seed42: reused seed 42 lives under the Phase 1 sweep dir, while
    every seed trained by run_seed_sweep.py itself (including seed 42 when run fresh)
    lives under seed_sweep_dir/<winner_name>_seed<N>/. Prefer the seed_sweep_dir location
    whenever it actually exists there, and fall back to phase1_sweep_dir only for the
    reused-seed-42 case."""
    same_dir_path = os.path.join(seed_sweep_dir, f"{winner_name}_seed{seed}", "models")
    if os.path.exists(os.path.join(same_dir_path, "sac_xauusd_pretrained.zip")):
        return same_dir_path
    if phase1_sweep_dir is None:
        raise SystemExit(
            f"No checkpoint for seed {seed} under {same_dir_path}, and --phase1-sweep-dir "
            f"was not given to fall back to (needed for a reused seed 42)."
        )
    return os.path.join(phase1_sweep_dir, winner_name, "models")


def week_checkpoint_path(model_dir: str, prev_processed_week: Optional[str]) -> str:
    if prev_processed_week is None:
        return os.path.join(model_dir, "sac_xauusd_pretrained.zip")
    return os.path.join(model_dir, f"sac_xauusd_week_{prev_processed_week}.zip")


def combine_mean(actions: np.ndarray) -> np.ndarray:
    return actions.mean(axis=0)


def combine_median(actions: np.ndarray) -> np.ndarray:
    return np.median(actions, axis=0)


COMBINE_FNS = {"mean": combine_mean, "median": combine_median}


def run_ensemble(
    df,
    winner_name: str,
    seeds: List[int],
    combine: str,
    pretrain_weeks_count: int,
    val_weeks_count: int,
    min_week_rows: int,
    phase1_sweep_dir: Optional[str],
    seed_sweep_dir: str,
) -> Dict[str, Any]:
    weeks = sorted(df['week_label'].unique())
    walk_forward_weeks = weeks[pretrain_weeks_count:]
    validation_weeks = set(walk_forward_weeks[:val_weeks_count])
    test_weeks = set(walk_forward_weeks[val_weeks_count:])

    model_dirs = {seed: seed_model_dir(winner_name, seed, phase1_sweep_dir, seed_sweep_dir) for seed in seeds}
    for seed, m_dir in model_dirs.items():
        ckpt = os.path.join(m_dir, "sac_xauusd_pretrained.zip")
        if not os.path.exists(ckpt):
            raise SystemExit(f"Missing checkpoint for seed {seed}: {ckpt}")

    models = [SAC.load(os.path.join(model_dirs[seed], "sac_xauusd_pretrained.zip")) for seed in seeds]
    combine_fn = COMBINE_FNS[combine]

    oos_equity_curve = [10000.0]
    equity_curve_weeks: List[str] = []
    running_peak_balance = 10000.0
    trades: List[Dict[str, Any]] = []
    weekly_pnl: List[float] = []
    weekly_returns: List[float] = []
    week_open_prices: List[float] = []
    week_close_prices: List[float] = []

    prev_processed_week: Optional[str] = None
    for w in tqdm(walk_forward_weeks, desc=f"Ensemble replay ({combine})", unit="week"):
        w_df = df[df['week_label'] == w].copy()
        if len(w_df) < min_week_rows:
            continue

        for seed, model in zip(seeds, models):
            model.set_parameters(week_checkpoint_path(model_dirs[seed], prev_processed_week))

        def predict_fn(obs, _models=models):
            actions = np.stack([m.predict(obs, deterministic=True)[0] for m in _models], axis=0)
            return combine_fn(actions)

        wf_env = DummyVecEnv([lambda: XAUEnv(w_df, initial_balance=oos_equity_curve[-1], initial_peak_balance=running_peak_balance)])
        rollout = rollout_week(wf_env, predict_fn, week_label=w)

        ep_pnl = rollout["ep_pnl"]
        trades.extend(rollout["trades"])
        last_info = rollout["last_info"]

        start_of_week_balance = oos_equity_curve[-1]
        weekly_pnl.append(ep_pnl)
        weekly_returns.append(ep_pnl / start_of_week_balance if start_of_week_balance else 0.0)
        week_open_prices.append(float(w_df.iloc[0]['open']))
        week_close_prices.append(float(w_df.iloc[-1]['close']))

        oos_equity_curve.append(oos_equity_curve[-1] + ep_pnl)
        equity_curve_weeks.append(w)
        running_peak_balance = last_info['peak_balance']

        prev_processed_week = w

    full_metrics = build_segment_metrics(
        oos_equity_curve, weekly_returns, weekly_pnl, trades,
        start_price=week_open_prices[0] if week_open_prices else None,
        end_price=week_close_prices[-1] if week_close_prices else None,
    )

    val_curve = [oos_equity_curve[0]]
    val_returns: List[float] = []
    val_pnl: List[float] = []
    val_open_price: Optional[float] = None
    val_close_price: Optional[float] = None
    test_curve: Optional[List[float]] = None
    test_returns: List[float] = []
    test_pnl: List[float] = []
    test_open_price: Optional[float] = None
    test_close_price: Optional[float] = None
    for w, bal, wret, wpnl, wopen, wclose in zip(
        equity_curve_weeks, oos_equity_curve[1:], weekly_returns, weekly_pnl, week_open_prices, week_close_prices
    ):
        if w in validation_weeks:
            val_curve.append(bal)
            val_returns.append(wret)
            val_pnl.append(wpnl)
            if val_open_price is None:
                val_open_price = wopen
            val_close_price = wclose
        elif w in test_weeks:
            if test_curve is None:
                test_curve = [val_curve[-1]]
                test_open_price = wopen
            test_curve.append(bal)
            test_returns.append(wret)
            test_pnl.append(wpnl)
            test_close_price = wclose

    val_trades = [t for t in trades if t["week"] in validation_weeks]
    test_trades = [t for t in trades if t["week"] in test_weeks]

    results: Dict[str, Any] = {
        "seeds": seeds,
        "combine": combine,
        "oos_equity_curve": oos_equity_curve,
        "equity_curve_weeks": equity_curve_weeks,
        "full": full_metrics,
    }
    if len(val_curve) > 1:
        results["validation"] = build_segment_metrics(val_curve, val_returns, val_pnl, val_trades, val_open_price, val_close_price)
    if test_curve and len(test_curve) > 1:
        results["test"] = build_segment_metrics(test_curve, test_returns, test_pnl, test_trades, test_open_price, test_close_price)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--winner-name", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--combine", choices=["mean", "median", "both"], default="both")
    parser.add_argument("--csv", default=REAL_CSV)
    parser.add_argument("--phase1-sweep-dir", default=None, help="Exact timestamped run_sweep.py output directory holding a reused seed 42's checkpoints, e.g. sweep_runs/20260819_195817. Only needed if seed 42 isn't already present under --seed-sweep-dir (i.e. run_seed_sweep.py reused it from Phase 1 rather than training it fresh).")
    parser.add_argument("--seed-sweep-dir", required=True, help="Exact timestamped run_seed_sweep.py output directory holding seeds 43-46's checkpoints, e.g. sweep_runs_seeds/20260819_195954.")
    parser.add_argument("--pretrain-weeks-count", type=int, default=26)
    parser.add_argument("--val-weeks-count", type=int, default=79)
    parser.add_argument("--min-week-rows", type=int, default=50)
    args = parser.parse_args()

    combine_methods = ["mean", "median"] if args.combine == "both" else [args.combine]

    print(f"Loading data from {args.csv}...")
    df = prepare_data(args.csv)

    runs: Dict[str, Dict[str, Any]] = {}
    for combine in combine_methods:
        print(f"\n{'='*66}\nRunning ensemble replay: combine={combine}, seeds={args.seeds}\n{'='*66}")
        runs[combine] = run_ensemble(
            df, args.winner_name, args.seeds, combine,
            args.pretrain_weeks_count, args.val_weeks_count, args.min_week_rows,
            args.phase1_sweep_dir, args.seed_sweep_dir,
        )

    print("\n" + "="*66)
    print("VALIDATION RANKING (combine methods only - test not inspected yet)")
    print("="*66)
    for combine, r in runs.items():
        val = r.get("validation", {})
        print(f"{combine:<10} validation net_profit={val.get('net_profit_pct', float('nan')):+.2f}%  "
              f"max_dd={val.get('max_drawdown_pct', float('nan')):.2f}%")

    winner = max(runs.items(), key=lambda kv: kv[1].get("validation", {}).get("net_profit_pct", float("-inf")))
    winner_combine, winner_result = winner
    print(f"\nValidation winner: combine={winner_combine}")

    print_backtest_summary(f"ENSEMBLE VALIDATION ({winner_combine}, seeds={args.seeds})", winner_result["validation"])
    if "test" in winner_result:
        print_backtest_summary(f"ENSEMBLE TEST ({winner_combine}, seeds={args.seeds}) - only inspect once", winner_result["test"])

    os.makedirs(args.seed_sweep_dir, exist_ok=True)
    out_path = os.path.join(args.seed_sweep_dir, f"{args.winner_name}_ensemble_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"runs": runs, "validation_winner": winner_combine}, f, indent=2, default=str)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
