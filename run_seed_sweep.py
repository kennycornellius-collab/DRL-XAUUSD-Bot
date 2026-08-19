"""
Seed-variance driver: reruns ONE already-chosen hyperparameter config (the winner picked
from run_sweep.py's validation-segment ranking) across multiple random seeds, to get a
mean +/- std estimate of validation and test performance instead of trusting a single
seed's outcome. DRL training is seed-sensitive, so a single run can be an outlier rather
than representative of a config's typical behavior - this quantifies that instead of
guessing at it. Uses the same inter-run process-parallel machinery as run_sweep.py (see
sweep_common.py).

The winner's seed=42 result already exists from its run_sweep.py run
(<phase1-sweep-dir>/<winner-name>/results.json) - by default this script loads that
instead of re-running seed 42, and only launches the additional seeds requested.
--phase1-sweep-dir must point at that Phase 1 run's exact timestamped directory (as
printed by run_sweep.py, e.g. sweep_runs/20260819_195817) - it is required, not
defaulted, because a bare "sweep_runs" no longer identifies a single run.

This run's own output goes under <seed-sweep-dir>/<run_id>/ (run_id defaults to the
launch timestamp), so two separate seed sweeps that happen to pick the same winning
config name can never collide - this bit a real run on 2026-08-19, when a later era's
Phase 2 silently overwrote an earlier era's trained checkpoints because both wrote to
the same config-name subdirectory with no run identifier.

Usage:
    # Auto-load hyperparams + reuse the existing seed=42 result from Phase 1
    python run_seed_sweep.py --winner-name 50k_1000_256 --phase1-sweep-dir sweep_runs/20260819_195817

    # Custom seed list (default is 43 44 45 46 -> 5 seeds total with the reused 42)
    python run_seed_sweep.py --winner-name 50k_1000_256 --phase1-sweep-dir sweep_runs/20260819_195817 --seeds 43 44 45 46 47

    # Run all seeds fresh instead of reusing Phase 1's seed=42 (e.g. config wasn't
    # part of Phase 1, or its sweep_runs/<run_id> has been archived/cleaned up)
    python run_seed_sweep.py --winner-name my_config --no-reuse-seed42 \
        --pretrain-timesteps 50000 --gradient-steps 1000 --batch-size 256 \
        --seeds 42 43 44 45 46
"""
import argparse
import json
import os
import statistics
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from sweep_common import run_parallel

REAL_CSV = "data/data.csv"


def load_phase1_result(winner_name: str, phase1_sweep_dir: str) -> Optional[Dict[str, Any]]:
    results_path = os.path.join(phase1_sweep_dir, winner_name, "results.json")
    if not os.path.exists(results_path):
        return None
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def mean_std(values: List[float]) -> Dict[str, Any]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
        "values": values,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--winner-name", required=True, help="Name for this config's seed runs (also the Phase 1 config directory name, if auto-loading hyperparams).")
    parser.add_argument("--phase1-sweep-dir", default=None, help="Exact timestamped run_sweep.py output directory to reuse seed 42 from, e.g. sweep_runs/20260819_195817 (required unless --no-reuse-seed42).")
    parser.add_argument("--seed-sweep-dir", default="sweep_runs_seeds")
    parser.add_argument("--run-id", default=None, help="Subdirectory name under --seed-sweep-dir for this run's own output. Defaults to the launch timestamp so separate runs can never collide.")
    parser.add_argument("--pretrain-timesteps", type=int, default=None)
    parser.add_argument("--gradient-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[43, 44, 45, 46])
    parser.add_argument("--reuse-seed42", dest="reuse_seed42", action="store_true", default=True)
    parser.add_argument("--no-reuse-seed42", dest="reuse_seed42", action="store_false")
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--csv", default=REAL_CSV)
    parser.add_argument("--pretrain-weeks-count", type=int, default=26)
    parser.add_argument("--val-weeks-count", type=int, default=79)
    parser.add_argument("--buffer-size", type=int, default=1920)
    parser.add_argument("--replay-window-weeks", type=int, default=4)
    parser.add_argument("--min-week-rows", type=int, default=50)
    args = parser.parse_args()

    pretrain_timesteps = args.pretrain_timesteps
    gradient_steps = args.gradient_steps
    batch_size = args.batch_size

    existing_seed42_result = None
    if args.reuse_seed42:
        if args.phase1_sweep_dir is None:
            raise SystemExit(
                "--phase1-sweep-dir is required when reusing seed 42 (the default) - pass the "
                "exact timestamped directory run_sweep.py printed for that Phase 1 run, e.g. "
                "sweep_runs/20260819_195817. Pass --no-reuse-seed42 instead to skip this and run "
                "all seeds fresh."
            )
        phase1_result = load_phase1_result(args.winner_name, args.phase1_sweep_dir)
        if phase1_result is None:
            raise SystemExit(
                f"Could not find {args.phase1_sweep_dir}/{args.winner_name}/results.json to reuse its "
                f"seed=42 result. Pass --no-reuse-seed42 plus explicit --pretrain-timesteps/"
                f"--gradient-steps/--batch-size (and include 42 in --seeds) to run all seeds fresh instead."
            )
        if phase1_result.get("status") != "ok":
            raise SystemExit(f"Phase 1 result for '{args.winner_name}' was not 'ok' (status={phase1_result.get('status')}) - can't reuse it.")
        cfg = phase1_result.get("config", {})
        pretrain_timesteps = pretrain_timesteps or cfg.get("pretrain_timesteps")
        gradient_steps = gradient_steps or cfg.get("gradient_steps")
        batch_size = batch_size or cfg.get("batch_size")
        existing_seed42_result = phase1_result

    if pretrain_timesteps is None or gradient_steps is None or batch_size is None:
        raise SystemExit("pretrain_timesteps/gradient_steps/batch_size must be known - either found via "
                          "--winner-name auto-load from Phase 1, or passed explicitly.")

    print(f"Config '{args.winner_name}': pretrain_timesteps={pretrain_timesteps} "
          f"gradient_steps={gradient_steps} batch_size={batch_size}")
    print(f"New seeds to run: {args.seeds}" +
          (" (+ reusing existing seed=42 result from Phase 1)" if existing_seed42_result else ""))

    run_specs = [
        {
            "name": f"{args.winner_name}_seed{seed}",
            "pretrain_timesteps": pretrain_timesteps,
            "gradient_steps": gradient_steps,
            "batch_size": batch_size,
            "seed": seed,
        }
        for seed in args.seeds
    ]

    extra_kwargs = {
        "pretrain_weeks_count": args.pretrain_weeks_count,
        "val_weeks_count": args.val_weeks_count,
        "buffer_size": args.buffer_size,
        "replay_window_weeks": args.replay_window_weeks,
        "min_week_rows": args.min_week_rows,
    }

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.seed_sweep_dir, run_id)
    print(f"Run directory: {run_dir}")

    started_at = time.time()
    results_by_name = run_parallel(
        run_specs=run_specs,
        csv_path=args.csv,
        sweep_dir=run_dir,
        max_parallel=args.max_parallel,
        extra_kwargs=extra_kwargs,
    )
    elapsed = time.time() - started_at
    print(f"\nNew seed runs finished in {elapsed/60:.1f} minutes.")

    all_results = dict(results_by_name)
    if existing_seed42_result is not None:
        all_results[f"{args.winner_name}_seed42_phase1"] = existing_seed42_result

    ok_results = {name: r for name, r in all_results.items() if r.get("status") == "ok"}
    failed = [name for name, r in all_results.items() if r.get("status") != "ok"]
    if failed:
        print(f"\nWARNING: {len(failed)} run(s) failed and are excluded from the aggregate: {failed}")

    print("\n" + "="*66)
    print(f"SEED VARIANCE SUMMARY for '{args.winner_name}' (n={len(ok_results)} seeds)")
    print("="*66)

    aggregate: Dict[str, Any] = {}
    for segment in ("validation", "test"):
        profits = [r[segment]["net_profit_pct"] for r in ok_results.values() if segment in r]
        dds = [r[segment]["max_drawdown_pct"] for r in ok_results.values() if segment in r]
        if not profits:
            print(f"{segment.upper()}: no data")
            continue
        p_stats = mean_std(profits)
        d_stats = mean_std(dds)
        aggregate[segment] = {"net_profit_pct": p_stats, "max_drawdown_pct": d_stats}
        print(f"{segment.upper()} (n={p_stats['n']}):")
        print(f"  Net Profit %:   mean={p_stats['mean']:+.2f}  std={p_stats['std']:.2f}  "
              f"values={[f'{v:+.2f}' for v in p_stats['values']]}")
        print(f"  Max Drawdown %: mean={d_stats['mean']:.2f}  std={d_stats['std']:.2f}  "
              f"values={[f'{v:.2f}' for v in d_stats['values']]}")
    print("="*66)

    os.makedirs(run_dir, exist_ok=True)
    summary_path = os.path.join(run_dir, f"{args.winner_name}_seed_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {"pretrain_timesteps": pretrain_timesteps, "gradient_steps": gradient_steps, "batch_size": batch_size},
            "aggregate": aggregate,
            "runs": all_results,
        }, f, indent=2, default=str)
    print(f"\nFull per-seed results + this summary saved to {summary_path}")
    phase1_hint = f"--phase1-sweep-dir {args.phase1_sweep_dir}  " if args.reuse_seed42 else ""
    print(f"For a follow-up ensemble_eval.py call, pass:  {phase1_hint}--seed-sweep-dir {run_dir}")


if __name__ == "__main__":
    main()
