"""
Phase 1 sweep driver for plan.md Problem 1 (OOS hyperparameter overfitting).

Runs multiple xau.py::run_wfo_pipeline configs concurrently as separate OS processes
(inter-run parallelism across independent runs) -- not intra-run environment
parallelism, which would fight WeeklyRollingBuffer's single-env week-tracking and the
cross-week balance/peak carry-over built for plan.md Problem 4.

Each config gets its own model_dir/tensorboard log name (no collisions) and its own
log file under sweep_runs/<name>/log.txt (stdout/stderr redirected there so 4 parallel
runs don't interleave in the terminal). Results are collected as JSON per run plus an
aggregated sweep_runs/summary.json, and a ranking table is printed using ONLY the
validation-segment metrics -- test-segment results are saved to disk but deliberately
not printed here, so they aren't glanced at before a winner is chosen (see plan.md
Problem 1's fix).

Usage:
    python run_sweep.py                      # real Phase 1 sweep, 4 parallel workers
    python run_sweep.py --max-parallel 2      # fewer concurrent workers
    python run_sweep.py --smoke               # tiny configs + tiny data slice, for
                                                 verifying the driver itself before a
                                                 real multi-hour run
"""
import argparse
import concurrent.futures
import contextlib
import json
import os
import time
import traceback
from typing import Any, Dict, List, Optional

from xau import run_wfo_pipeline

PHASE1_CONFIGS: List[Dict[str, Any]] = [
    {"name": "150k_500_256",  "pretrain_timesteps": 150_000, "gradient_steps": 500,  "batch_size": 256},
    {"name": "150k_1000_256", "pretrain_timesteps": 150_000, "gradient_steps": 1000, "batch_size": 256},
    {"name": "150k_2000_256", "pretrain_timesteps": 150_000, "gradient_steps": 2000, "batch_size": 256},
    {"name": "150k_1000_128", "pretrain_timesteps": 150_000, "gradient_steps": 1000, "batch_size": 128},
    {"name": "100k_1000_256", "pretrain_timesteps": 100_000, "gradient_steps": 1000, "batch_size": 256},
    {"name": "50k_1000_256",  "pretrain_timesteps": 50_000,  "gradient_steps": 1000, "batch_size": 256},
]

SMOKE_CONFIGS: List[Dict[str, Any]] = [
    {"name": "smoke_a", "pretrain_timesteps": 500, "gradient_steps": 10, "batch_size": 32},
    {"name": "smoke_b", "pretrain_timesteps": 500, "gradient_steps": 10, "batch_size": 32},
    {"name": "smoke_c", "pretrain_timesteps": 500, "gradient_steps": 10, "batch_size": 32},
]

REAL_CSV = "data/data.csv"


def run_one(
    config: Dict[str, Any],
    csv_path: str,
    sweep_dir: str,
    seed: int,
    num_torch_threads: Optional[int],
    extra_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    name = config["name"]
    run_dir = os.path.join(sweep_dir, name)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "log.txt")
    model_dir = os.path.join(run_dir, "models")

    with open(log_path, "w", encoding="utf-8") as log_file, \
         contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
        try:
            results = run_wfo_pipeline(
                csv_path=csv_path,
                pretrain_timesteps=config["pretrain_timesteps"],
                gradient_steps=config["gradient_steps"],
                batch_size=config["batch_size"],
                seed=seed,
                model_dir=model_dir,
                tb_log_name=f"SAC_Pretrain_{name}",
                num_torch_threads=num_torch_threads,
                **extra_kwargs,
            )
            results["config"] = config
            results["status"] = "ok"
        except Exception as e:
            results = {
                "config": config,
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", default=REAL_CSV)
    parser.add_argument("--sweep-dir", default="sweep_runs")
    parser.add_argument("--smoke", action="store_true", help="Run tiny configs against a small data slice to verify the driver itself.")
    args = parser.parse_args()

    if args.smoke:
        configs = SMOKE_CONFIGS
        extra_kwargs = dict(pretrain_weeks_count=4, val_weeks_count=4, buffer_size=256, replay_window_weeks=2, min_week_rows=20)
        csv_path = args.csv
    else:
        configs = PHASE1_CONFIGS
        extra_kwargs = {}
        csv_path = args.csv

    os.makedirs(args.sweep_dir, exist_ok=True)

    logical_cores = os.cpu_count() or args.max_parallel
    num_torch_threads = max(1, logical_cores // args.max_parallel)

    print(f"Launching {len(configs)} config(s), max {args.max_parallel} in parallel, "
          f"{num_torch_threads} torch thread(s) per worker...")
    started_at = time.time()

    results_by_name: Dict[str, Dict[str, Any]] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = {
            executor.submit(run_one, cfg, csv_path, args.sweep_dir, args.seed, num_torch_threads, extra_kwargs): cfg["name"]
            for cfg in configs
        }
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                if result.get("status") == "ok":
                    val = result.get("validation", {})
                    print(f"[DONE]  {name}: validation net_profit={val.get('net_profit_pct', float('nan')):+.2f}% "
                          f"max_dd={val.get('max_drawdown_pct', float('nan')):.2f}%")
                else:
                    print(f"[ERROR] {name}: {result.get('error')}  (see {args.sweep_dir}/{name}/log.txt)")
            except Exception as e:
                print(f"[CRASH] {name}: {e}")
                result = {"config": {"name": name}, "status": "crash", "error": str(e)}
            results_by_name[name] = result

    elapsed = time.time() - started_at
    print(f"\nAll runs finished in {elapsed/60:.1f} minutes.")

    with open(os.path.join(args.sweep_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(results_by_name, f, indent=2, default=str)

    print("\n" + "="*66)
    print("RANKING (validation segment only)")
    print("="*66)
    rows = []
    for name, result in results_by_name.items():
        if result.get("status") != "ok" or "validation" not in result:
            rows.append((name, None, None, "FAILED"))
            continue
        val = result["validation"]
        rows.append((name, val.get("net_profit_pct"), val.get("max_drawdown_pct"), "ok"))

    rows.sort(key=lambda r: (r[1] is None, -(r[1] if r[1] is not None else -1e9)))
    print(f"{'Config':<20}{'Net Profit %':>15}{'Max DD %':>12}{'Status':>10}")
    for name, np_pct, dd_pct, status in rows:
        np_str = f"{np_pct:+.2f}" if np_pct is not None else "N/A"
        dd_str = f"{dd_pct:.2f}" if dd_pct is not None else "N/A"
        print(f"{name:<20}{np_str:>15}{dd_str:>12}{status:>10}")
    print("="*66)
    print(f"\nFull per-run logs/results under ./{args.sweep_dir}/<config_name>/")
    print("Test-segment metrics are saved in each run's results.json but NOT printed above -")
    print("only look at those for the config chosen from this validation ranking.")


if __name__ == "__main__":
    main()
