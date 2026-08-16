"""
Shared process-parallel execution engine used by both sweep drivers:
run_sweep.py (compares several hyperparameter configs) and run_seed_sweep.py (reruns one
chosen config across several random seeds to measure its result's seed sensitivity).
Both just build a list of run_specs (each a dict with name/pretrain_timesteps/
gradient_steps/batch_size/seed) and hand them to run_parallel, which runs each one as an
isolated OS subprocess (its own model directory, log file, and TensorBoard run name) so
several runs can train concurrently without clobbering each other's output.

NOTE: run_sweep.py currently has its own inline copy of run_one/the ProcessPoolExecutor
orchestration rather than importing from here, because it was mid-run (a live ~1.5-2h
background sweep) when this module was added and editing that file wasn't worth the risk.
De-duplicate run_sweep.py to import from here once no sweep is actively running against it.
"""
import concurrent.futures
import contextlib
import json
import os
import traceback
from typing import Any, Dict, List, Optional

from xau import run_wfo_pipeline


def run_one(
    run_spec: Dict[str, Any],
    csv_path: str,
    sweep_dir: str,
    num_torch_threads: Optional[int],
    extra_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    name = run_spec["name"]
    run_dir = os.path.join(sweep_dir, name)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "log.txt")
    model_dir = os.path.join(run_dir, "models")

    with open(log_path, "w", encoding="utf-8") as log_file, \
         contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
        try:
            results = run_wfo_pipeline(
                csv_path=csv_path,
                pretrain_timesteps=run_spec["pretrain_timesteps"],
                gradient_steps=run_spec["gradient_steps"],
                batch_size=run_spec["batch_size"],
                seed=run_spec["seed"],
                model_dir=model_dir,
                tb_log_name=f"SAC_Pretrain_{name}",
                num_torch_threads=num_torch_threads,
                **extra_kwargs,
            )
            results["config"] = run_spec
            results["status"] = "ok"
        except Exception as e:
            results = {
                "config": run_spec,
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def run_parallel(
    run_specs: List[Dict[str, Any]],
    csv_path: str,
    sweep_dir: str,
    max_parallel: int,
    extra_kwargs: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    os.makedirs(sweep_dir, exist_ok=True)
    logical_cores = os.cpu_count() or max_parallel
    num_torch_threads = max(1, logical_cores // max_parallel)

    print(f"Launching {len(run_specs)} run(s), max {max_parallel} in parallel, "
          f"{num_torch_threads} torch thread(s) per worker...")

    results_by_name: Dict[str, Dict[str, Any]] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_parallel) as executor:
        futures = {
            executor.submit(run_one, spec, csv_path, sweep_dir, num_torch_threads, extra_kwargs): spec["name"]
            for spec in run_specs
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
                    print(f"[ERROR] {name}: {result.get('error')}  (see {sweep_dir}/{name}/log.txt)")
            except Exception as e:
                print(f"[CRASH] {name}: {e}")
                result = {"config": {"name": name}, "status": "crash", "error": str(e)}
            results_by_name[name] = result

    with open(os.path.join(sweep_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(results_by_name, f, indent=2, default=str)

    return results_by_name
