"""
Fast end-to-end pipeline check for xau.py::run_wfo_pipeline. Runs the exact same
function the real sweep/full training uses, just with a small real-data slice and
tiny step counts, so plumbing bugs (env stepping, buffer purge, checkpointing, the
validation/test segment split) surface in seconds/minutes instead of after a multi-hour
real run.

Usage: python smoke_test.py
"""
import os
import shutil
import tempfile

import pandas as pd

from xau import run_wfo_pipeline

REAL_CSV = "data/data.csv"
RAW_ROW_LIMIT = 20_000  # ~14-15 weeks of M5 bars, enough for pretrain+val+test below


def build_small_csv(dest_dir: str) -> str:
    if not os.path.exists(REAL_CSV):
        raise SystemExit(f"{REAL_CSV} not found - smoke test needs a slice of real data to run against.")

    dest_path = os.path.join(dest_dir, "smoke_data.csv")
    df = pd.read_csv(REAL_CSV, nrows=RAW_ROW_LIMIT)
    df.to_csv(dest_path, index=False)
    return dest_path


def main():
    tmp_dir = tempfile.mkdtemp(prefix="xau_smoke_")
    try:
        smoke_csv = build_small_csv(tmp_dir)
        smoke_model_dir = os.path.join(tmp_dir, "models")

        print(f"Smoke-testing run_wfo_pipeline against {RAW_ROW_LIMIT} rows of real data in {tmp_dir}...")
        results = run_wfo_pipeline(
            csv_path=smoke_csv,
            pretrain_weeks_count=4,
            val_weeks_count=4,
            pretrain_timesteps=3000,
            gradient_steps=20,
            batch_size=32,
            buffer_size=256,
            replay_window_weeks=2,
            min_week_rows=20,
            seed=42,
            model_dir=smoke_model_dir,
        )

        print("\nResults dict keys:", list(results.keys()))

        assert "full" in results, "missing full backtest metrics"
        assert "validation" in results, "missing validation segment metrics - val/test split reporting is broken"
        assert "test" in results, (
            "missing test segment metrics - the smoke data slice may not cover enough "
            "weeks past the validation cutoff; try raising RAW_ROW_LIMIT"
        )

        pretrain_ckpt = os.path.join(smoke_model_dir, "sac_xauusd_pretrained.zip")
        assert os.path.exists(pretrain_ckpt), "pretrain checkpoint was not saved"

        week_checkpoints = [f for f in os.listdir(smoke_model_dir) if f.startswith("sac_xauusd_week_")]
        assert len(week_checkpoints) >= 1, "no per-week checkpoints were saved during the walk-forward loop"

        # Real trading activity must have happened, not just an always-flat policy -
        # otherwise the spread-cost, unrealized-pnl and downside-std code paths never ran.
        eq_curve = results["oos_equity_curve"]
        assert any(bal != eq_curve[0] for bal in eq_curve[1:]), (
            "equity curve never moved off the starting balance - policy stayed flat the "
            "whole run, so this smoke test isn't exercising real trading code paths "
            "(try raising pretrain_timesteps)"
        )

        # Segment boundary must reconcile with the full curve: validation's ending balance
        # is exactly where test starts, and test's ending balance is exactly the full
        # curve's ending balance - catches off-by-one/double-counting bugs in the split.
        full_end = results["full"]["end_balance"]
        val_end = results["validation"]["end_balance"]
        test_start = results["test"]["start_balance"]
        test_end = results["test"]["end_balance"]
        assert val_end == test_start, (
            f"validation segment end ({val_end}) does not match test segment start "
            f"({test_start}) - the val/test split boundary is miscounting balance"
        )
        assert test_end == full_end, (
            f"test segment end ({test_end}) does not match full curve end ({full_end}) - "
            f"the val/test split is dropping or double-counting weeks"
        )

        print(
            f"\nSMOKE TEST PASSED - {len(week_checkpoints)} weekly checkpoints saved, "
            f"full/validation/test metrics all present."
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
