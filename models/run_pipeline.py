"""Rerun cross-validation and retrain the saved models at every horizon, in one command.

Run from the models/ folder after editing model_config.toml (or updating the data):
    python run_pipeline.py                          # both hybrids, all horizons
    python run_pipeline.py --models hybrid_lr       # one hybrid
    python run_pipeline.py --horizons 1             # one horizon
    python run_pipeline.py --skip-cv                # only retrain the saved models
    python run_pipeline.py --rebuild-table          # rebuild the weekly table first (new data)

For each horizon it runs hybrid_escalation/train.py (cross-validation: metrics and
predictions) and save_final_models.py (saved models), each in its own process because
the horizon is read once when the code is imported (common/config.py). It prints how
long each step took.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

MODELS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(MODELS_ROOT))

from common.model_config import load_model_config  # noqa: E402

MODEL_NAMES = ("hybrid_lr", "hybrid_xgb")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command-line options.

    Args:
        argv (list[str] | None): Arguments; sys.argv[1:] if None.

    Returns:
        argparse.Namespace: horizons, models, skip_cv, skip_save and rebuild_table.

    Example:
        >>> parse_args(["--horizons", "1", "4", "--skip-cv"]).horizons
        [1, 4]
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--horizons", type=int, nargs="+", default=None,
                        help="weeks ahead (default: [run] horizons in model_config.toml)")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--skip-cv", action="store_true", help="do not rerun cross-validation")
    parser.add_argument("--skip-save", action="store_true", help="do not retrain the saved models")
    parser.add_argument("--rebuild-table", action="store_true", help="rebuild the weekly table from data/ first")
    args = parser.parse_args(argv)
    args.horizons = args.horizons or load_model_config()["run"]["horizons"]
    return args


def list_steps(args: argparse.Namespace) -> list[tuple[int, str, list[str]]]:
    """List the commands to run, in order.

    Args:
        args (argparse.Namespace): Output of parse_args.

    Returns:
        list[tuple[int, str, list[str]]]: (horizon, step name, command) per step.

    Example:
        >>> [(h, s) for h, s, _ in list_steps(parse_args(["--horizons", "2", "--skip-save"]))]
        [(2, 'cross_validation')]
    """
    py = sys.executable
    steps = []
    for h in args.horizons:
        if args.rebuild_table:
            steps.append((h, "weekly_table",
                          [py, "-c", "from common.weekly import build_weekly; build_weekly(rebuild=True)"]))
        if not args.skip_cv:
            steps.append((h, "cross_validation", [py, "hybrid_escalation/train.py", *args.models]))
        if not args.skip_save:
            steps.append((h, "saved_models", [py, "save_final_models.py", *args.models]))
    return steps


def run_steps(steps: list[tuple[int, str, list[str]]]) -> list[tuple[int, str, float]]:
    """Run each step from the models/ folder with HAB_HORIZON set, stopping at the first failure.

    Args:
        steps (list[tuple[int, str, list[str]]]): Output of list_steps.

    Returns:
        list[tuple[int, str, float]]: (horizon, step name, seconds) per step.

    Raises:
        subprocess.CalledProcessError: If a step fails.
    """
    times = []
    for h, name, cmd in steps:
        print(f"\n### {h} week(s) ahead: {name}", flush=True)
        start = time.perf_counter()
        subprocess.run(cmd, cwd=MODELS_ROOT, env={**os.environ, "HAB_HORIZON": str(h)}, check=True)
        times.append((h, name, time.perf_counter() - start))
    return times


if __name__ == "__main__":
    times = run_steps(list_steps(parse_args()))
    print("\nRun times:")
    for h, name, sec in times:
        print(f"  {h} week(s) ahead  {name:<17} {int(sec // 60)} min {sec % 60:.0f} s")
