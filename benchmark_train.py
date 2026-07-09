import copy
# Z: subprocess runs an external process from actual script
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "detection" / "train_config.yaml"
RESULTS_ROOT = ROOT_DIR / "results" / "detect"


# Edit this list to define the benchmark grid.
# Keys use dot notation to target nested config values.
EXPERIMENTS = [
    {
        "model.family": "dinov3",
        "model.size": "small",
    },
    {
        "model.family": "dinov3",
        "model.size": "plus",
    },
    {"model.family": "dinov3", "model.size": "small", "training.batch": 64},
]


def load_yaml(path: Path) -> dict:
    """Z: Load a YAML file and return its contents as a dictionary."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_yaml(path: Path, data: dict) -> None:
    """Z: Save a dictionary to a YAML file."""
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def set_nested_value(data: dict, key_path: str, value) -> None:
    """Z: Set a value in a nested dictionary using a dot-separated key path."""
    # Z: split keys
    keys = key_path.split(".")
    current = data
    # Z: last key is the one we want to set, the rest are intermediate dicts
    for key in keys[:-1]:
        # Z: if the key doesn't exist or isn't a dict, create an empty dict
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        # Z: move to the next level
        current = current[key]
    # Z: set the final value
    current[keys[-1]] = value


def apply_overrides(base_config: dict, overrides: dict) -> dict:
    """Z: Create a new config by applying overrides to the base config."""
    config = copy.deepcopy(base_config)
    for key_path, value in overrides.items():
        set_nested_value(config, key_path, value)
    return config


def run_training() -> int:
    """Z: Run a training and return exit code."""
    command = [sys.executable, "main.py", "train", "--config", str(CONFIG_PATH)]
    process = subprocess.run(command, cwd=ROOT_DIR)
    return process.returncode


def load_metrics_history(metrics_path: Path) -> list[dict]:
    """Z: Load the metrics history from a .npy file and return it as a list of dicts."""
    # Z: allow_pickle is needed to load lists of dicts
    metrics = np.load(metrics_path, allow_pickle=True)
    if isinstance(metrics, np.ndarray):
        # Z: if 0-dim array, one python object inside, use item()
        if metrics.ndim == 0:
            metrics = metrics.item()
        # Z: otherwise, convert to list of dicts
        else:
            metrics = metrics.tolist()
    if not isinstance(metrics, list):
        raise ValueError(f"Unsupported metrics format in {metrics_path}")
    # Z: return only dict entries
    return [entry for entry in metrics if isinstance(entry, dict)]


def load_best_metrics(best_metrics_path: Path) -> dict | None:
    """Z: Load the best metrics from a .npy file and return it as a dict, or None if not found."""
    if not best_metrics_path.exists():
        return None

    metrics = np.load(best_metrics_path, allow_pickle=True)
    if isinstance(metrics, np.ndarray) and metrics.ndim == 0:
        metrics = metrics.item()
    if not isinstance(metrics, dict):
        raise ValueError(f"Unsupported best metrics format in {best_metrics_path}")
    return metrics


def collect_run_summaries(results_root: Path) -> list[dict]:
    """Z: Collect and return summaries of all benchmark runs
    by reading their metrics history and best metrics.
    """
    run_summaries = []
    # Z: for each metrics.npy file
    for metrics_path in sorted(results_root.glob("*/*/metrics.npy")):
        history = load_metrics_history(metrics_path)
        if not history:
            continue

        run_dir = metrics_path.parent
        best_metrics = load_best_metrics(run_dir / "best_metric.npy")
        map_50_history = [float(entry["map_50"]) for entry in history if "map_50" in entry]
        map_50_95_history = [float(entry["map_50_95"]) for entry in history if "map_50_95" in entry]
        epochs = [int(entry.get("epoch", index + 1)) for index, entry in enumerate(history)]

        if not map_50_history or not map_50_95_history:
            continue

        if best_metrics is not None:
            best_map_50 = float(best_metrics["map_50"])
            best_map_50_95 = float(best_metrics["map_50_95"])
        else:
            best_map_50 = max(map_50_history)
            best_map_50_95 = max(map_50_95_history)

        run_summaries.append(
            {
                "label": f"{run_dir.parent.name}/{run_dir.name}",
                "run_dir": run_dir,
                "epochs": epochs,
                "map_50_history": map_50_history,
                "map_50_95_history": map_50_95_history,
                "best_map_50": best_map_50,
                "best_map_50_95": best_map_50_95,
            }
        )
    return run_summaries


def print_benchmark_summary(run_summaries: list[dict]) -> None:
    if not run_summaries:
        print("\nNo benchmark runs with metrics were found under results/detect.")
        return

    print("\nBenchmark summary:")
    for run_summary in run_summaries:
        print(f"  {run_summary['label']}: best map50={run_summary['best_map_50']:.4f}, best map_50_95={run_summary['best_map_50_95']:.4f}")


def save_metric_plot(run_summaries: list[dict], metric_key: str, output_path: Path, ylabel: str, title: str) -> None:
    if not run_summaries:
        return

    plt.figure(figsize=(12, 7))
    for run_summary in run_summaries:
        plt.plot(
            run_summary["epochs"],
            run_summary[metric_key],
            label=run_summary["label"],
            linewidth=1.8,
        )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle=":")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def summarize_benchmarks(results_root: Path) -> None:
    """Z: Collect summaries of all benchmark runs and print a summary table,
    then save plots of mAP@50 and mAP@50:95 history for each run"""
    run_summaries = collect_run_summaries(results_root)
    print_benchmark_summary(run_summaries)
    save_metric_plot(
        run_summaries=run_summaries,
        metric_key="map_50_history",
        output_path=results_root / "benchmark_map50.png",
        ylabel="mAP@50",
        title="Benchmark summary: mAP@50 by run",
    )
    save_metric_plot(
        run_summaries=run_summaries,
        metric_key="map_50_95_history",
        output_path=results_root / "benchmark_map50_95.png",
        ylabel="mAP@50:95",
        title="Benchmark summary: mAP@50:95 by run",
    )


def main() -> int:
    """Z: Main function to run the benchmark training."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config file: {CONFIG_PATH}")

    base_config = load_yaml(CONFIG_PATH)
    original_config_text = CONFIG_PATH.read_text(encoding="utf-8")

    try:
        for index, experiment in enumerate(EXPERIMENTS, start=1):
            config = apply_overrides(base_config, experiment)

            print(f"\n[{index}/{len(EXPERIMENTS)}] Running benchmark: ")
            if experiment:
                for key_path, value in experiment.items():
                    print(f"  {key_path} = {value}")
            else:
                print("  no overrides")

            # Z: save the modified config for this run to CONFIG_PATH
            save_yaml(CONFIG_PATH, config)
            return_code = run_training()

            if return_code != 0:
                print(f"Training failed for benchmark with exit code {return_code}")
                return return_code

        summarize_benchmarks(RESULTS_ROOT)
        return 0
    # Z: we want to ensure that the original config is restored even if something goes wrong
    finally:
        CONFIG_PATH.write_text(original_config_text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
