"""Reproducible paired analyses for Adam-NV experiments E13--E15.

Primary outcome: mean validation accuracy over the final five epochs.
All optimizer contrasts are paired by seed. The output reports the paired
mean difference, a t-based 95% confidence interval, a paired t-test p-value,
an exact sign-flip permutation p-value, and Holm-adjusted t-test p-values
within each experiment family.

The script refuses publication-mode output when any required run is missing
or incomplete. Pass --allow-incomplete only for exploratory diagnostics.
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats


EXPECTED_EPOCHS = 60
TAIL_EPOCHS = 5
SCHEDULES = ("constant", "phased", "ramp_up", "intermittent")


def trailing_accuracy(path: Path, expected_epochs: int = EXPECTED_EPOCHS) -> float:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {"epoch", "val_acc"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if len(frame) != expected_epochs or int(frame["epoch"].iloc[-1]) != expected_epochs:
        raise ValueError(
            f"{path}: incomplete run ({len(frame)} rows, final epoch "
            f"{frame['epoch'].iloc[-1] if len(frame) else 'none'}; expected {expected_epochs})"
        )
    return float(frame["val_acc"].tail(TAIL_EPOCHS).mean())


def load_seeds(
    path_for_seed: Callable[[int], Path],
    seeds: list[int],
    issues: list[str],
) -> dict[int, float]:
    values: dict[int, float] = {}
    for seed in seeds:
        path = path_for_seed(seed)
        try:
            values[seed] = trailing_accuracy(path)
        except (FileNotFoundError, ValueError) as exc:
            issues.append(str(exc))
    return values


def exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    """Two-sided exact paired randomization test under sign exchangeability."""
    observed = abs(float(differences.mean()))
    means = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        means.append(abs(float(np.mean(differences * np.asarray(signs)))))
    return float(np.mean(np.asarray(means) >= observed - 1e-15))


def paired_result(
    experiment: str,
    comparison: str,
    left_label: str,
    right_label: str,
    left: dict[int, float],
    right: dict[int, float],
    expected_seeds: list[int],
) -> dict[str, object]:
    common = sorted(set(left) & set(right))
    if len(common) < 2:
        raise ValueError(f"{comparison}: fewer than two complete paired seeds")
    differences = np.asarray([left[s] - right[s] for s in common], dtype=float)
    n = len(differences)
    mean = float(differences.mean())
    sd = float(differences.std(ddof=1))
    se = sd / math.sqrt(n)
    critical = float(stats.t.ppf(0.975, df=n - 1))
    t_result = stats.ttest_rel(
        [left[s] for s in common],
        [right[s] for s in common],
    )
    dz = mean / sd if sd > 0 else (math.inf if mean != 0 else 0.0)
    return {
        "experiment": experiment,
        "comparison": comparison,
        "left": left_label,
        "right": right_label,
        "metric": "trailing_5_epoch_val_accuracy",
        "expected_seeds": " ".join(map(str, expected_seeds)),
        "paired_seeds": " ".join(map(str, common)),
        "complete": common == expected_seeds,
        "n_pairs": n,
        "left_mean": float(np.mean([left[s] for s in common])),
        "right_mean": float(np.mean([right[s] for s in common])),
        "mean_difference": mean,
        "mean_difference_percentage_points": 100.0 * mean,
        "ci95_low": mean - critical * se,
        "ci95_high": mean + critical * se,
        "paired_cohen_dz": dz,
        "p_paired_t_raw": float(t_result.pvalue),
        "p_sign_flip_exact": exact_sign_flip_pvalue(differences),
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm family-wise error correction, returned in original order."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (m - rank) * p_values[index]
        running = max(running, candidate)
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "e2_noisy_gradient" / "logs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "e13_e15_primary_statistics.csv",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Analyze available pairs despite incomplete runs; exploratory use only.",
    )
    args = parser.parse_args()

    root = args.logs_root
    issues: list[str] = []
    results: list[dict[str, object]] = []

    # E13: Adam-NV versus Adam, paired by seed. The expanded intermittent
    # rerun has eight seeds; the other schedules have the original four.
    e13_arms: dict[str, dict[str, dict[int, float]]] = {}
    for schedule in ("phased", "ramp_up"):
        seeds = list(range(4))
        folder = root / "dynamic_noise"
        e13_arms[schedule] = {
            optimizer: load_seeds(
                lambda seed, optimizer=optimizer, schedule=schedule, folder=folder:
                    folder / f"{optimizer}_schedule{schedule}_sigma2.0_seed{seed}.csv",
                seeds,
                issues,
            )
            for optimizer in ("adam_nv", "adam")
        }
        results.append(
            paired_result(
                "E13", f"{schedule}: Adam-NV minus Adam", "Adam-NV", "Adam",
                e13_arms[schedule]["adam_nv"], e13_arms[schedule]["adam"], seeds,
            )
        )

    schedule = "intermittent"
    seeds = list(range(8))
    folder = root / "Intermittent"
    e13_arms[schedule] = {
        optimizer: load_seeds(
            lambda seed, optimizer=optimizer, folder=folder:
                folder / f"{optimizer}_scheduleintermittent_sigma2.0_seed{seed}.csv",
            seeds,
            issues,
        )
        for optimizer in ("adam_nv", "adam")
    }
    results.append(
        paired_result(
            "E13", "intermittent: Adam-NV minus Adam", "Adam-NV", "Adam",
            e13_arms[schedule]["adam_nv"], e13_arms[schedule]["adam"], seeds,
        )
    )

    # Direct schedule-by-optimizer interaction tests: compare the paired
    # Adam-NV-minus-Adam gap between schedules using their common seeds.
    for first, second in itertools.combinations(("phased", "ramp_up", "intermittent"), 2):
        first_gap = {
            seed: e13_arms[first]["adam_nv"][seed] - e13_arms[first]["adam"][seed]
            for seed in set(e13_arms[first]["adam_nv"]) & set(e13_arms[first]["adam"])
        }
        second_gap = {
            seed: e13_arms[second]["adam_nv"][seed] - e13_arms[second]["adam"][seed]
            for seed in set(e13_arms[second]["adam_nv"]) & set(e13_arms[second]["adam"])
        }
        expected = sorted(set(first_gap) & set(second_gap))
        results.append(
            paired_result(
                "E13_interaction",
                f"optimizer gap: {first} minus {second}",
                f"Adam-NV−Adam gap ({first})",
                f"Adam-NV−Adam gap ({second})",
                first_gap,
                second_gap,
                expected,
            )
        )

    # Default coupled beta2=0.999 arms. Constant uses the equivalent gamma=1
    # reruns for seeds 2/3 because the original Part-B folder has only 0/1.
    default_arms: dict[str, dict[int, float]] = {}
    for schedule in SCHEDULES:
        seeds = list(range(4))
        if schedule == "constant":
            def default_path(seed: int) -> Path:
                if seed < 2:
                    return root / "part_b" / f"adam_nv_sigma2.0_seed{seed}.csv"
                return root / "Ablation" / f"adam_nv_sigma2.0_gamma1.0_seed{seed}.csv"
        else:
            def default_path(seed: int, schedule: str = schedule) -> Path:
                return root / "dynamic_noise" / f"adam_nv_schedule{schedule}_sigma2.0_seed{seed}.csv"
        default_arms[schedule] = load_seeds(default_path, seeds, issues)

    coupled: dict[tuple[str, float], dict[int, float]] = {}
    decoupled: dict[tuple[str, float], dict[int, float]] = {}
    for schedule in SCHEDULES:
        seeds = list(range(4))
        for beta in (0.99, 0.95):
            coupled[(schedule, beta)] = load_seeds(
                lambda seed, schedule=schedule, beta=beta:
                    root / "response_lag" /
                    f"adam_nv_schedule{schedule}_sigma2.0_beta2{beta}_seed{seed}.csv",
                seeds,
                issues,
            )
            decoupled[(schedule, beta)] = load_seeds(
                lambda seed, schedule=schedule, beta=beta:
                    root / "decoupling" /
                    f"adam_nv_schedule{schedule}_sigma2.0_beta2noise{beta}_seed{seed}.csv",
                seeds,
                issues,
            )

            results.append(
                paired_result(
                    "E14",
                    f"{schedule}: coupled beta2={beta} minus coupled beta2=0.999",
                    f"coupled beta2={beta}",
                    "coupled beta2=0.999",
                    coupled[(schedule, beta)],
                    default_arms[schedule],
                    seeds,
                )
            )
            results.append(
                paired_result(
                    "E15",
                    f"{schedule}: decoupled beta2_noise={beta} minus coupled beta2={beta}",
                    f"decoupled beta2_noise={beta}",
                    f"coupled beta2={beta}",
                    decoupled[(schedule, beta)],
                    coupled[(schedule, beta)],
                    seeds,
                )
            )

    frame = pd.DataFrame(results)
    frame["p_paired_t_holm"] = np.nan
    for experiment, indices in frame.groupby("experiment").groups.items():
        raw = frame.loc[list(indices), "p_paired_t_raw"].tolist()
        frame.loc[list(indices), "p_paired_t_holm"] = holm_adjust(raw)

    if issues:
        print("Run-completeness issues:")
        for issue in sorted(set(issues)):
            print(f"  - {issue}")
    incomplete_rows = frame[~frame["complete"]]
    if (issues or len(incomplete_rows)) and not args.allow_incomplete:
        raise SystemExit(
            "Refusing publication-mode output because required runs are missing "
            "or incomplete. Complete the runs, or use --allow-incomplete for an "
            "explicitly exploratory table."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame)} comparisons to {args.output}")
    if issues or len(incomplete_rows):
        print("WARNING: exploratory output contains incomplete comparisons.")


if __name__ == "__main__":
    main()
