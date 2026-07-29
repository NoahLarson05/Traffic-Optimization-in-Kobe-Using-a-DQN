import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LEADERBOARD = "optimization_results.csv"
OUTPUT_DIR = "optimization_analysis"

PARAMS = {
    "w_delay": True,
    "w_halt": True,
    "w_wait": True,
    "w_arr": True,
    "lr": True,
    "gamma": False,
    "action_interval": False,
    "target_update": False,
    "batch_size": False,
}


def spearman(x, y):
    rx = pd.Series(x).rank()
    ry = pd.Series(y).rank()
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    df = pd.read_csv(LEADERBOARD)
    if "batch_size" not in df.columns:
        df["batch_size"] = 64
    df = df.sort_values("score").reset_index(drop=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n = len(df)
    best = df.iloc[0]
    print(f"{n} trials | best: trial {int(best['trial'])} score={best['score']:.4f}")

    rows = []
    for param, use_log in PARAMS.items():
        values = df[param].astype(float)
        if values.nunique() < 2:
            continue
        x = np.log10(values) if use_log else values
        rho = spearman(x, df["score"])
        rows.append({
            "parameter": param,
            "spearman_with_score": round(rho, 3),
            "abs_influence": round(abs(rho), 3),
            "direction": "higher is better" if rho < 0 else "lower is better",
        })

    imp = pd.DataFrame(rows).sort_values("abs_influence", ascending=False)
    imp.to_csv(os.path.join(OUTPUT_DIR, "importance.csv"), index=False)
    print("\nInfluence ranking (|Spearman correlation| with score):")
    print(imp.to_string(index=False))

    plt.figure(figsize=(9, 5))
    plt.barh(imp["parameter"][::-1], imp["abs_influence"][::-1])
    for y, (_, r) in enumerate(imp[::-1].iterrows()):
        plt.annotate(f' {r["direction"]}', (r["abs_influence"], y),
                     va="center", fontsize=9)
    plt.xlim(0, imp["abs_influence"].max() * 1.35)
    plt.xlabel("|Spearman correlation| with validation score")
    plt.title(f"Which parameters influence the DQN result most ({n} trials)")
    plt.grid(axis="x")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "00_importance.png"), dpi=300)
    plt.close()

    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for ax, (param, use_log) in zip(axes.flat, PARAMS.items()):
        ax.scatter(df[param], df["score"], alpha=0.5, s=18)
        ax.scatter([best[param]], [best["score"]], color="red", zorder=3, s=40)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        if use_log:
            ax.set_xscale("log")
        ax.set_title(param, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
    fig.suptitle(f"Validation score vs each parameter ({n} trials; "
                 "red = best trial; dashed = best reference controller)",
                 fontsize=12)
    fig.supylabel("score (lower = better)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "01_scatter_grid.png"), dpi=300)
    plt.close(fig)

    for param, use_log in PARAMS.items():
        plt.figure(figsize=(7, 5))
        plt.scatter(df[param], df["score"], alpha=0.6, label="trials")
        plt.scatter([best[param]], [best["score"]], color="red", zorder=3,
                    label=f"best (trial {int(best['trial'])})")
        plt.axhline(1.0, color="gray", linestyle="--", linewidth=1,
                    label="best reference controller")
        if use_log:
            plt.xscale("log")
        plt.xlabel(param)
        plt.ylabel("score (lower = better)")
        plt.title(f"score vs {param}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"scatter_{param}.png"), dpi=300)
        plt.close()

    top_n = max(1, n // 4)
    top = df.head(top_n)
    rest = df.tail(n - top_n)
    comparison = pd.DataFrame({
        "top25pct_median": top[list(PARAMS)].median().round(5),
        "rest_median": rest[list(PARAMS)].median().round(5),
    })
    comparison["ratio_top_vs_rest"] = (
        comparison["top25pct_median"] / comparison["rest_median"]
    ).round(3)
    comparison.to_csv(os.path.join(OUTPUT_DIR, "top_vs_rest.csv"))
    print(f"\nTop {top_n} trials vs the rest (median parameter values):")
    print(comparison.to_string())

    print(f"\nSaved analysis to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
