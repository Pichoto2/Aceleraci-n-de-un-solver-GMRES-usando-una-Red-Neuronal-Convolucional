import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

LOG = "gpu_log.csv"
OUTPUT = "gpu_performance.png"

# --- Load and prepare data ---
df = pd.read_csv(LOG, comment="#")
df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

# t0 = first instant when ANY GPU has utilization > 0
start = df[df["gpu_util_pct"] > 0]
if start.empty:
    raise SystemExit("No GPU utilization > 0 detected in the log.")

t0 = start["timestamp"].iloc[0]
df["t_rel_s"] = df["timestamp"] - t0
df = df[df["t_rel_s"] >= 0]

print("t = 0 set at first detected GPU activity")

# --- Publication-quality styling ---
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 300
})

fig, ax = plt.subplots(figsize=(8, 5))

# --- Plot data ---
for idx in sorted(df["idx"].unique()):
    g = df[df["idx"] == idx]

    ax.plot(
        g["t_rel_s"],
        g["gpu_util_pct"],
        label=f"GPU {idx} Utilization (%)",
        linewidth=1.8
    )

    ax.plot(
        g["t_rel_s"],
        g["mem_used_pct"],
        linestyle="--",
        label=f"GPU {idx} Memory Usage (%)",
        linewidth=1.6
    )

# --- Labels and formatting ---
ax.set_xlabel("Time since first GPU activity (s)")
ax.set_ylabel("Percentage (%)")
ax.set_title("GPU Performance Over Time")

ax.set_ylim(0, 100)
ax.grid(True, which="both", linestyle=":", linewidth=0.6)

ax.legend(frameon=True)
fig.tight_layout()

# --- Save figure ---
fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
print(f"Figure saved as {OUTPUT}")

