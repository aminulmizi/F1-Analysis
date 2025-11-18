"""
Singapore 2025 - Race Lap Time vs Lap (stint trend chart)
CLEAN VERSION - Minimal noise, maximum clarity
Compatible with FastF1 v3.6.x
"""

import os
import sys
import fastf1
from fastf1.plotting import get_team_color
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline

# ---- USER CONFIG ----
YEAR = 2025
EVENT = "Singapore"
SESSION = "R"   # Race
DRIVERS = ["RUS", "VER", "NOR", "ANT", "LEC", "HAD"]
OUTPUT_PNG = "singapore_2025_race_trends_clean.png"
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "fastf1")
SMOOTH_WINDOW = 7   # Higher smoothing for cleaner lines
# ---------------------

# enable cache
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

# load session laps only (faster)
print(f"Loading {EVENT} {YEAR} race session...")
session = fastf1.get_session(YEAR, EVENT, SESSION)
session.load(laps=True, telemetry=False)
print("Session loaded.")

# helper to get safe team colour
def safe_team_color(laps, driver):
    try:
        # Override Antonelli's color to distinguish from Russell
        if driver == "ANT":
            return "#FCFCFC9B"  # Brighter teal/cyan for Antonelli
        team = laps.iloc[0]["Team"]
        return get_team_color(team, session)
    except Exception:
        return "#CCCCCC"

# collect driver laps DataFrames
driver_data = {}
for drv in DRIVERS:
    laps = session.laps.pick_drivers([drv])
    if laps.empty:
        print(f"Warning: no laps for {drv}, skipping.")
        continue

    laps = laps[laps["LapTime"].notna()].reset_index(drop=True)
    if laps.empty:
        print(f"Warning: {drv} has no valid lap times, skipping.")
        continue

    # Remove pit in/out laps for cleaner data (like Bahrain chart)
    laps = laps[(laps["PitInTime"].isna()) & (laps["PitOutTime"].isna())].reset_index(drop=True)
    
    if laps.empty:
        print(f"Warning: {drv} has no valid racing laps after removing pit laps, skipping.")
        continue

    if "Compound" not in laps.columns:
        laps["Compound"] = laps.get("TyreCompound", None)
    laps["Compound"] = laps["Compound"].fillna("UNKNOWN")

    # Create stint tracking based on compound changes
    laps = laps.copy()
    laps["StintCompound"] = (laps["Compound"] != laps["Compound"].shift()).cumsum()
    laps["StintID"] = laps["StintCompound"]

    driver_data[drv] = laps

if not driver_data:
    raise SystemExit("No driver lap data found.")

# ---- CLEAN PLOT (Bahrain style) ----
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(16, 8))
ax.set_facecolor("#0a0a0a")
fig.patch.set_facecolor("#0a0a0a")

# Tire compound symbols
COMPOUND_SYMBOLS = {
    "SOFT": "●",
    "MEDIUM": "●", 
    "HARD": "●",
}

for i, (drv, laps) in enumerate(driver_data.items()):
    color = safe_team_color(laps, drv)
    
    # Use dotted line for Russell to distinguish from Antonelli
    if drv == "RUS":
        driver_linestyle = ":"
        driver_linewidth = 3.0  # Thicker dotted line
    else:
        driver_linestyle = "-"
        driver_linewidth = 2.5
    
    for stint_id, stint in laps.groupby("StintID", sort=True):
        if len(stint) < 2:
            continue

        x = stint["LapNumber"].to_numpy()
        y = stint["LapTime"].dt.total_seconds().to_numpy()

        # Apply strong smoothing using rolling average
        y_smooth = (
            pd.Series(y)
            .rolling(window=SMOOTH_WINDOW, center=True, min_periods=1)
            .mean()
            .to_numpy()
        )
        
        comp = str(stint["Compound"].iloc[0]).upper()
        
        # Main clean line with driver-specific style
        ax.plot(
            x,
            y_smooth,
            color=color,
            linewidth=driver_linewidth,
            alpha=0.95,
            linestyle=driver_linestyle,
            label=drv if stint_id == laps["StintID"].unique()[0] else None,
            zorder=3
        )
        
        # No compound markers - keep it clean like Bahrain chart

# Single clean legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles,
    labels,
    title="Driver",
    loc="upper right",
    frameon=True,
    facecolor="#1a1a1a",
    edgecolor="#333",
    fontsize=11,
    title_fontsize=12,
)

# Clean title and labels
ax.set_title(
    f"{EVENT} {YEAR} - Race Pace Analysis",
    fontsize=16,
    weight="bold",
    pad=15,
    color="#ffffff"
)
ax.set_xlabel("Lap Number", fontsize=13)
ax.set_ylabel("Lap Time (s)", fontsize=13)

# Minimal grid
ax.grid(alpha=0.15, color="#333", linestyle="-", linewidth=0.5, zorder=0)

# Tighter y-axis to show fuel-load lap time reduction (like Bahrain)
all_times = pd.concat([df["LapTime"].dt.total_seconds() for df in driver_data.values()])

# Focus on clean racing laps only (exclude outliers)
race_pace = all_times[(all_times > 94) & (all_times < 102)]

if not race_pace.empty:
    # Tight range to emphasize the downward trend, but include lap 1
    ymin = race_pace.quantile(0.10) - 0.5  # 10th percentile with small margin
    ymax = race_pace.quantile(1) + 0.5  # 98th percentile to catch lap 1
else:
    ymin = all_times.min() - 0.5
    ymax = all_times.max() + 0.5

ax.set_ylim(ymin, ymax)
ax.set_xlim(left=0)

# Remove top and right spines for cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())

print(f"\n✓ Saved clean plot to: {os.path.abspath(OUTPUT_PNG)}")
print(f"  • Smoothing window: {SMOOTH_WINDOW}")
print(f"  • Tire compounds marked with ●")
plt.show()