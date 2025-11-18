"""
Singapore 2025 - Race Lap Time vs Lap (FUEL CORRECTED)
Copy this into VS Code and run. It will save a PNG and print a clickable file:// path.
Compatible with FastF1 v3.6.x

FUEL CORRECTION METHODOLOGY:
- Starting fuel: 110kg
- Race distance: 62 laps
- Fuel consumption: 110/62 = 1.774 kg/lap
- Performance impact: 10kg = 0.3s, therefore 1kg = 0.03s
- Lap N fuel advantage: N × 1.774kg × 0.03s/kg = N × 0.053s
- Corrected time: Actual lap time + fuel advantage (adds back the time saved)
"""

import os
import sys
import fastf1
from fastf1.plotting import get_team_color
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ---- USER CONFIG ----
YEAR = 2025
EVENT = "Singapore"
SESSION = "R"   # Race
DRIVERS = ["RUS", "VER", "NOR", "ANT", "LEC", "HAD"]
OUTPUT_PNG = "singapore_2025_race_fuel_corrected.png"
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "fastf1")
SMOOTH_WINDOW = 7

# FUEL CORRECTION PARAMETERS
STARTING_FUEL_KG = 110.0
TOTAL_LAPS = 62
FUEL_PER_LAP = STARTING_FUEL_KG / TOTAL_LAPS  # 1.774 kg/lap
PERFORMANCE_PER_KG = 0.03  # seconds per kg
# ---------------------

# enable cache
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

# load session laps only (faster)
print(f"Loading {EVENT} {YEAR} race session...")
session = fastf1.get_session(YEAR, EVENT, SESSION)
session.load(laps=True, telemetry=False)
print("Session loaded.")

print(f"\nFuel Correction Parameters:")
print(f"  Starting fuel: {STARTING_FUEL_KG}kg")
print(f"  Fuel per lap: {FUEL_PER_LAP:.3f}kg")
print(f"  Performance per kg: {PERFORMANCE_PER_KG}s")
print(f"  Correction per lap: {FUEL_PER_LAP * PERFORMANCE_PER_KG:.4f}s")

# helper to get safe team colour
def safe_team_color(laps, driver):
    try:
        # Override Antonelli's color to distinguish from Russell
        if driver == "ANT":
            return "#00D2BE"  # Brighter teal/cyan for Antonelli
        team = laps.iloc[0]["Team"]
        return get_team_color(team, session)
    except Exception:
        return "#CCCCCC"

# collect driver laps DataFrames
driver_data = {}
for drv in DRIVERS:
    # use pick_drivers (list) to avoid deprecation
    laps = session.laps.pick_drivers([drv])
    if laps.empty:
        print(f"Warning: no laps for {drv}, skipping.")
        continue

    # keep only laps with a LapTime
    laps = laps[laps["LapTime"].notna()].reset_index(drop=True)
    if laps.empty:
        print(f"Warning: {drv} has no valid lap times, skipping.")
        continue

    # Remove pit laps for cleaner data
    laps = laps[(laps["PitInTime"].isna()) & (laps["PitOutTime"].isna())].reset_index(drop=True)
    
    if laps.empty:
        print(f"Warning: {drv} has no valid racing laps after removing pit laps, skipping.")
        continue

    # ensure Compound column exists
    if "Compound" not in laps.columns:
        laps["Compound"] = laps.get("TyreCompound", None)
    laps["Compound"] = laps["Compound"].fillna("UNKNOWN")

    # Create stint tracking based on compound changes
    laps = laps.copy()
    laps["StintCompound"] = (laps["Compound"] != laps["Compound"].shift()).cumsum()
    laps["StintID"] = laps["StintCompound"]

    # CRITICAL: Apply fuel correction
    # Convert lap times to seconds
    laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()
    
    # Calculate fuel correction for each lap
    # Fuel burned by lap N = N × fuel_per_lap
    # Time advantage from fuel = fuel_burned × performance_per_kg
    # Corrected time = actual_time + time_advantage (normalize to full fuel)
    laps["FuelBurned"] = laps["LapNumber"] * FUEL_PER_LAP
    laps["FuelCorrection"] = laps["FuelBurned"] * PERFORMANCE_PER_KG
    laps["CorrectedLapTime"] = laps["LapTimeSeconds"] + laps["FuelCorrection"]
    
    print(f"\n{drv} - Sample lap corrections:")
    for idx in [0, len(laps)//2, -1]:
        if idx < len(laps):
            lap = laps.iloc[idx]
            print(f"  Lap {int(lap['LapNumber'])}: Raw={lap['LapTimeSeconds']:.3f}s, "
                  f"Correction=+{lap['FuelCorrection']:.3f}s, "
                  f"Corrected={lap['CorrectedLapTime']:.3f}s")

    driver_data[drv] = laps

if not driver_data:
    raise SystemExit("No driver lap data found. Check your FastF1 cache and driver codes.")

# ---- CLEAN PLOT ----
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_facecolor("#0c0c0c")
fig.patch.set_facecolor("#0c0c0c")

for i, (drv, laps) in enumerate(driver_data.items()):
    color = safe_team_color(laps, drv)
    
    # Use dotted line for Russell to distinguish from Antonelli
    if drv == "RUS":
        driver_linestyle = ":"
        driver_linewidth = 3.0
    else:
        driver_linestyle = "-"
        driver_linewidth = 2.5
    
    for stint_id, stint in laps.groupby("StintID", sort=True):
        if len(stint) < 2:
            continue

        x = stint["LapNumber"].to_numpy()
        # USE CORRECTED LAP TIMES
        y = stint["CorrectedLapTime"].to_numpy()

        y_smooth = (
            pd.Series(y)
            .rolling(window=SMOOTH_WINDOW, center=True, min_periods=1)
            .mean()
            .to_numpy()
        )

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

ax.set_title(
    f"{EVENT} {YEAR} - Race Pace Analysis (Fuel Corrected)",
    fontsize=15,
    weight="bold",
    pad=12,
)
ax.set_xlabel("Lap Number", fontsize=12)
ax.set_ylabel("Fuel-Corrected Lap Time (s)", fontsize=12)
ax.grid(alpha=0.15, color="#333", linestyle="-", linewidth=0.5, zorder=0)

# Calculate y-axis range from CORRECTED times
all_times = pd.concat([df["CorrectedLapTime"] for df in driver_data.values()])
race_pace = all_times[(all_times > 94) & (all_times < 110)]

if not race_pace.empty:
    ymin = race_pace.quantile(0.10) - 0.5
    ymax = race_pace.quantile(0.98) + 0.5
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

print(f"\nSaved fuel-corrected plot to: {os.path.abspath(OUTPUT_PNG)}")
print(f"Total fuel correction at race end: +{STARTING_FUEL_KG * PERFORMANCE_PER_KG:.2f}s")
plt.show()