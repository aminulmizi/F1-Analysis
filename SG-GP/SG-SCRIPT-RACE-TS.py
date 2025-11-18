"""
Singapore 2025 - Top Speed per Lap Analysis
Shows if Russell was slower on straights vs corners
Compatible with FastF1 v3.6.x
"""

import os
import fastf1
from fastf1.plotting import get_team_color
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ---- USER CONFIG ----
YEAR = 2025
EVENT = "Singapore"
SESSION = "R"
DRIVERS = ["RUS", "VER", "LEC", "ANT", "HAD" ]
OUTPUT_PNG = "singapore_2025_top_speed_analysis.png"
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "fastf1")
SMOOTH_WINDOW = 3
# ---------------------

os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

print(f"Loading {EVENT} {YEAR} race session with telemetry...")
session = fastf1.get_session(YEAR, EVENT, SESSION)
# Load telemetry this time to get speed data
session.load(laps=True, telemetry=True)
print("Session loaded.")

# Custom color for Antonelli
def safe_team_color(laps, driver):
    try:
        if driver == "ANT":
            return "#00D2BE"
        team = laps.iloc[0]["Team"]
        return get_team_color(team, session)
    except Exception:
        return "#CCCCCC"

# Collect top speed per lap for each driver
driver_speed_data = {}

for drv in DRIVERS:
    print(f"Processing {drv}...")
    laps = session.laps.pick_drivers([drv])
    
    if laps.empty:
        print(f"  Warning: no laps for {drv}")
        continue
    
    # Remove pit laps
    laps = laps[(laps["PitInTime"].isna()) & (laps["PitOutTime"].isna())]
    
    if laps.empty:
        print(f"  Warning: no valid laps for {drv}")
        continue
    
    lap_numbers = []
    top_speeds = []
    
    # Get top speed for each lap
    for idx, lap in laps.iterrows():
        try:
            telemetry = lap.get_telemetry()
            if telemetry is not None and not telemetry.empty and 'Speed' in telemetry.columns:
                max_speed = telemetry['Speed'].max()
                if pd.notna(max_speed) and max_speed > 0:
                    lap_numbers.append(lap['LapNumber'])
                    top_speeds.append(max_speed)
        except Exception as e:
            # Skip laps with telemetry issues
            continue
    
    if lap_numbers:
        driver_speed_data[drv] = {
            'lap_numbers': np.array(lap_numbers),
            'top_speeds': np.array(top_speeds),
            'laps_obj': laps
        }
        print(f"  ✓ {drv}: {len(lap_numbers)} laps with speed data")
    else:
        print(f"  Warning: no speed telemetry for {drv}")

if not driver_speed_data:
    raise SystemExit("No speed data found. Check telemetry availability.")

# ---- PLOT ----
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(16, 8))
ax.set_facecolor("#0a0a0a")
fig.patch.set_facecolor("#0a0a0a")

for drv, data in driver_speed_data.items():
    color = safe_team_color(data['laps_obj'], drv)
    
    # Russell dotted line for consistency
    if drv == "RUS":
        linestyle = ":"
        linewidth = 3.0
    else:
        linestyle = "-"
        linewidth = 2.5
    
    x = data['lap_numbers']
    y = data['top_speeds']
    
    # Apply smoothing
    y_smooth = (
        pd.Series(y)
        .rolling(window=SMOOTH_WINDOW, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )
    
    ax.plot(
        x,
        y_smooth,
        color=color,
        linewidth=linewidth,
        alpha=0.95,
        linestyle=linestyle,
        label=drv,
        zorder=3
    )

ax.legend(
    title="Driver",
    loc="upper right",
    frameon=True,
    facecolor="#1a1a1a",
    edgecolor="#333",
    fontsize=11,
    title_fontsize=12,
)

ax.set_title(
    f"{EVENT} {YEAR} - Top Speed per Lap",
    fontsize=16,
    weight="bold",
    pad=15,
    color="#ffffff"
)
ax.set_xlabel("Lap Number", fontsize=13)
ax.set_ylabel("Top Speed (km/h)", fontsize=13)
ax.grid(alpha=0.15, color="#333", linestyle="-", linewidth=0.5, zorder=0)

# Set reasonable y-axis limits
all_speeds = np.concatenate([data['top_speeds'] for data in driver_speed_data.values()])
ymin = all_speeds.min() - 5
ymax = all_speeds.max() + 5
ax.set_ylim(ymin, ymax)
ax.set_xlim(left=0)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())

print(f"\n✓ Saved top speed analysis to: {os.path.abspath(OUTPUT_PNG)}")

# Print average speed comparison
print("\n=== AVERAGE TOP SPEED COMPARISON ===")
for drv, data in driver_speed_data.items():
    avg_speed = data['top_speeds'].mean()
    print(f"{drv}: {avg_speed:.1f} km/h")

plt.show()