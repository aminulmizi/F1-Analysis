"""
Singapore 2025 Qualifying - Sector Analysis by Engine Supplier
Shows absolute sector times for all drivers, grouped by sector
Similar to Formula Data Analysis style
Compatible with FastF1 v3.6.x
"""

import os
import fastf1
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ---- CONFIG ----
YEAR = 2025
EVENT = "Singapore"
SESSION = "Q"
OUTPUT_PNG = "singapore_2025_sector_analysis_grouped.png"
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "fastf1")

# Engine supplier mapping (2025 grid) - COMPLETE LIST
ENGINE_SUPPLIERS = {
    # Ferrari engines
    "Ferrari": "Ferrari",
    "Haas F1 Team": "Ferrari",
    "Kick Sauber": "Ferrari",
    
    # Mercedes engines
    "Mercedes": "Mercedes",
    "Williams": "Mercedes",
    "McLaren": "Mercedes",
    "Aston Martin": "Mercedes",
    
    # Honda RBPT engines
    "Red Bull Racing": "Honda RBPT",
    "RB": "Honda RBPT",
    "Racing Bulls": "Honda RBPT",
    "Visa Cash App RB": "Honda RBPT",
    "AlphaTauri": "Honda RBPT",
    
    # Renault engines
    "Alpine F1 Team": "Renault",
    "Alpine": "Renault",
}

# Engine colors
ENGINE_COLORS = {
    "Ferrari": "#DC0000",       # Red
    "Mercedes": "#00D2BE",      # Cyan/Teal
    "Honda RBPT": "#3671C6",    # Blue
    "Renault": "#FF87BC",       # Pink
}

# ----------------

os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

print(f"Loading {EVENT} {YEAR} {SESSION}...")
session = fastf1.get_session(YEAR, EVENT, SESSION)
session.load(laps=True, telemetry=False)
print("Session loaded.\n")

# Get fastest lap per driver (one lap per driver only)
all_laps = session.laps[
    session.laps["Sector1Time"].notna() & 
    session.laps["Sector2Time"].notna() & 
    session.laps["Sector3Time"].notna()
]

# Group by driver and get fastest lap for each
fastest_laps = all_laps.loc[all_laps.groupby("Driver")["LapTime"].idxmin()]

if fastest_laps.empty:
    raise SystemExit("No valid laps with all sector times.")

# Collect data
sector_data = []

for idx, lap in fastest_laps.iterrows():
    driver = lap["Driver"]
    team = lap["Team"]
    
    engine = ENGINE_SUPPLIERS.get(team, "Unknown")
    color = ENGINE_COLORS.get(engine, "#808080")
    
    s1 = lap["Sector1Time"].total_seconds()
    s2 = lap["Sector2Time"].total_seconds()
    s3 = lap["Sector3Time"].total_seconds()
    
    sector_data.append({
        "Driver": driver,
        "Team": team,
        "Engine": engine,
        "Color": color,
        "Sector1": s1,
        "Sector2": s2,
        "Sector3": s3,
        "LapTime": lap["LapTime"].total_seconds()
    })

df = pd.DataFrame(sector_data)

# Sort by lap time (fastest first)
df = df.sort_values("LapTime")

print(f"{'='*70}")
print("SECTOR TIMES")
print(f"{'='*70}")
for idx, row in df.iterrows():
    print(f"{row['Driver']:3s} ({row['Engine']:8s}): "
          f"S1={row['Sector1']:.3f}s  S2={row['Sector2']:.3f}s  S3={row['Sector3']:.3f}s")

# ---- VISUALIZATION ----
plt.style.use("dark_background")
fig, axes = plt.subplots(3, 1, figsize=(16, 12))
fig.patch.set_facecolor("#0a0a0a")

sectors = ["Sector1", "Sector2", "Sector3"]
sector_labels = ["Sector 1 (s)", "Sector 2 (s)", "Sector 3 (s)"]

for ax_idx, (ax, sector, sector_label) in enumerate(zip(axes, sectors, sector_labels)):
    ax.set_facecolor("#0a0a0a")
    
    # Get data for this sector
    drivers = df["Driver"].values
    times = df[sector].values
    colors = df["Color"].values
    
    # Debug: Print colors to verify
    if ax_idx == 0:
        for driver, color, engine in zip(df["Driver"], df["Color"], df["Engine"]):
            print(f"{driver}: {engine} -> {color}")
    
    # Create bar chart
    x_pos = np.arange(len(drivers))
    bars = ax.bar(x_pos, times, color=colors, alpha=0.9,
                  edgecolor="white", linewidth=0.8, width=0.7)
    
    # Highlight fastest
    fastest_idx = times.argmin()
    bars[fastest_idx].set_edgecolor("gold")
    bars[fastest_idx].set_linewidth(2.5)
    
    # Add time labels on top of bars
    for i, (bar, time) in enumerate(zip(bars, times)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.2f}',
                ha='center', va='bottom', color='white',
                fontsize=9, weight='normal')
    
    # Styling
    ax.set_xticks(x_pos)
    ax.set_xticklabels(drivers, fontsize=11, rotation=0)
    ax.set_ylabel(sector_label, fontsize=13, color="white", weight="bold")
    ax.grid(alpha=0.2, color="#333", axis='y', linewidth=0.5)
    ax.tick_params(colors="white", labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set y-axis to start near minimum for better visualization
    ymin = times.min() * 0.98
    ymax = times.max() * 1.02
    ax.set_ylim(ymin, ymax)

# Create legend OUTSIDE the plot area - top right corner
legend_elements = [
    plt.Rectangle((0,0),1,1, facecolor=ENGINE_COLORS[engine], 
                  edgecolor='white', linewidth=1, label=engine)
    for engine in ["Ferrari", "Mercedes", "Honda RBPT", "Renault"]
]

fig.legend(handles=legend_elements, 
           title="Engine Supplier",
           loc='upper right',
           bbox_to_anchor=(0.99, 0.96),
           ncol=1,
           frameon=True,
           facecolor="#1a1a1a",
           edgecolor="#555",
           fontsize=10,
           title_fontsize=11)

fig.suptitle(f"Best Sector Times ({YEAR} {EVENT} Grand Prix - Qualifying)",
             fontsize=16, weight="bold", color="white", y=0.99)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())

print(f"\n✓ Saved to: {os.path.abspath(OUTPUT_PNG)}")

plt.show()