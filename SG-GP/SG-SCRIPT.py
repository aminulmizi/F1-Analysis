"""
Singapore 2025 Qualifying Telemetry + Delta Plot
Russell vs Verstappen - FastF1 v3.6.1 compatible
DARK BACKGROUND VERSION
Produces:
 - CSV with sector times
 - telemetry overlay PNG
 - delta-only PNG
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fastf1
from fastf1.plotting import get_team_color

# --- CONFIG ---
YEAR = 2025
EVENT = "Singapore"
SESSION = "Q"
DRIVER_REF = "RUS"  # reference (Russell)
DRIVER_CMP = "VER"  # comparison (Verstappen)
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "fastf1")
OUTPUT_DIR = "output"
# --------------

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

def ms(td):
    return td.total_seconds() * 1000.0 if pd.notna(td) else np.nan

def compute_delta(ref_tel, cmp_tel):
    """Compute delta time curve by distance alignment."""
    ref_dist = ref_tel['Distance']
    cmp_dist = cmp_tel['Distance']
    cmp_interp = np.interp(ref_dist, cmp_dist, cmp_tel['Time'])
    delta = cmp_interp - ref_tel['Time']
    delta -= delta[0]
    return delta


def main():
    print(f"Start: {datetime.utcnow().isoformat()} UTC")

    session = fastf1.get_session(YEAR, EVENT, SESSION)
    session.load(laps=True, telemetry=True)

    laps_ref = session.laps.pick_drivers([DRIVER_REF])
    laps_cmp = session.laps.pick_drivers([DRIVER_CMP])

    if laps_ref.empty or laps_cmp.empty:
        raise SystemExit("No laps found for one or both drivers.")

    fastest_ref = laps_ref.pick_fastest()
    fastest_cmp = laps_cmp.pick_fastest()

    # Retrieve telemetry with distance
    tel_ref = fastest_ref.get_car_data().add_distance()
    tel_cmp = fastest_cmp.get_car_data().add_distance()

    # Add lap-relative time in seconds
    tel_ref['Time'] = (tel_ref['SessionTime'] - tel_ref['SessionTime'].iloc[0]).dt.total_seconds()
    tel_cmp['Time'] = (tel_cmp['SessionTime'] - tel_cmp['SessionTime'].iloc[0]).dt.total_seconds()

    # Compute delta
    delta_time = compute_delta(tel_ref, tel_cmp)

    # --- Plot telemetry overlay + delta with DARK BACKGROUND ---
    plt.style.use("dark_background")
    
    col_ref = get_team_color(fastest_ref['Team'], session)
    col_cmp = get_team_color(fastest_cmp['Team'], session)

    fig, axes = plt.subplots(6, 1, sharex=True, figsize=(14, 14))
    fig.patch.set_facecolor("#0a0a0a")

    # Apply dark background to all axes
    for ax in axes:
        ax.set_facecolor("#0c0c0c")
        ax.grid(alpha=0.15, color="#333", linestyle="-", linewidth=0.5)
        ax.tick_params(colors="white", labelsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Speed
    axes[0].plot(tel_ref['Distance'], tel_ref['Speed'], color=col_ref, 
                 label=DRIVER_REF, linewidth=2, alpha=0.9)
    axes[0].plot(tel_cmp['Distance'], tel_cmp['Speed'], color=col_cmp, 
                 label=DRIVER_CMP, linewidth=2, alpha=0.9)
    axes[0].legend(frameon=True, facecolor="#1a1a1a", edgecolor="#333", fontsize=10)
    axes[0].set_ylabel('Speed (km/h)', fontsize=11, color="white", weight="bold")

    # Throttle
    axes[1].plot(tel_ref['Distance'], tel_ref['Throttle'], color=col_ref, 
                 linewidth=2, alpha=0.9)
    axes[1].plot(tel_cmp['Distance'], tel_cmp['Throttle'], color=col_cmp, 
                 linewidth=2, alpha=0.9)
    axes[1].set_ylabel('Throttle (%)', fontsize=11, color="white", weight="bold")

    # Brake
    axes[2].plot(tel_ref['Distance'], tel_ref['Brake'], color=col_ref, 
                 linewidth=2, alpha=0.9)
    axes[2].plot(tel_cmp['Distance'], tel_cmp['Brake'], color=col_cmp, 
                 linewidth=2, alpha=0.9)
    axes[2].set_ylabel('Brake', fontsize=11, color="white", weight="bold")

    # Gear
    axes[3].plot(tel_ref['Distance'], tel_ref['nGear'], color=col_ref, 
                 linewidth=2, alpha=0.9)
    axes[3].plot(tel_cmp['Distance'], tel_cmp['nGear'], color=col_cmp, 
                 linewidth=2, alpha=0.9)
    axes[3].set_ylabel('Gear', fontsize=11, color="white", weight="bold")

    # RPM
    axes[4].plot(tel_ref['Distance'], tel_ref['RPM'], color=col_ref, 
                 linewidth=2, alpha=0.9)
    axes[4].plot(tel_cmp['Distance'], tel_cmp['RPM'], color=col_cmp, 
                 linewidth=2, alpha=0.9)
    axes[4].set_ylabel('RPM', fontsize=11, color="white", weight="bold")

    # --- Delta plot (bottom) ---
    axes[5].plot(tel_ref['Distance'], delta_time, color="#FFFFFF", 
                 linewidth=2.5, alpha=0.9)
    axes[5].axhline(0, color='white', linestyle='--', alpha=0.4, linewidth=1)
    axes[5].fill_between(tel_ref['Distance'], delta_time, 0, 
                         where=(delta_time < 0), color=col_ref, alpha=0.2,
                         label=f'{DRIVER_REF} faster')
    axes[5].fill_between(tel_ref['Distance'], delta_time, 0, 
                         where=(delta_time >= 0), color=col_cmp, alpha=0.2,
                         label=f'{DRIVER_CMP} faster')
    axes[5].legend(frameon=True, facecolor="#1a1a1a", edgecolor="#333", 
                   fontsize=10, loc='upper right')
    axes[5].set_xlabel('Distance (m)', fontsize=12, color="white", weight="bold")
    axes[5].set_ylabel(f'Time Delta (s)', fontsize=11, color="white", weight="bold")

    fig.suptitle(f"{EVENT} {YEAR} Qualifying - {DRIVER_REF} vs {DRIVER_CMP}", 
                 fontsize=16, weight="bold", color="white", y=0.995)
    
    plt.tight_layout()

    out_tel = os.path.join(
        OUTPUT_DIR,
        f"{EVENT.lower()}_{YEAR}_qual_{DRIVER_REF}_{DRIVER_CMP}_telemetry_dark.png"
    )
    plt.savefig(out_tel, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote combined telemetry + delta plot: {out_tel}")

if __name__ == "__main__":
    main()