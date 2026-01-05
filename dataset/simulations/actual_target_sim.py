import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import sys
import os

# --- KONFIGURATION ---
# Pfad zu Ihrem Dataset (ggf. anpassen)
DATA_PATH = '../pick_n_place_procedure_dataset.csv.zip' 
# Wenn Sie das ZIP nutzen: 'dataset/pick_n_place_procedure_dataset.csv.zip'

# Simulations-Geschwindigkeit
SPEED_FACTOR = 5  # 1 = Echtzeit (langweilig), 5 = 5x schneller
WINDOW_SIZE_SEC = 10 # Wie viele Sekunden Historie im Graph angezeigt werden?

# --- SETUP ---
print(f"Lade Daten von {DATA_PATH}...")
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print("Fehler: Datei nicht gefunden. Bitte Pfad in 'DATA_PATH' anpassen!")
    sys.exit(1)

# Sampling Rate aus den Daten (angenommen 100Hz)
dt = df['Time'].iloc[1] - df['Time'].iloc[0]
fs = 1 / dt
window_samples = int(WINDOW_SIZE_SEC * fs)
step_size = int(SPEED_FACTOR) 

# Figure Layout
fig = plt.figure(figsize=(14, 9), facecolor='#f0f0f0')
gs = GridSpec(2, 2, height_ratios=[2, 1], hspace=0.3)

# 1. PLOT: 2D Trajectory (Die Draufsicht)
ax_traj = fig.add_subplot(gs[0, :])
ax_traj.set_title('Roboterarm Simulation: Soll (Rot) vs. Ist (Blau)', fontsize=14, fontweight='bold')
ax_traj.set_xlabel('Yaw (Drehung) [rad]')
ax_traj.set_ylabel('Pitch (Neigung) [rad]')
ax_traj.set_aspect('equal')
ax_traj.grid(True, linestyle='--', alpha=0.5)

# Limits festlegen (etwas Puffer um die Maxima)
margin = 0.5
ax_traj.set_xlim(df['Yaw'].min() - margin, df['Yaw'].max() + margin)
ax_traj.set_ylim(df['Pitch'].min() - margin, df['Pitch'].max() + margin)

# Statische Elemente (Alle Ziele als graue Punkte im Hintergrund)
ax_traj.scatter(df['TargetYaw'], df['TargetPitch'], c='#e0e0e0', s=10, label='Alle Ziele')

# Dynamische Elemente (werden animiert)
# Der "Schweif" (Vergangenheit)
line_path, = ax_traj.plot([], [], 'b-', alpha=0.4, lw=1)
# Der Roboter (Aktuelle Position)
point_robot, = ax_traj.plot([], [], 'bo', ms=12, markeredgecolor='white', label='Roboter (Ist)')
# Das Ziel (Wo er hin soll)
point_target, = ax_traj.plot([], [], 'rx', ms=12, markeredgewidth=3, label='Ziel (Soll)')
# Verbindungsline (Fehler)
line_error, = ax_traj.plot([], [], 'k:', alpha=0.5)

ax_traj.legend(loc='upper right', frameon=True)

# 2. PLOT: Yaw Zeitreihe
ax_yaw = fig.add_subplot(gs[1, 0])
ax_yaw.set_title('Yaw Regelung', fontsize=12)
ax_yaw.set_ylabel('Winkel [rad]')
ax_yaw.set_xlabel('Zeit [s]')
ax_yaw.grid(True, alpha=0.5)
line_yaw_tgt, = ax_yaw.plot([], [], 'r--', lw=1.5, label='Soll')
line_yaw_act, = ax_yaw.plot([], [], 'b-', lw=2, label='Ist')
ax_yaw.legend(loc='upper left')

# 3. PLOT: Pitch Zeitreihe
ax_pitch = fig.add_subplot(gs[1, 1])
ax_pitch.set_title('Pitch Regelung', fontsize=12)
ax_pitch.set_xlabel('Zeit [s]')
ax_pitch.grid(True, alpha=0.5)
line_pitch_tgt, = ax_pitch.plot([], [], 'r--', lw=1.5, label='Soll')
line_pitch_act, = ax_pitch.plot([], [], 'g-', lw=2, label='Ist')
ax_pitch.legend(loc='upper left')

# Text für Zeitstempel
time_text = ax_traj.text(0.02, 0.95, '', transform=ax_traj.transAxes, 
                         fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

def init():
    return line_path, point_robot, point_target, line_error, \
           line_yaw_tgt, line_yaw_act, line_pitch_tgt, line_pitch_act, time_text

def update(frame):
    # Frame ist der aktuelle Index
    current_idx = frame
    
    # Berechne Fenster-Start (für Oszilloskop-Ansicht)
    start_idx = max(0, current_idx - window_samples)
    
    # Daten-Slice holen
    data_slice = df.iloc[start_idx : current_idx+1]
    
    # Aktuelle Werte
    curr_t = df['Time'].iloc[current_idx]
    curr_yaw = df['Yaw'].iloc[current_idx]
    curr_pitch = df['Pitch'].iloc[current_idx]
    tgt_yaw = df['TargetYaw'].iloc[current_idx]
    tgt_pitch = df['TargetPitch'].iloc[current_idx]
    
    # --- UPDATE 2D PLOT ---
    # Schweif (letzte 2 Sekunden reichen für Schweif, sonst wirds unübersichtlich)
    trail_len = min(200, len(data_slice))
    line_path.set_data(data_slice['Yaw'].iloc[-trail_len:], 
                       data_slice['Pitch'].iloc[-trail_len:])
    
    point_robot.set_data([curr_yaw], [curr_pitch])
    point_target.set_data([tgt_yaw], [tgt_pitch])
    
    # Linie zwischen Ziel und Ist (zeigt den Regelfehler visuell)
    line_error.set_data([curr_yaw, tgt_yaw], [curr_pitch, tgt_pitch])
    
    # --- UPDATE ZEITREIHEN ---
    # Yaw
    ax_yaw.set_xlim(data_slice['Time'].min(), data_slice['Time'].max() + 0.1)
    ax_yaw.set_ylim(data_slice['Yaw'].min() - 0.1, data_slice['Yaw'].max() + 0.1)
    line_yaw_tgt.set_data(data_slice['Time'], data_slice['TargetYaw'])
    line_yaw_act.set_data(data_slice['Time'], data_slice['Yaw'])
    
    # Pitch
    ax_pitch.set_xlim(data_slice['Time'].min(), data_slice['Time'].max() + 0.1)
    ax_pitch.set_ylim(data_slice['Pitch'].min() - 0.1, data_slice['Pitch'].max() + 0.1)
    line_pitch_tgt.set_data(data_slice['Time'], data_slice['TargetPitch'])
    line_pitch_act.set_data(data_slice['Time'], data_slice['Pitch'])
    
    time_text.set_text(f'Zeit: {curr_t:.2f} s')
    
    return line_path, point_robot, point_target, line_error, \
           line_yaw_tgt, line_yaw_act, line_pitch_tgt, line_pitch_act, time_text

print("Starte Animation...")
print(f"Sampling: {fs} Hz | Speed-Factor: {SPEED_FACTOR}x")

# Frames berechnen (wir überspringen 'step_size' frames damit es schneller läuft)
frames = range(0, len(df), step_size)

ani = animation.FuncAnimation(fig, update, frames=frames, 
                              init_func=init, blit=False, interval=20)

plt.show()