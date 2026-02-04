'''
asked chat to move missletracker to 3d
'''

import numpy as np
import matplotlib.pyplot as plt

# --- Constants ---
TIME_STEP = 1.0
EPOCHS = 200
P_MEASURE = 0.55
ALPHA = 0.35
BETA = 0.05

# --- Truth model (constant velocity + random position drift) ---
r_true = np.array([0.0, 0.0, 0.0], dtype=float)
v_true = np.array([2.2, 0.8, 0.35], dtype=float)
sigma_process_pos = 0.8

# --- Sensor noise ---
sigma_measure = 2.0

# --- Initial estimate ---
r_est = np.array([0.0, 0.0, 0.0], dtype=float)
v_est = np.array([0.0, 0.0, 0.0], dtype=float)

# --- Storage ---
r_true_hist = np.zeros((EPOCHS, 3), dtype=float)
r_meas_hist = np.full((EPOCHS, 3), np.nan, dtype=float)
r_est_hist  = np.zeros((EPOCHS, 3), dtype=float)
meas_mask   = np.zeros(EPOCHS, dtype=bool)

rng = np.random.default_rng(7)

for k in range(EPOCHS):
    # True position update (constant velocity + process noise on position)
    r_true = r_true + v_true * TIME_STEP + rng.normal(0.0, sigma_process_pos, size=3)
    r_true_hist[k] = r_true

    # Measurement (with probability)
    r_meas = None
    if rng.random() < P_MEASURE:
        r_meas = r_true + rng.normal(0.0, sigma_measure, size=3)
        r_meas_hist[k] = r_meas
        meas_mask[k] = True

    # --- Predict ---
    r_pred = r_est + v_est * TIME_STEP
    v_pred = v_est

    # --- Update ---
    if r_meas is not None:
        e = r_meas - r_pred
        r_est = r_pred + ALPHA * e
        v_est = v_pred + (BETA / TIME_STEP) * e
    else:
        r_est = r_pred
        v_est = v_pred

    r_est_hist[k] = r_est

# --- Error metrics ---
pos_err = np.linalg.norm(r_est_hist - r_true_hist, axis=1)
rmse = float(np.sqrt(np.mean(pos_err**2)))
print(f"RMSE position error: {rmse:.3f}")
print(f"Measurement availability: {meas_mask.mean()*100:.1f}%")

# Optional: running RMSE (helps see convergence)
running_rmse = np.sqrt(np.cumsum(pos_err**2) / (np.arange(EPOCHS) + 1))

# --- Combined plot: 3D trajectory + error time series ---
fig = plt.figure(figsize=(12, 5))
gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0])

# 3D trajectory subplot
ax3d = fig.add_subplot(gs[0, 0], projection="3d")
ax3d.plot(r_true_hist[:, 0], r_true_hist[:, 1], r_true_hist[:, 2], label="True trajectory")
ax3d.plot(r_est_hist[:, 0],  r_est_hist[:, 1],  r_est_hist[:, 2],  label="Estimated trajectory")
ax3d.scatter(
    r_meas_hist[meas_mask, 0],
    r_meas_hist[meas_mask, 1],
    r_meas_hist[meas_mask, 2],
    s=16,
    label="Measurements",
)

ax3d.set_xlabel("x")
ax3d.set_ylabel("y")
ax3d.set_zlabel("z")
ax3d.set_title("3D Tracking: Truth vs Estimate vs Measurements")
ax3d.legend()

# Error subplot
axerr = fig.add_subplot(gs[0, 1])
axerr.plot(pos_err, label="||position error||")
axerr.plot(running_rmse, label="running RMSE")
axerr.set_xlabel("time step")
axerr.set_ylabel("error magnitude")
axerr.set_title(f"Error over time (RMSE={rmse:.3f})")
axerr.grid(True)
axerr.legend()

fig.tight_layout()
plt.show()
