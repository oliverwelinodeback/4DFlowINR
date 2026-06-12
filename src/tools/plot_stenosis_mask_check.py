# --- Stenosis region visualisation tool (normalized coords) -------------------
#
# Plots a 3D scatter of the HR reference fluid domain coloured by speed at
# peak flow. A bounding box defined in normalized coordinates is overlaid as a
# red wireframe. Adjust X_LO/X_HI/Y_LO/Y_HI/Z_LO/Z_HI and re-run until the
# box covers the stenotic jet.
#
# Coordinates are in the same normalized space used by xyz_ref / the PINN, so
# any crop defined here can be applied directly to metric evaluation.
#
# Run from src/:  python tools/plot_stenosis_mask_check.py
# ---------------------------------------------------------------------------
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from configs.tunings_260409_SAPINN_LBFGS.Config_260409_SAPINN_sweepN_factorial import get_config
from utils.prepare_data import prepare_data, load_data, load_ref_data, prepare_ref_data


# ---------------------------------------------------------------------------
# Settings — edit the bounding box in NORMALIZED coordinates
# Run once with defaults to see the printed coord ranges, then refine.
# ---------------------------------------------------------------------------
X_LO, X_HI = -1.65, -1.3
Y_LO, Y_HI = -1.55, -1.4
Z_LO, Z_HI = -1.4, -1.3

PEAK_T_IDX = 14
OUT_PATH   = "plots/stenosis_mask_check.png"
N_SCATTER  = 300_000        # max points in scatter (subsampled if more)


# ---------------------------------------------------------------------------
# Load data  (same pattern as export_R1_R2_vtk.py)
# ---------------------------------------------------------------------------
config = get_config()
config.data_file     = "../data/stenosis_50/ICAD48_05mm3_20ms_LR_sv13_tSNR10_newMask.h5"
config.data_file_ref = "../data/stenosis_50/ICAD48_05mm3_20ms.h5"
config.constants.venc = 1.3
config.predictions.peak_flow_idx = PEAK_T_IDX

print("Loading LR data...")
u_lr, v_lr, w_lr, p_lr, px_lr, py_lr, pz_lr, mask, config = load_data(config)
_, xyz_data, mask_flat, _, standardization_factors, U_max = \
    prepare_data(config, u_lr, v_lr, w_lr, p_lr, px_lr, py_lr, pz_lr, mask)

print("Loading HR reference data...")
u_hr, v_hr, w_hr, p_hr, px_hr, py_hr, pz_hr, mask_hr = load_ref_data(config)
uvw_ref, xyz_ref, mask_flat_ref, _ = prepare_ref_data(
    config, u_lr, u_hr, v_hr, w_hr, p_hr, px_hr, py_hr, pz_hr, mask_hr, U_max
)

T_ref, X_ref, Y_ref, Z_ref = u_hr.shape

# Raw speed at peak timestep, denormalized to m/s
U = config.constants.U
t = PEAK_T_IDX
speed_3d = np.sqrt(u_hr[t]**2 + v_hr[t]**2 + w_hr[t]**2) * mask_hr  # m/s

# ---------------------------------------------------------------------------
# Build point cloud in normalized coords at peak timestep, fluid only
# xyz_ref: (T*X*Y*Z, 4) columns [t_norm, x_norm, y_norm, z_norm]
# ---------------------------------------------------------------------------
t_vals = np.unique(xyz_ref[:, 0])
t_target = t_vals[t] if t < len(t_vals) else t_vals[-1]
t_mask   = xyz_ref[:, 0] == t_target
fluid_mask = mask_flat_ref == 1
peak_fluid = t_mask & fluid_mask          # fluid voxels at peak timestep

xyz_peak   = xyz_ref[peak_fluid]          # (N_fluid, 4)
x_pts = xyz_peak[:, 1]
y_pts = xyz_peak[:, 2]
z_pts = xyz_peak[:, 3]

# Speed values for the same voxels
mfr_4d = mask_flat_ref.reshape(T_ref, X_ref, Y_ref, Z_ref)
fluid_peak_3d = mfr_4d[t].ravel().astype(bool)
speed_pts = speed_3d.ravel()[fluid_peak_3d]   # (N_fluid,)

# Print coordinate ranges so you know where to set the box
print(f"\nNormalized coord ranges at peak timestep (fluid only):")
print(f"  x: [{x_pts.min():.3f}, {x_pts.max():.3f}]")
print(f"  y: [{y_pts.min():.3f}, {y_pts.max():.3f}]")
print(f"  z: [{z_pts.min():.3f}, {z_pts.max():.3f}]")
print(f"  speed: mean={speed_pts.mean():.3f}  max={speed_pts.max():.3f} m/s")
print(f"  total fluid points: {len(x_pts):,}")

# ---------------------------------------------------------------------------
# Stenosis crop in normalized coords
# ---------------------------------------------------------------------------
in_box = (
    (x_pts >= X_LO) & (x_pts <= X_HI) &
    (y_pts >= Y_LO) & (y_pts <= Y_HI) &
    (z_pts >= Z_LO) & (z_pts <= Z_HI)
)
n_box = in_box.sum()
print(f"\nStenosis box [{X_LO},{X_HI}] x [{Y_LO},{Y_HI}] x [{Z_LO},{Z_HI}]:")
print(f"  fluid points in box : {n_box:,}  ({100*n_box/len(x_pts):.1f}% of fluid)")
if n_box > 0:
    print(f"  speed in box: mean={speed_pts[in_box].mean():.3f}  "
          f"max={speed_pts[in_box].max():.3f} m/s")

# ---------------------------------------------------------------------------
# 3D scatter plot
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)

# Subsample if needed
N = len(x_pts)
if N > N_SCATTER:
    sel = np.sort(np.random.choice(N, size=N_SCATTER, replace=False))
    x_pl, y_pl, z_pl, s_pl = x_pts[sel], y_pts[sel], z_pts[sel], speed_pts[sel]
    in_box_pl = in_box[sel]
else:
    x_pl, y_pl, z_pl, s_pl, in_box_pl = x_pts, y_pts, z_pts, speed_pts, in_box

vmax = speed_pts.max()
norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
cmap_obj = plt.get_cmap('hot_r')

rgba = cmap_obj(norm(s_pl)).astype(np.float32)
rgba[:, 3] = 0.05 + 0.95 * norm(s_pl)   # high speed = opaque

fig = plt.figure(figsize=(11, 8))
ax  = fig.add_subplot(111, projection='3d')
ax.scatter(x_pl, y_pl, z_pl, c=rgba, s=2, edgecolors='none')

# Highlight the box points in a different color so they pop
if in_box_pl.sum() > 0:
    ax.scatter(x_pl[in_box_pl], y_pl[in_box_pl], z_pl[in_box_pl],
               c='cyan', s=6, edgecolors='none', alpha=0.8, label='stenosis box')

# Draw bounding box wireframe in normalized coords
corners = np.array([
    [X_LO, Y_LO, Z_LO], [X_HI, Y_LO, Z_LO],
    [X_HI, Y_HI, Z_LO], [X_LO, Y_HI, Z_LO],
    [X_LO, Y_LO, Z_HI], [X_HI, Y_LO, Z_HI],
    [X_HI, Y_HI, Z_HI], [X_LO, Y_HI, Z_HI],
])
edges = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7),
]
for a, b in edges:
    ax.plot(*zip(corners[a], corners[b]), color='red', lw=1.5)

ax.set_xlabel('X (norm)'); ax.set_ylabel('Y (norm)'); ax.set_zlabel('Z (norm)')
ax.set_title(
    f'ICAD48 HR reference — speed at t={PEAK_T_IDX}\n'
    f'Box: x[{X_LO},{X_HI}] y[{Y_LO},{Y_HI}] z[{Z_LO},{Z_HI}]  '
    f'({n_box:,} pts, {100*n_box/len(x_pts):.1f}% of fluid)'
)
ax.legend(fontsize=9)

sm = mpl.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.05).set_label('Speed [m/s]')

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=200)
plt.close(fig)
print(f"\nSaved: {OUT_PATH}")
print("Adjust X_LO/X_HI/Y_LO/Y_HI/Z_LO/Z_HI at the top of the script and re-run.")
