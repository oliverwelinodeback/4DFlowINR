# --- Stenosis region visualisation — Vincent cases (ICAD48, ICAD21) -----------
#
# 3D scatter of the HR reference fluid domain coloured by speed at peak flow.
# A red wireframe bounding box in normalised coords is overlaid.
#
# Coordinates are standardized using a FIXED TEMPLATE (use_baseline_normalization=True),
# so the same box in normalised coords covers the same physical region for both LR and HRLR.
# Find the box once (with either LR or HRLR) and reuse it for all data types of that case.
#
# The key difference between LR and HRLR is the ref_spatial_factor and dx — if these are
# set wrong, prepare_ref_data produces wrong-shaped arrays (broadcast error). Use DTYPE
# to load with the correct resolution settings.
#
# Workflow:
#   1. Set CASE = "ICAD48" or "ICAD21", DTYPE = "LR" or "HRLR"
#   2. Run: python tools/plot_stenosis_mask_check_vincent.py
#   3. Inspect plots/stenosis_mask_{CASE}_{DTYPE}.png — adjust box coords, re-run
#   4. Copy the final (single) box coords into evaluate_vincent_metrics.py for that case
#
# Run from src/:  python tools/plot_stenosis_mask_check_vincent.py
# ---------------------------------------------------------------------------
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from configs.tunings_260505_Vincent.Config_260505_Vincent_factorial import get_config
from utils.prepare_data import prepare_data, load_data, load_ref_data, prepare_ref_data


# ---------------------------------------------------------------------------
# Settings — change CASE, DTYPE, and adjust the bounding box
# ---------------------------------------------------------------------------
CASE  = "ICAD21"   # "ICAD48" or "ICAD21"
DTYPE = "HRLR"     # "LR" or "HRLR" — only affects which input file is loaded,
                   # NOT the normalised coords (those use the fixed template)

CASE_META = {
    "ICAD48": dict(
        data_file_ref = "../data/Vincent/stenosis_50/ICAD48_05mm3_20ms.h5",
        venc=1.3, peak_flow_idx=14,
        LR   = dict(
            data_file          = "../data/Vincent/stenosis_50/ICAD48_05mm3_20ms_LR_sv13_tSNR10_newMask.h5",
            ref_spatial_factor = 2, ref_temporal_factor = 2,
            dx=0.001, dt=0.04,
        ),
        HRLR = dict(
            data_file          = "../data/Vincent/stenosis_50/ICAD48_05mm3_20ms_HRLR_sv13_tSNR10.h5",
            ref_spatial_factor = 1, ref_temporal_factor = 1,
            dx=0.0005, dt=0.02,
        ),
    ),
    "ICAD21": dict(
        data_file_ref = "../data/Vincent/stenosis_70/ICAD21_05mm3_20ms.h5",
        venc=2.6, peak_flow_idx=12,
        LR   = dict(
            data_file          = "../data/Vincent/stenosis_70/ICAD21_05mm3_20ms_LR_sv26_tSNR10_newMask.h5",
            ref_spatial_factor = 2, ref_temporal_factor = 2,
            dx=0.001, dt=0.04,
        ),
        HRLR = dict(
            data_file          = "../data/Vincent/stenosis_70/ICAD21_05mm3_20ms_HRLR_sv26_tSNR10.h5",
            ref_spatial_factor = 1, ref_temporal_factor = 1,
            dx=0.0005, dt=0.02,
        ),
    ),
}

# Bounding box in normalised coordinates.
# Because use_baseline_normalization=True, these coords are the same for LR and HRLR.
# Find the box with either DTYPE and copy the same values for both data types in evaluate_vincent_metrics.py.
X_LO, X_HI = -1.7, -1.4
Y_LO, Y_HI = -1.55, -1.35
Z_LO, Z_HI = -1.4, -1.3

N_SCATTER = 300_000
OUT_PATH  = f"plots/stenosis_mask_{CASE}_{DTYPE}.png"


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
cm       = CASE_META[CASE]
dtype_cm = cm[DTYPE]

config = get_config()
config.data_file              = dtype_cm["data_file"]
config.data_file_ref          = cm["data_file_ref"]
config.constants.venc         = cm["venc"]
config.predictions.peak_flow_idx = cm["peak_flow_idx"]
config.ref_spatial_factor     = dtype_cm["ref_spatial_factor"]
config.ref_temporal_factor    = dtype_cm["ref_temporal_factor"]
config.resolution.from_file   = False
config.resolution.dx = config.resolution.dy = config.resolution.dz = dtype_cm["dx"]
config.resolution.dt          = dtype_cm["dt"]

print(f"Loading {DTYPE} data for {CASE}...")
u_lr, v_lr, w_lr, p_lr, px_lr, py_lr, pz_lr, mask, config = load_data(config)
_, xyz_data, mask_flat, _, standardization_factors, U_max = \
    prepare_data(config, u_lr, v_lr, w_lr, p_lr, px_lr, py_lr, pz_lr, mask)

print("Loading HR reference data...")
u_hr, v_hr, w_hr, p_hr, px_hr, py_hr, pz_hr, mask_hr = load_ref_data(config)
uvw_ref, xyz_ref, mask_flat_ref, _ = prepare_ref_data(
    config, u_lr, u_hr, v_hr, w_hr, p_hr, px_hr, py_hr, pz_hr, mask_hr, U_max
)

T_ref, X_ref, Y_ref, Z_ref = u_hr.shape
t = cm["peak_flow_idx"]

# ---------------------------------------------------------------------------
# Build fluid point cloud at peak timestep in normalised coords
# ---------------------------------------------------------------------------
t_vals   = np.unique(xyz_ref[:, 0])
t_target = t_vals[t] if t < len(t_vals) else t_vals[-1]
t_mask   = xyz_ref[:, 0] == t_target
fluid_mask = mask_flat_ref == 1
peak_fluid = t_mask & fluid_mask

xyz_peak = xyz_ref[peak_fluid]
x_pts = xyz_peak[:, 1]
y_pts = xyz_peak[:, 2]
z_pts = xyz_peak[:, 3]

mfr_4d        = mask_flat_ref.reshape(T_ref, X_ref, Y_ref, Z_ref)
fluid_peak_3d = mfr_4d[t].ravel().astype(bool)
speed_3d      = np.sqrt(u_hr[t]**2 + v_hr[t]**2 + w_hr[t]**2)
speed_pts     = speed_3d.ravel()[fluid_peak_3d]

print(f"\nNormalised coord ranges at peak timestep (fluid only) [{DTYPE}]:")
print(f"  x: [{x_pts.min():.3f}, {x_pts.max():.3f}]")
print(f"  y: [{y_pts.min():.3f}, {y_pts.max():.3f}]")
print(f"  z: [{z_pts.min():.3f}, {z_pts.max():.3f}]")
print(f"  speed: mean={speed_pts.mean():.3f}  max={speed_pts.max():.3f} m/s")
print(f"  total fluid points: {len(x_pts):,}")

# ---------------------------------------------------------------------------
# Stenosis bounding box
# ---------------------------------------------------------------------------
in_box = (
    (x_pts >= X_LO) & (x_pts <= X_HI) &
    (y_pts >= Y_LO) & (y_pts <= Y_HI) &
    (z_pts >= Z_LO) & (z_pts <= Z_HI)
)
n_box = in_box.sum()
print(f"\nStenosis box [{X_LO},{X_HI}] x [{Y_LO},{Y_HI}] x [{Z_LO},{Z_HI}]:")
print(f"  fluid points in box: {n_box:,}  ({100*n_box/len(x_pts):.1f}% of fluid)")
if n_box > 0:
    print(f"  speed in box: mean={speed_pts[in_box].mean():.3f}  "
          f"max={speed_pts[in_box].max():.3f} m/s")

# ---------------------------------------------------------------------------
# 3D scatter plot
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)

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
rgba[:, 3] = 0.05 + 0.95 * norm(s_pl)

fig = plt.figure(figsize=(11, 8))
ax  = fig.add_subplot(111, projection='3d')
ax.scatter(x_pl, y_pl, z_pl, c=rgba, s=2, edgecolors='none')

if in_box_pl.sum() > 0:
    ax.scatter(x_pl[in_box_pl], y_pl[in_box_pl], z_pl[in_box_pl],
               c='cyan', s=6, edgecolors='none', alpha=0.8, label='stenosis box')

# Red wireframe bounding box
corners = np.array([
    [X_LO, Y_LO, Z_LO], [X_HI, Y_LO, Z_LO],
    [X_HI, Y_HI, Z_LO], [X_LO, Y_HI, Z_LO],
    [X_LO, Y_LO, Z_HI], [X_HI, Y_LO, Z_HI],
    [X_HI, Y_HI, Z_HI], [X_LO, Y_HI, Z_HI],
])
edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
for a, b in edges:
    ax.plot(*zip(corners[a], corners[b]), color='red', lw=1.5)

ax.set_xlabel('X (norm)'); ax.set_ylabel('Y (norm)'); ax.set_zlabel('Z (norm)')
ax.set_title(
    f'{CASE} [{DTYPE}] HR reference — speed at t={t}\n'
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
print("Box coords are the same for LR and HRLR (fixed template normalization).")
print("Copy the confirmed box into evaluate_vincent_metrics.py as a single stenosis_box per case.")
