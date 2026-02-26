import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# === File paths ===
HR_FILE = '../data/healthy/HV06_05mm3_20ms.h5'
META_FILE = '../models/MetaLearning_MAML_DataDriven_h5/MetaMAML_1000it_h5_../data/healthy/HV06_05mm3_20ms_LR_sv12_tSNR10_newMask.h5_loadMetaInitTrue_20260212-1600/SR_it000050.h5'
NOMETA_FILE = '../models/MetaLearning_MAML_DataDriven_h5/MetaMAML_1000it_h5_../data/healthy/HV06_05mm3_20ms_LR_sv12_tSNR10_newMask.h5_loadMetaInitFalse_20260212-1602/SR_it000050.h5'
OUTPUT_PNG = '2D_plots/roi_exploration.png'

t_idx = 6
meas = 'w'
z_slices = [41, 42, 48, 89, 90]
threshold = 0.05

# === Load HR data ===
print("=" * 70)
print("LOADING HR DATA")
print("=" * 70)
with h5py.File(HR_FILE, 'r') as f:
    hr_w = f[meas][:]        # (25, 128, 94, 70)
    hr_mask = f['mask'][:]    # (1, 128, 94, 70)

hr_mask = hr_mask[0]  # (128, 94, 70)

print(f"HR w shape: {hr_w.shape} (t, z, y, x)")
print(f"HR mask shape: {hr_mask.shape} (z, y, x)")
print()

# === Step 1-3: Analyze each z-slice ===
print("=" * 70)
print(f"ANALYSIS OF Z-SLICES at t={t_idx}, meas='{meas}', threshold={threshold}")
print("=" * 70)

for z in z_slices:
    sl = hr_w[t_idx, z, :, :]  # (94, 70) = (y, x)
    mask_sl = hr_mask[z, :, :]  # (94, 70)
    
    significant = np.abs(sl) > threshold
    ys, xs = np.where(significant)
    
    print(f"\n--- z={z} ---")
    print(f"  Slice shape: {sl.shape} (y, x)")
    print(f"  Max |w|: {np.abs(sl).max():.6f}")
    print(f"  Non-zero voxels (|w|>{threshold}): {significant.sum()}")
    
    if len(ys) > 0:
        print(f"  Bounding box: x_min={xs.min()}, x_max={xs.max()}, y_min={ys.min()}, y_max={ys.max()}")
        print(f"  BB size: {xs.max()-xs.min()+1} x {ys.max()-ys.min()+1} (x * y)")
    else:
        print(f"  No voxels above threshold!")
    
    # Step 4: mask info
    mask_nz = np.count_nonzero(mask_sl)
    if mask_nz > 0:
        mys, mxs = np.where(mask_sl > 0)
        print(f"  Mask nonzero voxels: {mask_nz}")
        print(f"  Mask bounding box: x_min={mxs.min()}, x_max={mxs.max()}, y_min={mys.min()}, y_max={mys.max()}")
    else:
        print(f"  Mask: all zero at this z-slice")

# === Step 5: Meta vs NoMeta comparison at z=48 and z=41 ===
print("\n" + "=" * 70)
print("META vs NOMETA COMPARISON (iteration 50)")
print("=" * 70)

with h5py.File(META_FILE, 'r') as f:
    meta_w = f[meas][:]
with h5py.File(NOMETA_FILE, 'r') as f:
    nometa_w = f[meas][:]

print(f"Meta shape: {meta_w.shape}, NoMeta shape: {nometa_w.shape}")

roi_suggestions = {}

for z in [48, 41]:
    hr_sl = hr_w[t_idx, z, :, :]
    meta_sl = meta_w[t_idx, z, :, :]
    nometa_sl = nometa_w[t_idx, z, :, :]
    mask_sl = hr_mask[z, :, :]
    
    diff = np.abs(nometa_sl - meta_sl)
    # Mask the difference to only look at vessel regions
    diff_masked = diff * (mask_sl > 0)
    
    print(f"\n--- z={z} ---")
    print(f"  Max |NoMeta - Meta|: {diff_masked.max():.6f}")
    print(f"  Mean |NoMeta - Meta| (in mask): {diff_masked[mask_sl > 0].mean():.6f}")
    print(f"  Meta max |w|: {np.abs(meta_sl).max():.6f}")
    print(f"  NoMeta max |w|: {np.abs(nometa_sl).max():.6f}")
    print(f"  HR max |w|: {np.abs(hr_sl).max():.6f}")
    
    # Find the sub-region with highest difference using a sliding window
    # Try different ROI sizes
    best_score = -1
    best_roi = None
    
    for dim in [16, 20, 24, 28, 32]:
        ny, nx = diff_masked.shape
        for y_start in range(0, ny - dim + 1, 2):
            for x_start in range(0, nx - dim + 1, 2):
                roi_diff = diff_masked[y_start:y_start+dim, x_start:x_start+dim]
                score = roi_diff.sum()
                if score > best_score:
                    best_score = score
                    best_roi = (x_start, y_start, dim, score)
    
    x_s, y_s, d, sc = best_roi
    print(f"  Best ROI: x_start={x_s}, y_start={y_s}, dim={d}, total_diff={sc:.4f}")
    
    # Also check a few fixed sizes centered on max diff location
    max_loc = np.unravel_index(np.argmax(diff_masked), diff_masked.shape)
    print(f"  Max diff location: (y={max_loc[0]}, x={max_loc[1]}), value={diff_masked[max_loc]:.6f}")
    
    # Suggest a nice ROI that captures vessel structure
    # Center on the max-diff region but ensure within bounds
    for dim in [20, 24, 28, 32]:
        cy, cx = max_loc
        y_start = max(0, min(cy - dim//2, 94 - dim))
        x_start = max(0, min(cx - dim//2, 70 - dim))
        roi_hr = np.abs(hr_sl[y_start:y_start+dim, x_start:x_start+dim])
        roi_diff_check = diff_masked[y_start:y_start+dim, x_start:x_start+dim]
        n_vessel = np.count_nonzero(mask_sl[y_start:y_start+dim, x_start:x_start+dim])
        print(f"    Centered ROI (dim={dim}): x_start={x_start}, y_start={y_start}, "
              f"vessel_voxels={n_vessel}, sum_diff={roi_diff_check.sum():.4f}, max_hr={roi_hr.max():.4f}")
    
    roi_suggestions[z] = best_roi

# === Step 6: Diagnostic PNG images ===
print("\n" + "=" * 70)
print("GENERATING DIAGNOSTIC IMAGES")
print("=" * 70)

z_main = 48
hr_sl = hr_w[t_idx, z_main, :, :]
meta_sl = meta_w[t_idx, z_main, :, :]
nometa_sl = nometa_w[t_idx, z_main, :, :]
mask_sl = hr_mask[z_main, :, :]

x_s, y_s, d, _ = roi_suggestions[z_main]

# Also get z=41 data
hr_sl_41 = hr_w[t_idx, 41, :, :]
meta_sl_41 = meta_w[t_idx, 41, :, :]
nometa_sl_41 = nometa_w[t_idx, 41, :, :]

x_s_41, y_s_41, d_41, _ = roi_suggestions[41]

fig, axes = plt.subplots(3, 5, figsize=(22, 13))

# Determine color range from HR
vmin, vmax = -np.abs(hr_sl).max(), np.abs(hr_sl).max()

# Row 0: Full HR slices with ROI boxes
for i, (z, sl, xs, ys, dd) in enumerate([
    (48, hr_sl, x_s, y_s, d),
    (41, hr_sl_41, x_s_41, y_s_41, d_41)
]):
    ax = axes[0, i]
    im = ax.imshow(sl, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower', aspect='equal')
    rect = patches.Rectangle((xs-0.5, ys-0.5), dd, dd, linewidth=2, edgecolor='lime', facecolor='none')
    ax.add_patch(rect)
    ax.set_title(f'HR z={z}, t={t_idx}, w\nROI: x={xs}, y={ys}, dim={dd}', fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046)

# Show mask for z=48
ax = axes[0, 2]
ax.imshow(mask_sl, cmap='gray', origin='lower', aspect='equal')
rect = patches.Rectangle((x_s-0.5, y_s-0.5), d, d, linewidth=2, edgecolor='lime', facecolor='none')
ax.add_patch(rect)
ax.set_title(f'Mask z={z_main}', fontsize=10)

# Show |NoMeta - Meta| diff for z=48
diff_sl = np.abs(nometa_sl - meta_sl) * (mask_sl > 0)
ax = axes[0, 3]
im = ax.imshow(diff_sl, cmap='hot', origin='lower', aspect='equal')
rect = patches.Rectangle((x_s-0.5, y_s-0.5), d, d, linewidth=2, edgecolor='lime', facecolor='none')
ax.add_patch(rect)
ax.set_title(f'|NoMeta - Meta| z={z_main}', fontsize=10)
plt.colorbar(im, ax=ax, fraction=0.046)

# Show |NoMeta - Meta| diff for z=41
diff_sl_41 = np.abs(nometa_sl_41 - meta_sl_41) * (hr_mask[41, :, :] > 0)
ax = axes[0, 4]
im = ax.imshow(diff_sl_41, cmap='hot', origin='lower', aspect='equal')
rect = patches.Rectangle((x_s_41-0.5, y_s_41-0.5), d_41, d_41, linewidth=2, edgecolor='lime', facecolor='none')
ax.add_patch(rect)
ax.set_title(f'|NoMeta - Meta| z=41', fontsize=10)
plt.colorbar(im, ax=ax, fraction=0.046)

# Row 1: Zoomed comparison for z=48
zoom_vmin = min(hr_sl[y_s:y_s+d, x_s:x_s+d].min(), 
                meta_sl[y_s:y_s+d, x_s:x_s+d].min(),
                nometa_sl[y_s:y_s+d, x_s:x_s+d].min())
zoom_vmax = max(hr_sl[y_s:y_s+d, x_s:x_s+d].max(),
                meta_sl[y_s:y_s+d, x_s:x_s+d].max(),
                nometa_sl[y_s:y_s+d, x_s:x_s+d].max())
zoom_vlim = max(abs(zoom_vmin), abs(zoom_vmax))

for j, (data, label) in enumerate([
    (hr_sl[y_s:y_s+d, x_s:x_s+d], 'HR (Ground Truth)'),
    (meta_sl[y_s:y_s+d, x_s:x_s+d], 'Meta (it50)'),
    (nometa_sl[y_s:y_s+d, x_s:x_s+d], 'NoMeta (it50)'),
    (np.abs(meta_sl[y_s:y_s+d, x_s:x_s+d] - hr_sl[y_s:y_s+d, x_s:x_s+d]), '|Meta - HR|'),
    (np.abs(nometa_sl[y_s:y_s+d, x_s:x_s+d] - hr_sl[y_s:y_s+d, x_s:x_s+d]), '|NoMeta - HR|'),
]):
    ax = axes[1, j]
    if j < 3:
        im = ax.imshow(data, cmap='RdBu_r', vmin=-zoom_vlim, vmax=zoom_vlim, origin='lower', aspect='equal')
    else:
        im = ax.imshow(data, cmap='hot', origin='lower', aspect='equal')
    ax.set_title(f'z=48 zoom: {label}', fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046)

# Row 2: Zoomed comparison for z=41
zoom_vmin_41 = min(hr_sl_41[y_s_41:y_s_41+d_41, x_s_41:x_s_41+d_41].min(),
                   meta_sl_41[y_s_41:y_s_41+d_41, x_s_41:x_s_41+d_41].min(),
                   nometa_sl_41[y_s_41:y_s_41+d_41, x_s_41:x_s_41+d_41].min())
zoom_vmax_41 = max(hr_sl_41[y_s_41:y_s_41+d_41, x_s_41:x_s_41+d_41].max(),
                   meta_sl_41[y_s_41:y_s_41+d_41, x_s_41:x_s_41+d_41].max(),
                   nometa_sl_41[y_s_41:y_s_41+d_41, x_s_41:x_s_41+d_41].max())
zoom_vlim_41 = max(abs(zoom_vmin_41), abs(zoom_vmax_41))

for j, (data, label) in enumerate([
    (hr_sl_41[y_s_41:y_s_41+d_41, x_s_41:x_s_41+d_41], 'HR (Ground Truth)'),
    (meta_sl_41[y_s_41:y_s_41+d_41, x_s_41:x_s_41+d_41], 'Meta (it50)'),
    (nometa_sl_41[y_s_41:y_s_41+d_41, x_s_41:x_s_41+d_41], 'NoMeta (it50)'),
    (np.abs(meta_sl_41[y_s_41:y_s_41+d_41, x_s_41:x_s_41+d_41] - hr_sl_41[y_s_41:y_s_41+d_41, x_s_41:x_s_41+d_41]), '|Meta - HR|'),
    (np.abs(nometa_sl_41[y_s_41:y_s_41+d_41, x_s_41:x_s_41+d_41] - hr_sl_41[y_s_41:y_s_41+d_41, x_s_41:x_s_41+d_41]), '|NoMeta - HR|'),
]):
    ax = axes[2, j]
    if j < 3:
        im = ax.imshow(data, cmap='RdBu_r', vmin=-zoom_vlim_41, vmax=zoom_vlim_41, origin='lower', aspect='equal')
    else:
        im = ax.imshow(data, cmap='hot', origin='lower', aspect='equal')
    ax.set_title(f'z=41 zoom: {label}', fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046)

fig.suptitle(f'ROI Exploration: HV06, meas=w, t={t_idx}\nMeta vs NoMeta at iteration 50', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight')
print(f"\nSaved: {OUTPUT_PNG}")
print("Done!")
