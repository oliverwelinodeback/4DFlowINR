import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

hr_path = '../data/healthy/HV06_05mm3_20ms.h5'
meta_dir = '../models/MetaLearning_MAML_DataDriven_h5/MetaMAML_1000it_h5_../data/healthy/HV06_05mm3_20ms_LR_sv12_tSNR10_newMask.h5_loadMetaInitTrue_20260212-1600'
nometa_dir = '../models/MetaLearning_MAML_DataDriven_h5/MetaMAML_1000it_h5_../data/healthy/HV06_05mm3_20ms_LR_sv12_tSNR10_newMask.h5_loadMetaInitFalse_20260212-1602'

meta_50 = os.path.join(meta_dir, 'SR_it000050.h5')
nometa_50 = os.path.join(nometa_dir, 'SR_it000050.h5')
meta_100 = os.path.join(meta_dir, 'SR_it000100.h5')
nometa_100 = os.path.join(nometa_dir, 'SR_it000100.h5')

# Data shape: (t=25, z=128, y=94, x=70)
with h5py.File(hr_path, 'r') as f:
    print("Keys:", list(f.keys()))
    print("w shape:", f['w'].shape)
    print("u shape:", f['u'].shape)
    print("v shape:", f['v'].shape)

# ---- PART 1: Find slices with complex flow (high spatial gradient + bidirectional flow) ----
print("\n=== PART 1: Scoring slices by flow complexity ===")

with h5py.File(hr_path, 'r') as fhr, \
     h5py.File(meta_50, 'r') as fmeta, \
     h5py.File(nometa_50, 'r') as fnometa:
    
    results = []
    
    for meas in ['w', 'u', 'v']:
        hr_data = fhr[meas][:]   # (25, 128, 94, 70)
        meta_data = fmeta[meas][:]
        nometa_data = fnometa[meas][:]
        
        for t in range(0, 25, 2):  # scan every other timestep
            for z in range(10, 120):
                hr_sl = hr_data[t, z, :, :]  # (y, x)
                
                # Skip nearly empty slices
                if np.max(np.abs(hr_sl)) < 0.1:
                    continue
                
                # Flow complexity metrics:
                # 1. Spatial gradient magnitude (interesting structures)
                gy, gx = np.gradient(hr_sl)
                grad_mag = np.mean(np.sqrt(gx**2 + gy**2))
                
                # 2. Bidirectional flow (both positive and negative values present)
                pos_frac = np.sum(hr_sl > 0.05) / max(1, np.sum(np.abs(hr_sl) > 0.05))
                bidir_score = 1.0 - abs(pos_frac - 0.5) * 2  # 1.0 = perfectly balanced
                
                # 3. Dynamic range
                dyn_range = np.max(hr_sl) - np.min(hr_sl)
                
                # 4. Meta advantage
                meta_sl = meta_data[t, z, :, :]
                nometa_sl = nometa_data[t, z, :, :]
                mae_meta = np.mean(np.abs(meta_sl - hr_sl))
                mae_nometa = np.mean(np.abs(nometa_sl - hr_sl))
                advantage = mae_nometa - mae_meta
                
                if mae_meta > 0:
                    ratio = mae_nometa / mae_meta
                else:
                    ratio = 1.0
                
                # Combined score: we want high gradients, bidirectional flow, AND meta advantage
                complexity = grad_mag * dyn_range * (1 + bidir_score)
                combined = complexity * max(0, advantage) * 1000
                
                results.append({
                    'meas': meas, 't': t, 'z': z,
                    'grad_mag': grad_mag, 'bidir': bidir_score, 'dyn_range': dyn_range,
                    'complexity': complexity, 'advantage': advantage, 'ratio': ratio,
                    'combined': combined, 'max_abs': np.max(np.abs(hr_sl)),
                    'mae_meta': mae_meta, 'mae_nometa': mae_nometa
                })
    
    # Sort by combined score
    results.sort(key=lambda x: x['combined'], reverse=True)
    
    print(f"\nTop 30 slices by combined (complexity * meta advantage):")
    print(f"{'Rank':>4} {'meas':>4} {'t':>3} {'z':>4} {'grad':>6} {'bidir':>6} {'range':>6} {'max|v|':>7} {'MAE_M':>7} {'MAE_NM':>7} {'adv':>7} {'ratio':>6} {'combined':>9}")
    for i, r in enumerate(results[:30]):
        print(f"{i+1:4d} {r['meas']:>4} {r['t']:3d} {r['z']:4d} {r['grad_mag']:6.4f} {r['bidir']:6.3f} {r['dyn_range']:6.3f} {r['max_abs']:7.3f} {r['mae_meta']:7.5f} {r['mae_nometa']:7.5f} {r['advantage']:7.5f} {r['ratio']:6.2f} {r['combined']:9.4f}")

# ---- PART 2: Also look at x and y planes (sagittal/coronal) ----
print("\n=== PART 2: x and y plane slices ===")

with h5py.File(hr_path, 'r') as fhr, \
     h5py.File(meta_50, 'r') as fmeta, \
     h5py.File(nometa_50, 'r') as fnometa:
    
    results_planes = []
    
    for meas in ['w']:  # focus on w for now
        hr_data = fhr[meas][:]
        meta_data = fmeta[meas][:]
        nometa_data = fnometa[meas][:]
        
        for t in [4, 6, 8, 10, 12]:
            # x-plane slices (sagittal): data[t, z, y, x_idx] -> (z, y)
            for xi in range(15, 55):
                hr_sl = hr_data[t, :, :, xi]
                if np.max(np.abs(hr_sl)) < 0.1:
                    continue
                gy, gx = np.gradient(hr_sl)
                grad_mag = np.mean(np.sqrt(gx**2 + gy**2))
                dyn_range = np.max(hr_sl) - np.min(hr_sl)
                pos_frac = np.sum(hr_sl > 0.05) / max(1, np.sum(np.abs(hr_sl) > 0.05))
                bidir_score = 1.0 - abs(pos_frac - 0.5) * 2
                
                meta_sl = meta_data[t, :, :, xi]
                nometa_sl = nometa_data[t, :, :, xi]
                mae_meta = np.mean(np.abs(meta_sl - hr_sl))
                mae_nometa = np.mean(np.abs(nometa_sl - hr_sl))
                advantage = mae_nometa - mae_meta
                ratio = mae_nometa / mae_meta if mae_meta > 0 else 1
                complexity = grad_mag * dyn_range * (1 + bidir_score)
                combined = complexity * max(0, advantage) * 1000
                
                results_planes.append({
                    'plane': 'x', 'slice_idx': xi, 'meas': meas, 't': t,
                    'complexity': complexity, 'advantage': advantage, 'ratio': ratio,
                    'combined': combined, 'dyn_range': dyn_range, 'bidir': bidir_score,
                    'grad_mag': grad_mag, 'max_abs': np.max(np.abs(hr_sl))
                })
            
            # y-plane slices (coronal): data[t, z, y_idx, x] -> (z, x)
            for yi in range(10, 80):
                hr_sl = hr_data[t, :, yi, :]
                if np.max(np.abs(hr_sl)) < 0.1:
                    continue
                gy, gx = np.gradient(hr_sl)
                grad_mag = np.mean(np.sqrt(gx**2 + gy**2))
                dyn_range = np.max(hr_sl) - np.min(hr_sl)
                pos_frac = np.sum(hr_sl > 0.05) / max(1, np.sum(np.abs(hr_sl) > 0.05))
                bidir_score = 1.0 - abs(pos_frac - 0.5) * 2
                
                meta_sl = meta_data[t, :, yi, :]
                nometa_sl = nometa_data[t, :, yi, :]
                mae_meta = np.mean(np.abs(meta_sl - hr_sl))
                mae_nometa = np.mean(np.abs(nometa_sl - hr_sl))
                advantage = mae_nometa - mae_meta
                ratio = mae_nometa / mae_meta if mae_meta > 0 else 1
                complexity = grad_mag * dyn_range * (1 + bidir_score)
                combined = complexity * max(0, advantage) * 1000
                
                results_planes.append({
                    'plane': 'y', 'slice_idx': yi, 'meas': meas, 't': t,
                    'complexity': complexity, 'advantage': advantage, 'ratio': ratio,
                    'combined': combined, 'dyn_range': dyn_range, 'bidir': bidir_score,
                    'grad_mag': grad_mag, 'max_abs': np.max(np.abs(hr_sl))
                })
    
    results_planes.sort(key=lambda x: x['combined'], reverse=True)
    print(f"\nTop 20 x/y plane slices:")
    print(f"{'Rank':>4} {'plane':>5} {'idx':>4} {'t':>3} {'grad':>6} {'bidir':>6} {'range':>6} {'max|v|':>7} {'adv':>7} {'ratio':>6} {'combined':>9}")
    for i, r in enumerate(results_planes[:20]):
        print(f"{i+1:4d} {r['plane']:>5} {r['slice_idx']:4d} {r['t']:3d} {r['grad_mag']:6.4f} {r['bidir']:6.3f} {r['dyn_range']:6.3f} {r['max_abs']:7.3f} {r['advantage']:7.5f} {r['ratio']:6.2f} {r['combined']:9.4f}")


# ---- PART 3: Create visual comparison of top candidates ----
print("\n=== PART 3: Generating visual comparison ===")

# Pick top 3 unique (meas, t, z) from axial and top 3 from other planes
top_axial = []
seen_axial = set()
for r in results:
    key = (r['meas'], r['t'], r['z'])
    if key not in seen_axial and len(top_axial) < 4:
        top_axial.append(r)
        seen_axial.add(key)

top_other = []
seen_other = set()
for r in results_planes:
    key = (r['plane'], r['slice_idx'], r['t'])
    if key not in seen_other and len(top_other) < 3:
        top_other.append(r)
        seen_other.add(key)

print("\nTop axial candidates:")
for r in top_axial:
    print(f"  meas={r['meas']}, t={r['t']}, z={r['z']}, combined={r['combined']:.4f}")

print("\nTop other plane candidates:")  
for r in top_other:
    print(f"  plane={r['plane']}, idx={r['slice_idx']}, t={r['t']}, combined={r['combined']:.4f}")

# Now create a big figure showing the top candidates
with h5py.File(hr_path, 'r') as fhr, \
     h5py.File(meta_50, 'r') as fmeta, \
     h5py.File(nometa_50, 'r') as fnometa:
    
    all_candidates = []
    
    for r in top_axial:
        m, t, z = r['meas'], r['t'], r['z']
        hr_sl = fhr[m][t, z, :, :]
        meta_sl = fmeta[m][t, z, :, :]
        nometa_sl = fnometa[m][t, z, :, :]
        all_candidates.append((f"{m}, t={t}, z={z}\naxial", hr_sl, meta_sl, nometa_sl, r))
    
    for r in top_other:
        m, t = r['meas'], r['t']
        if r['plane'] == 'x':
            hr_sl = fhr[m][t, :, :, r['slice_idx']]
            meta_sl = fmeta[m][t, :, :, r['slice_idx']]
            nometa_sl = fnometa[m][t, :, :, r['slice_idx']]
        else:
            hr_sl = fhr[m][t, :, r['slice_idx'], :]
            meta_sl = fmeta[m][t, :, r['slice_idx'], :]
            nometa_sl = fnometa[m][t, :, r['slice_idx'], :]
        all_candidates.append((f"{m}, t={t}, {r['plane']}={r['slice_idx']}", hr_sl, meta_sl, nometa_sl, r))
    
    n = len(all_candidates)
    fig, axes = plt.subplots(n, 4, figsize=(16, n * 3.5))
    
    for row, (title, hr_sl, meta_sl, nometa_sl, r) in enumerate(all_candidates):
        vmin_auto = np.percentile(hr_sl[hr_sl != 0], 2) if np.any(hr_sl != 0) else -0.3
        vmax_auto = np.percentile(hr_sl[hr_sl != 0], 98) if np.any(hr_sl != 0) else 0.7
        
        for col, (label, sl) in enumerate([
            ('HR', hr_sl), ('Meta@50', meta_sl), ('NoMeta@50', nometa_sl),
            ('|NoMeta-Meta|', np.abs(nometa_sl - meta_sl))
        ]):
            ax = axes[row, col]
            if col < 3:
                ax.imshow(sl.T, cmap='viridis', origin='lower', vmin=vmin_auto, vmax=vmax_auto, interpolation='nearest')
            else:
                ax.imshow(sl.T, cmap='hot', origin='lower', interpolation='nearest')
            ax.axis('off')
            if row == 0:
                ax.set_title(label, fontsize=12)
            if col == 0:
                ax.set_ylabel(title, fontsize=9, rotation=0, labelpad=80, ha='right')
    
    plt.suptitle('Top candidate cross-sections for Meta vs NoMeta comparison (iteration 50)', fontsize=13, y=1.01)
    plt.tight_layout()
    os.makedirs('2D_plots', exist_ok=True)
    plt.savefig('2D_plots/candidate_cross_sections.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved: 2D_plots/candidate_cross_sections.png")


# ---- PART 4: For the top z-plane candidate, find optimal zoomed ROI ----
print("\n=== PART 4: Optimal ROI for top candidate ===")
best = top_axial[0]
print(f"Best candidate: meas={best['meas']}, t={best['t']}, z={best['z']}")

with h5py.File(hr_path, 'r') as fhr:
    hr_sl = fhr[best['meas']][best['t'], best['z'], :, :]  # (y, x)
    
    # Find vessel region: where |velocity| is significant
    mask = np.abs(hr_sl) > 0.05
    if np.any(mask):
        ys, xs = np.where(mask)
        y_center = (ys.min() + ys.max()) // 2
        x_center = (xs.min() + xs.max()) // 2
        y_extent = ys.max() - ys.min()
        x_extent = xs.max() - xs.min()
        
        # Try different ROI sizes centered on the vessel
        for roi_dim in [24, 28, 32, 36, 40]:
            y0 = max(0, y_center - roi_dim // 2)
            x0 = max(0, x_center - roi_dim // 2)
            y0 = min(y0, hr_sl.shape[0] - roi_dim)
            x0 = min(x0, hr_sl.shape[1] - roi_dim)
            roi = hr_sl[y0:y0+roi_dim, x0:x0+roi_dim]
            nonzero_frac = np.sum(np.abs(roi) > 0.02) / roi_dim**2
            print(f"  dim={roi_dim}: y_start={y0}, x_start={x0}, nonzero_fraction={nonzero_frac:.2f}, range=[{roi.min():.3f}, {roi.max():.3f}]")
        
        print(f"\n  Vessel center: y={y_center}, x={x_center}")
        print(f"  Vessel extent: y={y_extent}, x={x_extent}")
        print(f"  Bounding box: y=[{ys.min()}, {ys.max()}], x=[{xs.min()}, {xs.max()}]")

print("\nDone!")
