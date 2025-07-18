import numpy as np
import tensorflow as tf
import h5py
import loss_utils
import evaluation_utils as e_utils
import pandas as pd
from scipy.ndimage import zoom
from PIL import Image
import os
import sys
sys.path.append('../')


data_dir = '../../../data/icad_sim'

lr_filename = '../../../data/icad_sim/ICAD48_05mm_dv_lowSNR_x2.h5'

hr_filename = '../../../data/icad_sim/ICAD48_05mm.h5'   

prediction_dir = "../../results/icad/benchmark_metrics/ICAD48"

if not os.path.isdir(prediction_dir):
    os.makedirs(prediction_dir)

ground_truth_file = f"{hr_filename}"
lr_file = f"{lr_filename}"

peak_flow_idx = 0

t_start = 0
t_end = 0
x_start = 0
x_end =   140
y_start = 0
y_end =   102
z_start = 0
z_end =   200

resolution = 0.0005*2


ref_spatial_factor = 2

# Open HR file
with h5py.File(ground_truth_file, 'r') as hf:

    u_hr = np.asarray(hf['u'])
    v_hr = np.asarray(hf['v'])
    w_hr = np.asarray(hf['w'])
    end = np.shape(u_hr)
    x_end =end[1]
    y_end = end[2]
    z_end = end[3]
    

    u_hr = u_hr[t_start:t_end+1,x_start:x_end,y_start:y_end,z_start:z_end]
    v_hr = v_hr[t_start:t_end+1,x_start:x_end,y_start:y_end,z_start:z_end]
    w_hr = w_hr[t_start:t_end+1,x_start:x_end,y_start:y_end,z_start:z_end]

    T = len(u_hr)

    mask = np.asarray(hf['mask'])
    if len(mask.shape) == 4: 
        mask = mask[0]
    mask = mask[x_start:x_end,y_start:y_end,z_start:z_end]

    nf_mask = 1.0 - mask
    boundary_mask, core_mask = e_utils.create_boundary_and_core_masks(mask, 0.1, 'voxels')

    X,Y,Z = mask.shape
    cov_a = np.sum(mask)/(X*Y*Z)
    cov_b = np.sum(boundary_mask)/(X*Y*Z)
    cov_c = np.sum(core_mask)/(X*Y*Z)
    ratio_b = np.sum(boundary_mask)/np.sum(mask)
    ratio_c = np.sum(core_mask)/np.sum(mask)

    print(' ')
    print(f'Coverage: {100*cov_a:.3f} %')
    print(f'Boundary --- cov: {100*cov_b:.3f} %, ratio: {100*ratio_b:.3f} %')
    print(f'Core --- cov: {100*cov_c:.3f} %, ratio: {100*ratio_c:.3f} %')

# Open LR file
with h5py.File(lr_file, mode = 'r') as pf:
    u_lr = np.asarray(pf['u']) 
    v_lr = np.asarray(pf['v'])
    w_lr = np.asarray(pf['w'])

u_lr = u_lr[t_start:t_end+1,int(x_start/ref_spatial_factor):int(x_end/ref_spatial_factor),int(y_start/ref_spatial_factor):int(y_end/ref_spatial_factor),int(z_start/ref_spatial_factor):int(z_end/ref_spatial_factor)]
v_lr = v_lr[t_start:t_end+1,int(x_start/ref_spatial_factor):int(x_end/ref_spatial_factor),int(y_start/ref_spatial_factor):int(y_end/ref_spatial_factor),int(z_start/ref_spatial_factor):int(z_end/ref_spatial_factor)]
w_lr = w_lr[t_start:t_end+1,int(x_start/ref_spatial_factor):int(x_end/ref_spatial_factor),int(y_start/ref_spatial_factor):int(y_end/ref_spatial_factor),int(z_start/ref_spatial_factor):int(z_end/ref_spatial_factor)]

# Function for Lanczos (Sinc) Interpolation
def lanczos_upsampling(volume):
    upsampled_volume = np.zeros((volume.shape[0], volume.shape[1]*2, volume.shape[2]*2, volume.shape[3]*2))
    for t in range(volume.shape[0]):
        for z in range(volume.shape[3]):
            img = Image.fromarray(volume[t,:,:,z])
            img = img.resize((volume.shape[2]*2, volume.shape[1]*2), Image.LANCZOS)
            upsampled_volume[t,:,:,z] = np.array(img)
    return upsampled_volume

# Function for Bicubic Interpolation
def bicubic_upsampling(volume):
    return zoom(volume, (1, 2, 2, 2), order=3)

# Apply upsampling
print(u_lr.shape)

#u_lr = lanczos_upsampling(u_lr)
#v_lr = lanczos_upsampling(v_lr)
#w_lr = lanczos_upsampling(w_lr)

u_lr = bicubic_upsampling(u_lr)
v_lr = bicubic_upsampling(v_lr)
w_lr = bicubic_upsampling(w_lr)


rel_err = np.zeros((T,3))
abs_err = np.zeros((T,4))
rmse = np.zeros((T,4))

vnrmse = np.zeros((T,4))
d_error = np.zeros((T,4))
div_err = np.zeros((T,4))

Ks = np.zeros((T,3,3))
Ms = np.zeros((T,3,3))
Rs = np.zeros((T,3,3))

for t in range(T):
    print(np.shape(u_lr),np.shape(u_hr))
    rel_err[t,0] = (e_utils.calculate_relative_error(u_lr[t], v_lr[t], w_lr[t], u_hr[t], v_hr[t], w_hr[t], mask))
    rel_err[t,1] = (e_utils.calculate_relative_error(u_lr[t], v_lr[t], w_lr[t], u_hr[t], v_hr[t], w_hr[t], boundary_mask))
    rel_err[t,2] = (e_utils.calculate_relative_error(u_lr[t], v_lr[t], w_lr[t], u_hr[t], v_hr[t], w_hr[t], core_mask))

    abs_err[t,0] = (e_utils.calculate_absolute_error(u_lr[t], v_lr[t], w_lr[t], u_hr[t], v_hr[t], w_hr[t], mask))
    abs_err[t,1] = (e_utils.calculate_absolute_error(u_lr[t], v_lr[t], w_lr[t], u_hr[t], v_hr[t], w_hr[t], boundary_mask))
    abs_err[t,2] = (e_utils.calculate_absolute_error(u_lr[t], v_lr[t], w_lr[t], u_hr[t], v_hr[t], w_hr[t], core_mask))
    abs_err[t,3] = (e_utils.calculate_absolute_error(u_lr[t], v_lr[t], w_lr[t], u_hr[t], v_hr[t], w_hr[t], nf_mask))

    rmse[t,0] = (e_utils.calculate_rmse(u_lr[t], v_lr[t], w_lr[t], u_hr[t], v_hr[t], w_hr[t], mask))
    rmse[t,1] = (e_utils.calculate_rmse(u_lr[t], v_lr[t], w_lr[t], u_hr[t], v_hr[t], w_hr[t], boundary_mask))
    rmse[t,2] = (e_utils.calculate_rmse(u_lr[t], v_lr[t], w_lr[t], u_hr[t], v_hr[t], w_hr[t], core_mask))
    rmse[t,3] = (e_utils.calculate_rmse(u_lr[t], v_lr[t], w_lr[t], u_hr[t], v_hr[t], w_hr[t], nf_mask))

    ### OPTIONALLY - ADD MORE METRICS

    vnrmse[t,0] = (e_utils.calculate_vnrmse(u_lr[t], v_lr[t], w_lr[t], u_hr[t], v_hr[t], w_hr[t], mask))
    vnrmse[t,1] = (e_utils.calculate_vnrmse(u_lr[t], v_lr[t], w_lr[t], u_hr[t], v_hr[t], w_hr[t], boundary_mask))
    vnrmse[t,2] = (e_utils.calculate_vnrmse(u_lr[t], v_lr[t], w_lr[t], u_hr[t], v_hr[t], w_hr[t], core_mask))
    vnrmse[t,3] = (e_utils.calculate_vnrmse(u_lr[t], v_lr[t], w_lr[t], u_hr[t], v_hr[t], w_hr[t], nf_mask))

    d_error[t,0] = (e_utils.calculate_directional_error(u_lr[t], v_lr[t], w_lr[t], u_hr[t], v_hr[t], w_hr[t], mask))
    d_error[t,1] = (e_utils.calculate_directional_error(u_lr[t], v_lr[t], w_lr[t], u_hr[t], v_hr[t], w_hr[t], boundary_mask))
    d_error[t,2] = (e_utils.calculate_directional_error(u_lr[t], v_lr[t], w_lr[t], u_hr[t], v_hr[t], w_hr[t], core_mask))
    d_error[t,3] = (e_utils.calculate_directional_error(u_lr[t], v_lr[t], w_lr[t], u_hr[t], v_hr[t], w_hr[t], nf_mask))

    div_err[t,0] = (e_utils.calculate_divergence([u_lr[t], v_lr[t], w_lr[t]], [resolution, resolution, resolution], mask))
    div_err[t,1] = (e_utils.calculate_divergence([u_lr[t], v_lr[t], w_lr[t]], [resolution, resolution, resolution], boundary_mask))
    div_err[t,2] = (e_utils.calculate_divergence([u_lr[t], v_lr[t], w_lr[t]], [resolution, resolution, resolution], core_mask))
    div_err[t,3] = (e_utils.calculate_divergence([u_lr[t], v_lr[t], w_lr[t]], [resolution, resolution, resolution], nf_mask))


    Ks[t][0][0], Ms[t][0][0], Rs[t][0][0] = e_utils.linreg(u_lr[t], u_hr[t], mask)
    Ks[t][1][0], Ms[t][1][0], Rs[t][1][0] = e_utils.linreg(v_lr[t], v_hr[t], mask)
    Ks[t][2][0], Ms[t][2][0], Rs[t][2][0] = e_utils.linreg(w_lr[t], w_hr[t], mask)

    Ks[t][0][1], Ms[t][0][1], Rs[t][0][1] = e_utils.linreg(u_lr[t], u_hr[t], boundary_mask)
    Ks[t][1][1], Ms[t][1][1], Rs[t][1][1] = e_utils.linreg(v_lr[t], v_hr[t], boundary_mask)
    Ks[t][2][1], Ms[t][2][1], Rs[t][2][1] = e_utils.linreg(w_lr[t], w_hr[t], boundary_mask)

    Ks[t][0][2], Ms[t][0][2], Rs[t][0][2] = e_utils.linreg(u_lr[t], u_hr[t], core_mask)
    Ks[t][1][2], Ms[t][1][2], Rs[t][1][2] = e_utils.linreg(v_lr[t], v_hr[t], core_mask)
    Ks[t][2][2], Ms[t][2][2], Rs[t][2][2] = e_utils.linreg(w_lr[t], w_hr[t], core_mask)


print('Total avg')
rel_err_tot = np.mean(rel_err, axis=0)
print(f'Relative error [Fluid] {rel_err_tot[0]:.1f}')
print(f'Relative error [Bound] {rel_err_tot[1]:.1f}')
print(f'Relative error [Core] {rel_err_tot[2]:.1f}')

abs_err_tot = np.mean(abs_err, axis=0)
print(f'Absolute error [Fluid] {abs_err_tot[0]:.1f}')
print(f'Absolute error [Bound] {abs_err_tot[1]:.1f}')
print(f'Absolute error [Core] {abs_err_tot[2]:.1f}')
print(f'Absolute error [Non-F] {abs_err_tot[3]:.1f}')

rmse_tot = np.mean(rmse, axis=0)
print(f'R.M.S.   error [Fluid] {rmse_tot[0]:.1f}')
print(f'R.M.S.   error [Bound] {rmse_tot[1]:.1f}')
print(f'R.M.S.   error [Core] {rmse_tot[2]:.1f}')
print(f'R.M.S.   error [Non-F] {rmse_tot[3]:.1f}')

print('-  '*9)
print('Peak Flow')

print(f'Relative error [Fluid] {rel_err[peak_flow_idx][0]:.1f}')
print(f'Relative error [Bound] {rel_err[peak_flow_idx][1]:.1f}')
print(f'Relative error [Core] {rel_err[peak_flow_idx][2]:.1f}')

print(f'Absolute error [Fluid] {abs_err[peak_flow_idx][0]:.1f}')
print(f'Absolute error [Bound] {abs_err[peak_flow_idx][1]:.1f}')
print(f'Absolute error [Core] {abs_err[peak_flow_idx][2]:.1f}')
print(f'Absolute error [Non-F] {abs_err[peak_flow_idx][3]:.1f}')

print(f'R.M.S.   error [Fluid] {rmse[peak_flow_idx][0]:.1f}')
print(f'R.M.S.   error [Bound] {rmse[peak_flow_idx][1]:.1f}')
print(f'R.M.S.   error [Core] {rmse[peak_flow_idx][2]:.1f}')
print(f'R.M.S.   error [Non-F] {rmse[peak_flow_idx][3]:.1f}')

print(' ')
print(f'U [Fluid] k: {Ks[peak_flow_idx][0][0]:.4f} \t m: {Ms[peak_flow_idx][0][0]:.4f} \t r^2: {Rs[peak_flow_idx][0][0]:.4f}')
print(f'  [Bound] k: {Ks[peak_flow_idx][0][1]:.4f} \t m: {Ms[peak_flow_idx][0][1]:.4f} \t r^2: {Rs[peak_flow_idx][0][1]:.4f}')
print(f'  [Core] k: {Ks[peak_flow_idx][0][2]:.4f} \t m: {Ms[peak_flow_idx][0][2]:.4f} \t r^2: {Rs[peak_flow_idx][0][2]:.4f}')

print(' ')
print(f'V [Fluid] k: {Ks[peak_flow_idx][1][0]:.4f} \t m: {Ms[peak_flow_idx][1][0]:.4f} \t r^2: {Rs[peak_flow_idx][1][0]:.4f}')
print(f'  [Bound] k: {Ks[peak_flow_idx][1][1]:.4f} \t m: {Ms[peak_flow_idx][1][1]:.4f} \t r^2: {Rs[peak_flow_idx][1][1]:.4f}')
print(f'  [Core] k: {Ks[peak_flow_idx][1][2]:.4f} \t m: {Ms[peak_flow_idx][1][2]:.4f} \t r^2: {Rs[peak_flow_idx][1][2]:.4f}')

print(' ')
print(f'W [Fluid] k: {Ks[peak_flow_idx][2][0]:.4f} \t m: {Ms[peak_flow_idx][2][0]:.4f} \t r^2: {Rs[peak_flow_idx][2][0]:.4f}')
print(f'  [Bound] k: {Ks[peak_flow_idx][2][1]:.4f} \t m: {Ms[peak_flow_idx][2][1]:.4f} \t r^2: {Rs[peak_flow_idx][2][1]:.4f}')
print(f'  [Core] k: {Ks[peak_flow_idx][2][2]:.4f} \t m: {Ms[peak_flow_idx][2][2]:.4f} \t r^2: {Rs[peak_flow_idx][2][2]:.4f}')

# Save metrics to csv
metrics = {
    'lr_filename': lr_filename,
    'hr_filename': hr_filename,

    'Coverage [%]': 100*cov_a,
    'Fluid Coverage [%]': 100*cov_b,
    'Core Coverage [%]': 100*cov_c,
    'Ratio Boundary/Core [%]': 100*ratio_c,

    'Relative error [Fluid]': rel_err_tot[0],
    'Relative error [Bound]': rel_err_tot[1],
    'Relative error [Core]': rel_err_tot[2],
    'Absolute error [Fluid]': abs_err_tot[0],
    'Absolute error [Bound]': abs_err_tot[1],
    'Absolute error [Core]': abs_err_tot[2],
    'Absolute error [Non-F]': abs_err_tot[3],
    'R.M.S. error [Fluid]': rmse_tot[0],
    'R.M.S. error [Bound]': rmse_tot[1],
    'R.M.S. error [Core]': rmse_tot[2],
    'R.M.S. error [Non-F]': rmse_tot[3],

    'PEAK FLOW INDEX:': peak_flow_idx,
    'Relative error [Fluid] Peak': rel_err[peak_flow_idx][0],
    'Relative error [Bound] Peak': rel_err[peak_flow_idx][1],
    'Relative error [Core] Peak': rel_err[peak_flow_idx][2],
    'Absolute error [Fluid] Peak': abs_err[peak_flow_idx][0],
    'Absolute error [Bound] Peak': abs_err[peak_flow_idx][1],
    'Absolute error [Core] Peak': abs_err[peak_flow_idx][2],
    'Absolute error [Non-F] Peak': abs_err[peak_flow_idx][3],
    'R.M.S. error [Fluid] Peak': rmse[peak_flow_idx][0],
    'R.M.S. error [Bound] Peak': rmse[peak_flow_idx][1],
    'R.M.S. error [Core] Peak': rmse[peak_flow_idx][2],
    'R.M.S. error [Non-F] Peak': rmse[peak_flow_idx][3],

    'VNRMSE [Fluid]': vnrmse[0,0],
    'VNRMSE [Bound]': vnrmse[0,1],
    'VNRMSE [Core]': vnrmse[0,2],
    'VNRMSE [Non-F]': vnrmse[0,3],

    'Directional error [Fluid]': d_error[0,0],
    'Directional error [Bound]': d_error[0,1],
    'Directional error [Core]': d_error[0,2],
    'Directional error [Non-F]': d_error[0,3],

    'Divergence prediction [Fluid]': div_err[0,0],
    'Divergence prediction [Bound]': div_err[0,1],
    'Divergence prediction [Core]': div_err[0,2],
    'Divergence prediction [Non-F]': div_err[0,3],

    'U [Fluid] k': Ks[peak_flow_idx][0][0],
    'U [Bound] k': Ks[peak_flow_idx][0][1],
    'U [Core] k': Ks[peak_flow_idx][0][2],
    'U [Fluid] m': Ms[peak_flow_idx][0][0],
    'U [Bound] m': Ms[peak_flow_idx][0][1],
    'U [Core] m': Ms[peak_flow_idx][0][2],
    'U [Fluid] r^2': Rs[peak_flow_idx][0][0],
    'U [Bound] r^2': Rs[peak_flow_idx][0][1],
    'U [Core] r^2': Rs[peak_flow_idx][0][2],

    'V [Fluid] k': Ks[peak_flow_idx][1][0],
    'V [Bound] k': Ks[peak_flow_idx][1][1],
    'V [Core] k': Ks[peak_flow_idx][1][2],
    'V [Fluid] m': Ms[peak_flow_idx][1][0],
    'V [Bound] m': Ms[peak_flow_idx][1][1],
    'V [Core] m': Ms[peak_flow_idx][1][2],
    'V [Fluid] r^2': Rs[peak_flow_idx][1][0],
    'V [Bound] r^2': Rs[peak_flow_idx][1][1],
    'V [Core] r^2': Rs[peak_flow_idx][1][2],


    'W [Fluid] k': Ks[peak_flow_idx][2][0],
    'W [Bound] k': Ks[peak_flow_idx][2][1],
    'W [Core] k': Ks[peak_flow_idx][2][2],
    'W [Fluid] m': Ms[peak_flow_idx][2][0],
    'W [Bound] m': Ms[peak_flow_idx][2][1],
    'W [Core] m': Ms[peak_flow_idx][2][2],
    'W [Fluid] r^2': Rs[peak_flow_idx][2][0],
    'W [Bound] r^2': Rs[peak_flow_idx][2][1],
    'W [Core] r^2': Rs[peak_flow_idx][2][2],

}

metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
metrics_filename = f'{prediction_dir}/benchmark_metrics.csv'
metrics_df.to_csv(metrics_filename, index=False)

def save_to_h5(output_filepath, col_name, dataset, expand=False):
    if expand:
        dataset = np.expand_dims(dataset, axis=0)

    # convert float64 to float32 to save space
    if dataset.dtype == 'float64':
        dataset = np.array(dataset, dtype='float32')
    
    with h5py.File(output_filepath, 'a') as hf:    
        if col_name not in hf:
            datashape = (None, )
            if (dataset.ndim > 1):
                datashape = (None, ) + dataset.shape[1:]
            hf.create_dataset(col_name, data=dataset, maxshape=datashape)
        else:
            hf[col_name].resize((hf[col_name].shape[0]) + dataset.shape[0], axis = 0)
            hf[col_name][-dataset.shape[0]:] = dataset



save_to_h5(f"{prediction_dir}/healthy-05mm3_dv_lowSNR_x2_interpolated.h5", "u", u_lr[peak_flow_idx]*mask)
save_to_h5(f"{prediction_dir}/healthy-05mm3_dv_lowSNR_x2_interpolated.h5", "v", v_lr[peak_flow_idx]*mask)
save_to_h5(f"{prediction_dir}/healthy-05mm3_dv_lowSNR_x2_interpolated.h5", "w", w_lr[peak_flow_idx]*mask)