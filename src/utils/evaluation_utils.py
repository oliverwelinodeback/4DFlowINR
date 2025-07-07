import matplotlib.pyplot as plt
import matplotlib.cm as cm
import h5py
import numpy as np
from scipy import stats
from skimage import morphology
from PIL import Image

def crop(hr, pred):
    # We assume that if there is a mismatch it's because SR is smaller than HR.
    crop = np.asarray(hr.shape) - np.asarray(pred.shape)
    hr = hr[crop[0]//2:-crop[0]//2,:,:] if crop[0] else hr
    hr = hr[:, crop[1]//2:-crop[1]//2,:] if crop[1] else hr
    hr = hr[:, :, crop[2]//2:-crop[2]//2] if crop[2] else hr
    return hr
 
def get_slice_values(body, idx, axis='x'):
    if axis=='x':
        vals = body[idx, :, :]
    elif axis=='y':
        vals = body[:, idx, :]
    elif axis=='z':
        vals = body[:,:,idx]
    else:
        print("Error: x, y, z are available axes")
        return
    return vals

def slice(ax, file, frame, idx, vel_dir='u', axis='x'):
    with h5py.File(file, mode = 'r' ) as hf:
        body = np.asarray(hf[vel_dir][frame])
        sliced = get_slice_values(body, idx, axis)
    ax.imshow(sliced, interpolation='nearest', cmap='viridis', origin='lower')

def plot_relative_mean_error(relative_mean_error, N, save_file,fig_nr):
    print(f"Plotting relative mean error...")
    plt.figure(fig_nr)
    if N == 1:
        plt.scatter(N, relative_mean_error)
    else:
        plt.plot(np.arange(N), relative_mean_error)
    plt.xlabel("Frame")
    plt.ylabel("Relative error (%)")
    plt.title("Relative speed error")
    plt.savefig(f"{save_file[:-3]}_RME.png")
    return fig_nr + 1

def plot_mean_speed_old(mean_speed, N, save_file, fig_nr):
    plt.figure(fig_nr)
    fig_nr += 1
    if N == 1:
        plt.scatter(N, mean_speed)
    else:
        plt.plot(np.arange(N), mean_speed)
    plt.xlabel("Frame")
    plt.ylabel("Avg. speed (cm/s)")
    plt.savefig(f"{save_file[:-3]}_speed.png")
    return fig_nr

def plot_mean_speed(mean_speed, N, save_file, fig_nr):
    print("Plotting average speed...")
    plt.figure(fig_nr)
    fig, ax = plt.subplots()

    colors = ['r', 'g', 'b', 'y']
    labels = ['$\mathregular{|V|}$', '$\mathregular{V_x}$', '$\mathregular{V_y}$', '$\mathregular{V_z}$']
    for i in range(4):
        ax.plot(np.arange(N), mean_speed[:, i], color=colors[i], label=labels[i])

    ax.set_xlabel("Frames")
    ax.set_ylabel("Avg. speed (cm/s)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{save_file[:-3]}_speed.png")
    plt.show()
    return fig_nr + 1

def reg_stats(hr_vals, sr_vals):
    reg = stats.linregress(hr_vals, sr_vals)
    x = np.array([-10, 10]) # Start, End point for the regression slope lines
    if reg.intercept < 0.0:
        reg_stats = f'$y = {reg.slope:.3f}x {reg.intercept:.3f}$\n$r^2 = {reg.rvalue**2:.3f}$'
    else:
        reg_stats = f'$y = {reg.slope:.3f}x + {reg.intercept:.3f}$\n$r^2 = {reg.rvalue**2:.3f}$'
    plt.plot(x, reg.intercept + reg.slope*x, 'k', linestyle='--', alpha=0.3)
    return reg_stats

def plot_linear_regression(fig_nr, dimension, boundary_hr_vals, boundary_sr_vals, core_hr_vals, core_sr_vals):
    plt.figure(fig_nr)
    
    # Set limits and ticks
    xlim = ylim = 1.0
    plt.xlim(-xlim,xlim); plt.ylim(-ylim,ylim); plt.xticks([-xlim, xlim]); plt.yticks([-ylim, ylim])
    
    plt.scatter(core_hr_vals[::5], core_sr_vals[::5], s=0.8, c=["black"])
    plt.scatter(boundary_hr_vals[::5], boundary_sr_vals[::5], s=0.2, c=["red"])
    
    boundary_reg_stats = reg_stats(boundary_hr_vals, boundary_sr_vals)
    plt.text(-xlim/2, ylim/2, boundary_reg_stats, horizontalalignment='center', verticalalignment='bottom', fontsize=10, color='red')
    core_reg_stats = reg_stats(core_hr_vals, core_sr_vals)
    plt.text(xlim/2, -ylim/2, core_reg_stats, horizontalalignment='center', verticalalignment='top', fontsize=10, color="black")
    
    # Set title and labels
    plt.title(f"Correlation in V_{dimension}"); plt.xlabel("V_HR [m/s]"); plt.ylabel("V_SR [m/s]")

def create_boundary_and_core_masks(binary_mask, cut_off, cut_off_type='voxels'):
    if cut_off < 1:
        selem = None
    elif cut_off_type == 'voxels':
        selem = np.ones((cut_off, cut_off, cut_off))
    elif cut_off_type == 'percentage':
        cut_off = int(np.ceil(cut_off * np.min(binary_mask.shape)))
        selem = np.ones((cut_off, cut_off, cut_off))
    else:
        raise ValueError("Invalid cut_off_type, choose 'voxels' or 'percentage'")
    
    core_mask = 1.0-morphology.binary_dilation(1.0-binary_mask, selem)
    boundary_mask = np.logical_xor(binary_mask, core_mask)
    return boundary_mask.astype(np.int32), core_mask.astype(np.int32)

def _sample_hrsr(ground_truth_file, prediction_file, mask, peak_flow_idx, ratio):
    # Use mask to find interesting samples
    sample_pot = np.where(mask == 1)
    rng = np.random.default_rng(123)

    # Sample <ratio> samples
    sample_idx = rng.choice(len(sample_pot[0]), replace=False, size=(int(len(sample_pot[0])*ratio)))

    # Get indexes
    x_idx = sample_pot[0][sample_idx]
    y_idx = sample_pot[1][sample_idx]
    z_idx = sample_pot[2][sample_idx]

    with h5py.File(prediction_file, mode = 'r' ) as hf:
        sr_u = np.asarray(hf['u'][peak_flow_idx])
        sr_u_vals = sr_u[x_idx, y_idx, z_idx]
        sr_v = np.asarray(hf['v'][peak_flow_idx])
        sr_v_vals = sr_v[x_idx, y_idx, z_idx]
        sr_w = np.asarray(hf['w'][peak_flow_idx])
        sr_w_vals = sr_w[x_idx, y_idx, z_idx]
        
    with h5py.File(ground_truth_file, mode = 'r' ) as hf:
        # Get velocity values in all directions
        hr_u = crop(np.asarray(hf['u'][peak_flow_idx]), sr_u)
        hr_u_vals = hr_u[x_idx, y_idx, z_idx]
        hr_v = crop(np.asarray(hf['v'][peak_flow_idx]), sr_v)
        hr_v_vals = hr_v[x_idx, y_idx, z_idx]
        hr_w = crop(np.asarray(hf['w'][peak_flow_idx]), sr_w)
        hr_w_vals = hr_w[x_idx, y_idx, z_idx] 
        
    return [hr_u_vals, hr_v_vals, hr_w_vals], [sr_u_vals, sr_v_vals, sr_w_vals]
    
def draw_reg_line(ground_truth_file, prediction_file, peak_flow_idx, binary_mask, fig_nr):
    """ Plot a linear regression between HR and predicted SR in peak flow frame """
    #
    # Parameters
    #
    fig_nr += 1
    ratio = 1.0 #0.1
    
    boundary_mask, core_mask = create_boundary_and_core_masks(binary_mask, 0.1, 'voxels',) 
    
    boundary_hr, boundary_sr = _sample_hrsr(ground_truth_file, prediction_file, boundary_mask, peak_flow_idx, ratio)
    core_hr, core_sr = _sample_hrsr(ground_truth_file, prediction_file, core_mask, peak_flow_idx, ratio)

    print(f"Plotting regression lines...")
    
    plot_linear_regression(fig_nr, "x", boundary_hr[0], boundary_sr[0], core_hr[0], core_sr[0])
    plt.savefig(f"{prediction_file[:-3]}_LRXplot.png")
    plot_linear_regression(fig_nr+1, "y", boundary_hr[1], boundary_sr[1], core_hr[1], core_sr[1])
    plt.savefig(f"{prediction_file[:-3]}_LRYplot.png")
    plot_linear_regression(fig_nr+2, "z", boundary_hr[2], boundary_sr[2], core_hr[2], core_sr[2])
    plt.savefig(f"{prediction_file[:-3]}_LRZplot.png")
    return fig_nr+3

def calculate_absolute_error(u_pred, v_pred, w_pred, u_hi, v_hi, w_hi, mask=None):
    u_diff = np.square(u_pred - u_hi)
    v_diff = np.square(v_pred - v_hi)
    w_diff = np.square(w_pred - w_hi)

    diff_speed = np.sqrt(u_diff + v_diff + w_diff)

    abs_err = np.sum(diff_speed*mask) / (np.sum(mask) + 1) if mask is not None else np.mean(diff_speed)
    return abs_err

def calculate_absolute_error_pressure(p_pred, p_hi, mask=None):
    p_diff = np.square(p_pred - p_hi)
    diff_speed = np.sqrt(p_diff)
    abs_err = np.sum(diff_speed*mask) / (np.sum(mask) + 1) if mask is not None else np.mean(diff_speed)
    return abs_err

def calculate_relative_error(u_pred, v_pred, w_pred, u_hi, v_hi, w_hi, binary_mask):
    # if epsilon is set to 0, we will get nan and inf
    epsilon = 1e-5

    u_diff = np.square(u_pred - u_hi)
    v_diff = np.square(v_pred - v_hi)
    w_diff = np.square(w_pred - w_hi)

    diff_speed = np.sqrt(u_diff + v_diff + w_diff)
    actual_speed = np.sqrt(np.square(u_hi) + np.square(v_hi) + np.square(w_hi)) 

    # actual speed can be 0, resulting in inf
    relative_speed_loss = diff_speed / (actual_speed + epsilon)
    
    # Make sure the range is between 0 and 1
    relative_speed_loss = np.clip(relative_speed_loss, 0., 1.)

    # Apply correction, only use the diff speed if actual speed is zero
    condition = np.not_equal(actual_speed, 0.)
    corrected_speed_loss = np.where(condition, relative_speed_loss, diff_speed)

    multiplier = 1e4 # round it so we don't get any infinitesimal number
    corrected_speed_loss = np.round(corrected_speed_loss * multiplier) / multiplier
    # print(corrected_speed_loss)
    
    # Apply mask
    # binary_mask_condition = (mask > threshold)
    binary_mask_condition = np.equal(binary_mask, 1.0)          
    corrected_speed_loss = np.where(binary_mask_condition, corrected_speed_loss, np.zeros_like(corrected_speed_loss))
    # print(found_indexes)

    # Calculate the mean from the total non zero accuracy, divided by the masked area
    # reduce first to the 'batch' axis
    mean_err = np.sum(corrected_speed_loss) / (np.sum(binary_mask) + 1) 

    # now take the actual mean
    # mean_err = np.mean(mean_err) * 100 # in percentage
    mean_err = mean_err * 100

    return mean_err

def calculate_rmse(u_pred, v_pred, w_pred, u_hi, v_hi, w_hi, mask=None):
    u_diff = np.square(u_pred - u_hi)
    v_diff = np.square(v_pred - v_hi)
    w_diff = np.square(w_pred - w_hi)

    diff_speed = u_diff + v_diff + w_diff

    mse = np.sum(diff_speed*mask) / (np.sum(mask)) if mask is not None else np.mean(diff_speed)
    return np.sqrt(mse)

def calculate_rmse_pressure(p_pred, p_hi, mask=None):
    p_diff = np.square(p_pred - p_hi)

    mse = np.sum(p_diff*mask) / (np.sum(mask)) if mask is not None else np.mean(p_diff)
    return np.sqrt(mse)

def linreg(sr, hr, mask):
    hr_vals = hr[mask > 0.5]
    sr_vals = sr[mask > 0.5]
    reg = stats.linregress(hr_vals, sr_vals)
    return reg.slope, reg.intercept, reg.rvalue**2

def calculate_gradient_absolute_error(px_pred, py_pred, pz_pred, px_ref, py_ref, pz_ref, mask):
    diff_x = px_pred - px_ref
    diff_y = py_pred - py_ref
    diff_z = pz_pred - pz_ref
    mag = np.sqrt(diff_x**2 + diff_y**2 + diff_z**2)
    return np.mean(mag[mask==1])

def calculate_gradient_relative_error(px_pred, py_pred, pz_pred, px_ref, py_ref, pz_ref, mask, eps=1e-6):
    error_mag = np.sqrt((px_pred - px_ref)**2 + (py_pred - py_ref)**2 + (pz_pred - pz_ref)**2)
    ref_mag = np.sqrt(px_ref**2 + py_ref**2 + pz_ref**2)
    rel = error_mag / (ref_mag + eps)
    return np.mean(rel[mask==1])

def calculate_gradient_directional_error(px_pred, py_pred, pz_pred, px_ref, py_ref, pz_ref, mask):
    pred_vec = np.stack([px_pred, py_pred, pz_pred], axis=-1)
    ref_vec = np.stack([px_ref, py_ref, pz_ref], axis=-1)
    dot = np.sum(pred_vec * ref_vec, axis=-1)
    pred_norm = np.linalg.norm(pred_vec, axis=-1)
    ref_norm = np.linalg.norm(ref_vec, axis=-1)
    cos_theta = np.clip(dot / (pred_norm * ref_norm + 1e-6), -1.0, 1.0)
    angle = np.arccos(cos_theta)  # radians
    return np.degrees(np.mean(angle[mask==1]))

def calculate_gradient_nrmse(px_pred, py_pred, pz_pred, px_ref, py_ref, pz_ref, mask):
    pred_mag = np.sqrt(px_pred**2 + py_pred**2 + pz_pred**2)
    ref_mag = np.sqrt(px_ref**2 + py_ref**2 + pz_ref**2)
    mse = np.mean(((pred_mag - ref_mag)**2)[mask==1])
    nrmse = np.sqrt(mse) / (np.max(ref_mag[mask==1]) + 1e-6)
    return nrmse