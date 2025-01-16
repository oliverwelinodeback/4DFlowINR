import numpy as np
import os
import h5py
import random as rnd
import matplotlib.pyplot as plt
import preprocessing as pre
import time
import fft_downsampling as fft
import h5_utils as h5utils
import scipy.ndimage as ndimage
import pandas as pd
from scipy.signal import tukey
from scipy.fft import fftn, ifftn, fftshift, ifftshift

def apply_tukey_filter(image, alpha=0.5):
    """
    Apply a Tukey filter to a 3D magnitude image without reducing resolution.
    
    Args:
        image (np.ndarray): 3D magnitude image to filter.
        alpha (float): Fraction of the Tukey window that is tapered (0 <= alpha <= 1).
    
    Returns:
        np.ndarray: Filtered 3D magnitude image.
    """
    # Get image shape
    shape = image.shape

    # Transform the image to K-space
    kspace = fftshift(fftn(image))

    # Create a 3D Tukey filter
    tukey_x = tukey(shape[0], alpha)
    tukey_y = tukey(shape[1], alpha)
    tukey_z = tukey(shape[2], alpha)
    tukey_3d = np.outer(tukey_x, tukey_y).reshape(shape[0], shape[1], 1) * tukey_z.reshape(1, 1, shape[2])

    # Apply the Tukey filter in K-space
    filtered_kspace = kspace * tukey_3d

    # Transform back to image space
    filtered_image = np.abs(ifftn(ifftshift(filtered_kspace)))

    return filtered_image

def calculate_pad(arr_shape, upsample_rate):
    """
        Calculate padding to ensure the size is halved when using k-space downsample
    """
    divisor = upsample_rate * 2 # double the downsample rate because of half k-space center
    pad_x = arr_shape[0] % divisor
    pad_y = arr_shape[1] % divisor
    pad_z = arr_shape[2] % divisor
    print('pad_x', pad_x)
    print('pad_y', pad_y)
    print('pad_z', pad_z)
    print('---')

    pad_x = (0 if pad_x==0 else divisor - pad_x)
    pad_y = (0 if pad_y==0 else divisor - pad_y)
    pad_z = (0 if pad_z==0 else divisor - pad_z)
    print('pad_x', pad_x)
    print('pad_y', pad_y)
    print('pad_z', pad_z)
    return pad_x, pad_y, pad_z

def pad(u, x, y, z):
    return np.pad(u, ((0,x),(0,y), (0,z) ), 'constant')

def unpad(u, x, y, z):
    # https://stackoverflow.com/questions/21913935/numpy-negative-indexing-a-0
    return u[:-x or None,:-y  or None,:-z  or None]

def scale_and_repeat(img, target_img):
    # calculate the scale to adjust the template
    scale_x = target_img.shape[0] / img.shape[0]
    scale_y = target_img.shape[1] / img.shape[1]
    scale_z = target_img.shape[2] / img.shape[2]
    
    # stretch in y and z, we will tile it in x
    scale = (1, scale_y, scale_z)
    
    # scale the template to the mask size
    img = ndimage.zoom(img, scale)

    # repeat (tile) along first axis
    # np.tile only works on last axis, so we transpose the matrix first and back
    img = np.transpose(img, (2,1,0))
    img = np.tile(img, int(np.ceil(scale_x)))
    img = np.transpose(img, (2,1,0))
    
    # cut the excess from the tiling
    img = img[:target_img.shape[0],:,:]
    
    return img

def prepare_magnitude(template, vessel, template_mask, case_mask, threshold=0):
    # Get the mean values of template vessel
    vessel_1d = vessel[vessel > 0]
    meanVal = np.mean(vessel_1d)
    stdVal = np.std(vessel_1d)

    # Scale and repeat the template to fit the CFD size
    new_template = scale_and_repeat(template, case_mask)
    vessel = scale_and_repeat(vessel, case_mask)
    template_mask = scale_and_repeat(template_mask, case_mask)

    # Fill in the no signal region with the mean value of the vessel
    nosig_mask = template_mask < 1
    nosig = meanVal * nosig_mask

    # New magnitude image, with nosignal region assigned with mean value
    noisy_vessel = vessel + nosig

    # Cut out the CFD mask from the template
    new_template[case_mask > 0] = 0
    # Fill in the new vessel to the CFD mask
    new_vessel = case_mask * noisy_vessel
    
    # Add them up
    new_magnitude = new_template + new_vessel

    # Set values below the threshold to zero
    new_magnitude[new_magnitude < threshold] = 0

    return new_magnitude


# DUAL VENC RECONSTRUCTION AS IN VELOMAP
def flow_dualvenc_reconstruction(vel_lv, vel_hv, venc_l, venc_h):
    
    # Pre-allocate corrected velocity data (same shape as vel_lv)
    dataFlowDV = np.zeros_like(vel_lv, dtype='float32')

    # Compute difference between high and low velocity images
    diff = vel_hv - vel_lv

    # Thresholds for fold detection (phase wrapping)
    #fold1 = venc_l * 1.2  # 2
    #fold1plus = venc_l * 2.8
    #fold2 = venc_l * 3.2 # 4 ## 1.92
    #fold2plus = venc_l * 4.8  ## 2.88
    #fold3 = venc_l * 5.0 # 6 ## 3
    #fold3plus = venc_l * 7.0 ## 4.2
    ##fold4 = venc_l * 7.0 # Diff 8*venc_l (= 4)
    ##fold4plus = venc_l * 9.0

    fold1 = venc_l * 1.2
    fold1plus = venc_l * 3.0
    fold2 = venc_l * 3.0
    fold2plus = venc_l * 5.0
    fold3 = venc_l * 5.0
    fold3plus = venc_l * 7.0

    # Find aliased regions for 1-2 wraps
    idx_aliased_pos_fold1 = np.where((diff > fold1) & (diff < fold1plus))
    idx_aliased_neg_fold1 = np.where((diff < -fold1) & (diff > -fold1plus))

    #print(f"diff: {diff}")
    #print(f"idx_aliased_pos_fold1: {idx_aliased_pos_fold1}")

    # Find aliased regions for 3-4 wraps
    diff2 = diff.copy()
    diff2[idx_aliased_pos_fold1] = 0
    diff2[idx_aliased_neg_fold1] = 0

    idx_aliased_pos_fold2 = np.where((diff2 >= fold2) & (diff2 < fold2plus))
    idx_aliased_neg_fold2 = np.where((diff2 <= -fold2) & (diff2 > -fold2plus))

    #print(f"diff2: {diff2}")
    #print(f"idx_aliased_pos_fold2: {idx_aliased_pos_fold2}")

    # Find aliased regions for 5-6 wraps
    diff3 = diff.copy()
    diff3[idx_aliased_pos_fold1] = 0
    diff3[idx_aliased_neg_fold1] = 0
    diff3[idx_aliased_pos_fold2] = 0
    diff3[idx_aliased_neg_fold2] = 0

    idx_aliased_pos_fold3 = np.where((diff3 >= fold3) & (diff3 < fold3plus))
    idx_aliased_neg_fold3 = np.where((diff3 <= -fold3) & (diff3 > -fold3plus))
    
    #print(f"diff3: {diff3}")
    #print(f"idx_aliased_pos_fold3: {idx_aliased_pos_fold3}")

    # Find aliased regions for 7-8 wraps
    ## diff4 = diff3.copy()
    ## diff4[idx_aliased_pos_fold1] = 0
    ## diff4[idx_aliased_neg_fold1] = 0
    ## diff4[idx_aliased_pos_fold2] = 0
    ## diff4[idx_aliased_neg_fold2] = 0
    ## diff4[idx_aliased_pos_fold3] = 0
    ## diff4[idx_aliased_neg_fold3] = 0
    ## idx_aliased_pos_fold4 = np.where((diff4 > fold4) & (diff4 < fold4plus))
    ## idx_aliased_neg_fold4 = np.where((diff4 < -fold4) & (diff4 > -fold4plus))
    ## print(f"diff4: {diff4}")
    ## print(f"idx_aliased_pos_fold3: {idx_aliased_neg_fold4}")

    # Start with the low venc image
    dataFlowDV = vel_lv.copy()

    # Apply corrections for 1-2 wraps
    dataFlowDV[idx_aliased_pos_fold1] += 2 * venc_l
    dataFlowDV[idx_aliased_neg_fold1] -= 2 * venc_l

    # Apply corrections for 3-4 wraps
    dataFlowDV[idx_aliased_pos_fold2] += 4 * venc_l
    dataFlowDV[idx_aliased_neg_fold2] -= 4 * venc_l

    # Apply corrections for 5-6 wraps
    dataFlowDV[idx_aliased_pos_fold3] += 6 * venc_l
    dataFlowDV[idx_aliased_neg_fold3] -= 6 * venc_l

    # Apply corrections for 7-8 wraps
    #dataFlowDV[idx_aliased_pos_fold4] += 8 * venc_l
    #dataFlowDV[idx_aliased_neg_fold4] -= 8 * venc_l

    # Replace top/bottom slices with low venc values # why this?
    #bottomslice = 0
    #topslice = vel_lv.shape[2] - 1
    #dataFlowDV[:, :, bottomslice, :, :] = vel_lv[:, :, bottomslice, :, :]
    #dataFlowDV[:, :, topslice, :, :] = vel_lv[:, :, topslice, :, :]

    # Return the corrected data
    return dataFlowDV


if __name__ == '__main__':
    # Update your path here
    #base_path = '../../../data/michigan_cerebro_CFD/patient3-postOp' #r'D:\Dataset\pressure_data\no_fsaa_synth'
    base_path = "../../../data/data_05mm_incl_pressure"
    #output_dir = '../../../data/michigan_cerebro_CFD/patient3-postOp' #r'D:\Dataset\pressure_data\test_output'
    output_dir = '../../../data/data_05mm_incl_pressure' #r'D:\Dataset\pressure_data\test_output'
    template_filepath = '../../../data/mag_templates.h5'   #r'D:\Dataset\pressure_data\in-vivo\mag_template_aligned.h5'
    
    # Change your case name, template idx (see mag_template_aligned.h5), and target SNR here
    case_name = 'healthy-05mm3'
    targetSNR_list = [8, 10, 12]
    targetSNR_hv = 15
    targetSNR_hv = targetSNR_hv**2
    mag_threshold = 0 # 30
    template_idx = 0

    # Set random seed for reproducability
    seed_value = 432
    rnd.seed(432)
    np.random.seed(432)

    # Make output dir
    os.makedirs(output_dir, exist_ok=True)
    input_filepath  = f'{base_path}/{case_name}.h5'
    outputLR_filename = f'{output_dir}/{case_name}_LR_dv_241211.h5'
    
    #outputLR_filename = f'{output_dir}/{case_name}_LR_dv_TSNR6-test.h5'

    downsample = 2
    crop_ratio = 1 / downsample
    #-----------------------
    is_mask_saved = False 

    # Load the magnitude template
    # load template
    with h5py.File(template_filepath, 'r') as hf:
        template = np.asarray(hf.get('mag')[template_idx])
        #template = np.asarray(hf.get('mag_template')[template_idx])
        vessel = np.asarray(hf.get('vessels')[template_idx])
        #vessel = np.asarray(hf.get('vessel')[template_idx])
        template_mask = np.asarray(hf.get('mask')[template_idx])

    # Load the mask once
    with h5py.File(input_filepath, mode = 'r' ) as hf:
        dx = np.asarray(hf['dx'])
        data_count = len(hf.get("u"))
        case_mask = np.asarray(hf.get('mask'))
    
    # Create the synthetic magnitude based on template and case_mask
    print("Preparing magnitude from template...")
    mag_image = prepare_magnitude(template, vessel, template_mask, case_mask, threshold=mag_threshold)
    
    pad_x, pad_y, pad_z = calculate_pad(mag_image.shape, downsample)
    # print(padding)
    print('padding', pad_x, pad_y, pad_z)

    # Create the magnitude based on the possible values
    ## This is a part of augmentation to make sure we have varying magnitude    
    mag_image = pad(mag_image, pad_x, pad_y, pad_z)

    mag_image = apply_tukey_filter(mag_image, alpha=0.5)

    start_time = time.time()

    for idx in range(data_count):
        
        # Set targetSNR (temporally varying)
        targetSNR = rnd.choice(targetSNR_list)
        print(f"\nProcessing {idx+1}/{data_count} - SNR {targetSNR}")
        targetSNR = targetSNR**2

        # Load the velocity U V W from H5
        with h5py.File(input_filepath, mode = 'r' ) as hf:
            
            hr_u = np.asarray(hf['u'][idx])
            hr_v = np.asarray(hf['v'][idx])
            hr_w = np.asarray(hf['w'][idx])
            hr_p = np.asarray(hf['p'][idx])

            hr_u = pad(hr_u, pad_x, pad_y, pad_z)
            hr_v = pad(hr_v, pad_x, pad_y, pad_z) 
            hr_w = pad(hr_w, pad_x, pad_y, pad_z)
            hr_p = pad(hr_p, pad_x, pad_y, pad_z)

            max_u = np.asarray(hf['max_u'][idx])
            max_v = np.asarray(hf['max_v'][idx])
            max_w = np.asarray(hf['max_w'][idx])

        low_venc_values = np.asarray([0.5, 0.55, 0.6, 0.65, 0.7]) # in m/s
        
        # Select high, low, and max_vel
        low_venc = np.random.choice(low_venc_values)
        max_vel = np.max((max_u, max_v, max_w))

        # Set high_venc based on the ranges of max_vel
        if max_vel < 2 * low_venc:
            high_venc = 2 * low_venc
        else:
            high_venc = max_vel  
        
        print('high venc', high_venc)
        print('low venc', low_venc)
        print('max vel', max_vel)
            
        # DO the downsampling
        lr_u_lv, mag_u = fft.downsample_phase_img(hr_u, mag_image, low_venc, crop_ratio, targetSNR)
        lr_v_lv, mag_v = fft.downsample_phase_img(hr_v, mag_image, low_venc, crop_ratio, targetSNR)
        lr_w_lv, mag_w = fft.downsample_phase_img(hr_w, mag_image, low_venc, crop_ratio, targetSNR)

        lr_u_hv, _ = fft.downsample_phase_img(hr_u, mag_image, high_venc, crop_ratio, targetSNR_hv, hamming=True)
        lr_v_hv, _ = fft.downsample_phase_img(hr_v, mag_image, high_venc, crop_ratio, targetSNR_hv, hamming=True)
        lr_w_hv, _ = fft.downsample_phase_img(hr_w, mag_image, high_venc, crop_ratio, targetSNR_hv, hamming=True)

        lr_u_dv_reconstruct = flow_dualvenc_reconstruction(lr_u_lv, lr_u_hv, low_venc, high_venc)
        lr_v_dv_reconstruct = flow_dualvenc_reconstruction(lr_v_lv, lr_v_hv, low_venc, high_venc)
        lr_w_dv_reconstruct = flow_dualvenc_reconstruction(lr_w_lv, lr_w_hv, low_venc, high_venc)

        lr_u = unpad(lr_u_dv_reconstruct, pad_x//downsample, pad_y//downsample, pad_z//downsample)
        lr_v = unpad(lr_v_dv_reconstruct, pad_x//downsample, pad_y//downsample, pad_z//downsample)
        lr_w = unpad(lr_w_dv_reconstruct, pad_x//downsample, pad_y//downsample, pad_z//downsample)

        lr_u_lv = unpad(lr_u_lv, pad_x//downsample, pad_y//downsample, pad_z//downsample)
        lr_v_lv = unpad(lr_v_lv, pad_x//downsample, pad_y//downsample, pad_z//downsample)
        lr_w_lv = unpad(lr_w_lv, pad_x//downsample, pad_y//downsample, pad_z//downsample)

        lr_u_hv = unpad(lr_u_hv, pad_x//downsample, pad_y//downsample, pad_z//downsample)
        lr_v_hv = unpad(lr_v_hv, pad_x//downsample, pad_y//downsample, pad_z//downsample)
        lr_w_hv = unpad(lr_w_hv, pad_x//downsample, pad_y//downsample, pad_z//downsample)

        mag_u = unpad(mag_u, pad_x//downsample, pad_y//downsample, pad_z//downsample)
        mag_v = unpad(mag_v, pad_x//downsample, pad_y//downsample, pad_z//downsample)
        mag_w = unpad(mag_w, pad_x//downsample, pad_y//downsample, pad_z//downsample)

        # Pressure 
        p_lr = ndimage.zoom(hr_p, crop_ratio, order=3)
        p_lr = unpad(p_lr, pad_x // downsample, pad_y // downsample, pad_z // downsample)
        h5utils.save_to_h5(outputLR_filename, "p", p_lr)
        
        # Save the downsampled image
        h5utils.save_to_h5(outputLR_filename, "u", lr_u)
        h5utils.save_to_h5(outputLR_filename, "v", lr_v)
        h5utils.save_to_h5(outputLR_filename, "w", lr_w)

        h5utils.save_to_h5(outputLR_filename, "u_lv", lr_u_lv)
        h5utils.save_to_h5(outputLR_filename, "v_lv", lr_v_lv)
        h5utils.save_to_h5(outputLR_filename, "w_lv", lr_w_lv)

        h5utils.save_to_h5(outputLR_filename, "u_hv", lr_u_hv)
        h5utils.save_to_h5(outputLR_filename, "v_hv", lr_v_hv)
        h5utils.save_to_h5(outputLR_filename, "w_hv", lr_w_hv)

        h5utils.save_to_h5(outputLR_filename, "mag_u", mag_u)
        h5utils.save_to_h5(outputLR_filename, "mag_v", mag_v)
        h5utils.save_to_h5(outputLR_filename, "mag_w", mag_w)

        h5utils.save_to_h5(outputLR_filename, "high_venc", high_venc)
        h5utils.save_to_h5(outputLR_filename, "low_venc", low_venc)
        h5utils.save_to_h5(outputLR_filename, "max_vel", max_vel)
        
        h5utils.save_to_h5(outputLR_filename, "_targetSNR", np.sqrt(targetSNR))
        
        if idx == 0:
            # only save once

            mask_image = pad(case_mask, pad_x, pad_y, pad_z)
            mask_image = ndimage.zoom(mask_image, crop_ratio, order=0, grid_mode=True, mode='nearest')
            mask_image = unpad(mask_image, pad_x//downsample, pad_y//downsample, pad_z//downsample)
            mag_image_stored = unpad(mag_image, pad_x, pad_y, pad_z)
            h5utils.save_to_h5(outputLR_filename, "mag_image", mag_image_stored)
            h5utils.save_to_h5(outputLR_filename, "dx", dx*downsample)
            h5utils.save_to_h5(outputLR_filename, "template_idx", template_idx)
            h5utils.save_to_h5(outputLR_filename, 'mask', mask_image)

        print(f"Time taken {(time.time() - start_time):.1f} secs.")

    print(f"Done! \nSaved in {outputLR_filename}")