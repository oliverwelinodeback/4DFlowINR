import numpy as np
import os
import h5py
import random
import matplotlib.pyplot as plt
import time
import fft_downsampling as fft
import h5_utils as h5utils
import scipy.ndimage as ndimage
import pandas as pd
from scipy.signal import tukey
from scipy.fft import fftn, ifftn, fftshift, ifftshift

def calculate_pad(arr_shape, upsample_rate):
    """
        Calculate padding to ensure the size is halved when using k-space downsample
    """
    divisor = upsample_rate * 2 # double the downsample rate because of half k-space center
    pad_x = arr_shape[0] % divisor
    pad_y = arr_shape[1] % divisor
    pad_z = arr_shape[2] % divisor

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
    
    # stretch in x and y, we will tile it in z
    scale = (scale_x, scale_y, 1)

    # scale the template to the mask size
    img = ndimage.zoom(img, scale)

    # repeat (tile) along first axis
    img = np.tile(img, int(np.ceil(scale_z)))

    # cut the excess from the tiling
    img = img[:,:, :target_img.shape[2]]

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
    base_path = '../../data/healthy' 
    output_dir = '../../data/healthy'

    # Mag template
    template_filepath = '../../data/mag_templates.h5'
    template_idx = np.random.randint(0, 5)

    # Downsampling parameters
    targetSNR = 10
    targetSNR = targetSNR**2 # convert to variance    
    downsample = 2
    case_name = 'HV01_05mm3_20ms'
    venc = 0.4 # in m/s
    mag_threshold = 30

    # tSNR = 12 (high), 8 (med), 4 (low) for downsample = 2
    # tSNR = 8 (high), 4 (med), 2 (low) for downsample = 1

    # -----------------------
    input_filepath  =   f'{base_path}/{case_name}.h5'
    outputLR_filename = f'{base_path}/{case_name}_sv04_tSNR10.h5'
    
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

    if len(case_mask.shape) == 4:
        case_mask = case_mask[0]

    # Create the synthetic magnitude based on template and case_mask
    print("Preparing magnitude from template...")
    mag_image = prepare_magnitude(template, vessel, template_mask, case_mask, threshold=mag_threshold)
    pad_x, pad_y, pad_z = calculate_pad(mag_image.shape, downsample)
    mag_image = pad(mag_image, pad_x, pad_y, pad_z)

    non_temporal_params_saved = False

    start_time = time.time()

    #for idx in range(0, data_count):
    for idx in range(0, data_count, 2): # Temporal downsample
    #for idx in range(0, 4, 2): # testing

        print(f"\nProcessing {idx+1}/{data_count} - SNR {targetSNR}")
        
        # Load the velocity U V W from H5
        with h5py.File(input_filepath, mode = 'r' ) as hf:
            
            hr_u = np.asarray(hf['u'][idx])
            hr_v = np.asarray(hf['v'][idx])
            hr_w = np.asarray(hf['w'][idx])

            p = np.asarray(hf['p'][idx])

            px = np.asarray(hf['px'][idx])
            py = np.asarray(hf['py'][idx])
            pz = np.asarray(hf['pz'][idx])

            hr_u = pad(hr_u, pad_x, pad_y, pad_z)
            hr_v = pad(hr_v, pad_x, pad_y, pad_z) 
            hr_w = pad(hr_w, pad_x, pad_y, pad_z)
            hr_p = pad(p, pad_x, pad_y, pad_z)
            hr_px = pad(px, pad_x, pad_y, pad_z)
            hr_py = pad(py, pad_x, pad_y, pad_z)
            hr_pz = pad(pz, pad_x, pad_y, pad_z)

            #max_u = np.asarray(hf['max_u'][idx])
            #max_v = np.asarray(hf['max_v'][idx])
            #max_w = np.asarray(hf['max_w'][idx])
        
        print('venc', venc)

        # Execute downsampling
        lr_u, mag = fft.downsample_phase_img(hr_u, mag_image, venc, crop_ratio, targetSNR)
        lr_v, _ = fft.downsample_phase_img(hr_v, mag_image, venc, crop_ratio, targetSNR)
        lr_w, _ = fft.downsample_phase_img(hr_w, mag_image, venc, crop_ratio, targetSNR)

        lr_u = unpad(lr_u, pad_x//downsample, pad_y//downsample, pad_z//downsample)
        lr_v = unpad(lr_v, pad_x//downsample, pad_y//downsample, pad_z//downsample)
        lr_w = unpad(lr_w, pad_x//downsample, pad_y//downsample, pad_z//downsample)

        mag = unpad(mag, pad_x//downsample, pad_y//downsample, pad_z//downsample)

        # Save the downsampled image
        h5utils.save_to_h5(outputLR_filename, "u", lr_u)
        h5utils.save_to_h5(outputLR_filename, "v", lr_v)
        h5utils.save_to_h5(outputLR_filename, "w", lr_w)

        h5utils.save_to_h5(outputLR_filename, "mag_image", mag_image)

        h5utils.save_to_h5(outputLR_filename, "mag", mag)

        print(lr_u.shape)

        # Save pressure
        if crop_ratio != 1.0: 
            p_lr = ndimage.zoom(hr_p, crop_ratio, order=3)
            p_lr = unpad(p_lr, pad_x // downsample, pad_y // downsample, pad_z // downsample)
            h5utils.save_to_h5(outputLR_filename, "p", p_lr)
            print("p lr shape: ", p_lr.shape)

            px_lr = ndimage.zoom(hr_px, crop_ratio, order=3)
            px_lr = unpad(px_lr, pad_x // downsample, pad_y // downsample, pad_z // downsample)
            h5utils.save_to_h5(outputLR_filename, "px", px_lr)
            py_lr = ndimage.zoom(hr_py, crop_ratio, order=3)
            py_lr = unpad(py_lr, pad_x // downsample, pad_y // downsample, pad_z // downsample)
            h5utils.save_to_h5(outputLR_filename, "py", py_lr)          
            pz_lr = ndimage.zoom(hr_pz, crop_ratio, order=3)
            pz_lr = unpad(pz_lr, pad_x // downsample, pad_y // downsample, pad_z // downsample)
            h5utils.save_to_h5(outputLR_filename, "pz", pz_lr)
            print("pz lr shape:", pz_lr.shape)

        else:
            p = unpad(hr_p, pad_x // downsample, pad_y // downsample, pad_z // downsample)
            h5utils.save_to_h5(outputLR_filename, 'p', p)
            px = unpad(hr_px, pad_x // downsample, pad_y // downsample, pad_z // downsample)
            h5utils.save_to_h5(outputLR_filename, 'px', px)
            py = unpad(hr_py, pad_x // downsample, pad_y // downsample, pad_z // downsample)
            h5utils.save_to_h5(outputLR_filename, 'py', py)
            pz = unpad(hr_pz, pad_x // downsample, pad_y // downsample, pad_z // downsample)
            h5utils.save_to_h5(outputLR_filename, 'pz', pz)
        
        if not non_temporal_params_saved:
            
            if crop_ratio != 1.0: 
                mask_image = pad(case_mask, pad_x, pad_y, pad_z)
                mask_image = ndimage.zoom(mask_image, crop_ratio, order=0)
                mask_image = unpad(mask_image, pad_x//downsample, pad_y//downsample, pad_z//downsample)
            else:
                mask_image = case_mask

            print('Original mask: ', case_mask.shape)
            print('New downsampled mask: ', mask_image.shape)

            # only save once
            h5utils.save_to_h5(outputLR_filename, "dx", dx*downsample)
            h5utils.save_to_h5(outputLR_filename, "template_idx", template_idx) # magnitude template idx
            h5utils.save_to_h5(outputLR_filename, 'mask', mask_image)
            h5utils.save_to_h5(outputLR_filename, "venc", venc)

            non_temporal_params_saved = True

        print(f"Time taken {(time.time() - start_time):.1f} secs.")

    print(f"Done! \nSaved in {outputLR_filename}")