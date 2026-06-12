"""
prepare_HRLR_sv.py
------------------
Creates HR-resolution LR files for all 9 cases (3 healthy, 3 stenosis_50, 3 stenosis_70).

"HR-resolution LR" means: same spatial grid as the CFD reference data, but with
realistic MRI noise added via k-space simulation (FFT noise model + magnitude template
+ VENC encoding). No spatial downsampling is performed (downsample=1, crop_ratio=1.0).

Purpose: test how well the PINN recovers pressure when trained on full-resolution
but noisy velocity measurements (no super-resolution component, pure denoising + physics).

Output filenames:  {case}_05mm3_20ms_HRLR_sv{venc_int}_tSNR10.h5
Output location:   same folder as the input HR file

Usage
-----
  cd /proj/multipress/users/x_sebjo/SRFlow/src/prepare_data
  python3 prepare_HRLR_sv.py
"""

import numpy as np
import os
import h5py
import time
import scipy.ndimage as ndimage

import fft_downsampling as fft
import h5_utils as h5utils


# ---------------------------------------------------------------------------
# Case definitions
# venc values from Config_MetaLearning_Reptile_DataDriven.py case_venc dict
# template_idx fixed per case for reproducibility (0-8, one per case)
# ---------------------------------------------------------------------------
CASES = [
    # healthy
    {"folder": "healthy",     "case": "HV01_05mm3_20ms",  "venc": 1.7, "template_idx": 0},
    {"folder": "healthy",     "case": "HV03_05mm3_20ms",  "venc": 1.3, "template_idx": 1},
    {"folder": "healthy",     "case": "HV06_05mm3_20ms",  "venc": 1.2, "template_idx": 2},
    # stenosis 50%
    {"folder": "stenosis_50", "case": "ICAD28_05mm3_20ms", "venc": 1.3, "template_idx": 3},
    {"folder": "stenosis_50", "case": "ICAD48_05mm3_20ms", "venc": 1.3, "template_idx": 4},
    {"folder": "stenosis_50", "case": "ICAD98_05mm3_20ms", "venc": 5.1, "template_idx": 0},
    # stenosis 70%
    {"folder": "stenosis_70", "case": "ICAD17_05mm3_20ms",  "venc": 4.1, "template_idx": 1},
    {"folder": "stenosis_70", "case": "ICAD21_05mm3_20ms",  "venc": 2.6, "template_idx": 2},
    {"folder": "stenosis_70", "case": "ICAD146_05mm3_20ms", "venc": 1.7, "template_idx": 3},
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DATA_DIR    = "../../data"
TEMPLATE_FILE    = "../../data/mag_templates.h5"
TARGET_SNR       = 2
DOWNSAMPLE       = 1        # No spatial downsampling — HR-resolution output
CROP_RATIO       = 1.0      # Full k-space kept
MAG_THRESHOLD    = 30


# ---------------------------------------------------------------------------
# Helpers (identical to prepare_LR_250919_sv.py)
# ---------------------------------------------------------------------------

def calculate_pad(arr_shape, upsample_rate):
    divisor = upsample_rate * 2
    pad_x = arr_shape[0] % divisor
    pad_y = arr_shape[1] % divisor
    pad_z = arr_shape[2] % divisor
    pad_x = (0 if pad_x == 0 else divisor - pad_x)
    pad_y = (0 if pad_y == 0 else divisor - pad_y)
    pad_z = (0 if pad_z == 0 else divisor - pad_z)
    return pad_x, pad_y, pad_z


def pad(u, x, y, z):
    return np.pad(u, ((0, x), (0, y), (0, z)), 'constant')


def unpad(u, x, y, z):
    return u[:-x or None, :-y or None, :-z or None]


def scale_and_repeat(img, target_img):
    scale_x = target_img.shape[0] / img.shape[0]
    scale_y = target_img.shape[1] / img.shape[1]
    scale_z = target_img.shape[2] / img.shape[2]
    scale = (scale_x, scale_y, 1)
    img = ndimage.zoom(img, scale)
    img = np.tile(img, int(np.ceil(scale_z)))
    img = img[:, :, :target_img.shape[2]]
    return img


def prepare_magnitude(template, vessel, template_mask, case_mask, threshold=0):
    vessel_1d = vessel[vessel > 0]
    meanVal = np.mean(vessel_1d)
    new_template = scale_and_repeat(template, case_mask)
    vessel      = scale_and_repeat(vessel, case_mask)
    template_mask = scale_and_repeat(template_mask, case_mask)
    nosig_mask  = template_mask < 1
    nosig       = meanVal * nosig_mask
    noisy_vessel = vessel + nosig
    new_template[case_mask > 0] = 0
    new_vessel  = case_mask * noisy_vessel
    new_magnitude = new_template + new_vessel
    new_magnitude[new_magnitude < threshold] = 0
    return new_magnitude


# ---------------------------------------------------------------------------
# Per-case processing
# ---------------------------------------------------------------------------

def process_case(case_def, templates):
    folder       = case_def["folder"]
    case_name    = case_def["case"]
    venc         = case_def["venc"]
    template_idx = case_def["template_idx"]

    # Derive integer venc for filename (e.g. 1.7 → "17")
    venc_int = str(venc).replace(".", "")

    input_path  = os.path.join(BASE_DATA_DIR, folder, f"{case_name}.h5")
    output_path = os.path.join(BASE_DATA_DIR, folder,
                               f"{case_name}_HRLR_sv{venc_int}_tSNR{TARGET_SNR}.h5")

    print(f"\n{'='*60}")
    print(f"  Case    : {case_name}")
    print(f"  venc    : {venc} m/s  (sv{venc_int})")
    print(f"  Template: {template_idx}")
    print(f"  Input   : {input_path}")
    print(f"  Output  : {output_path}")

    if not os.path.exists(input_path):
        print(f"  ERROR: input file not found — skipping")
        return False

    if os.path.exists(output_path):
        print(f"  WARNING: output already exists — overwriting")
        os.remove(output_path)

    # Load magnitude template
    template      = templates["mag"][template_idx]
    vessel        = templates["vessels"][template_idx]
    template_mask = templates["mask"][template_idx]

    # Load mask and dx
    with h5py.File(input_path, "r") as hf:
        dx         = np.asarray(hf["dx"])
        data_count = len(hf["u"])
        case_mask  = np.asarray(hf["mask"])

    if case_mask.ndim == 4:
        case_mask = case_mask[0]

    print(f"  Mask    : {case_mask.shape}  |  timesteps: {data_count}")

    # Build magnitude image and compute padding
    mag_image = prepare_magnitude(template, vessel, template_mask, case_mask,
                                  threshold=MAG_THRESHOLD)
    pad_x, pad_y, pad_z = calculate_pad(mag_image.shape, DOWNSAMPLE)
    mag_image = pad(mag_image, pad_x, pad_y, pad_z)

    targetSNR_var = TARGET_SNR ** 2  # convert tSNR to variance (matches original script)

    non_temporal_saved = False
    start_time = time.time()

    for idx in range(data_count):
        print(f"  Timestep {idx+1}/{data_count}  ({time.time()-start_time:.1f}s)")

        with h5py.File(input_path, "r") as hf:
            hr_u  = pad(np.asarray(hf["u"][idx]),  pad_x, pad_y, pad_z)
            hr_v  = pad(np.asarray(hf["v"][idx]),  pad_x, pad_y, pad_z)
            hr_w  = pad(np.asarray(hf["w"][idx]),  pad_x, pad_y, pad_z)
            hr_p  = pad(np.asarray(hf["p"][idx]),  pad_x, pad_y, pad_z)
            hr_px = pad(np.asarray(hf["px"][idx]), pad_x, pad_y, pad_z)
            hr_py = pad(np.asarray(hf["py"][idx]), pad_x, pad_y, pad_z)
            hr_pz = pad(np.asarray(hf["pz"][idx]), pad_x, pad_y, pad_z)

        # k-space noise simulation — no spatial downsampling (crop_ratio=1.0)
        lr_u, mag = fft.downsample_phase_img(hr_u, mag_image, venc, CROP_RATIO, targetSNR_var)
        lr_v, _   = fft.downsample_phase_img(hr_v, mag_image, venc, CROP_RATIO, targetSNR_var)
        lr_w, _   = fft.downsample_phase_img(hr_w, mag_image, venc, CROP_RATIO, targetSNR_var)

        # Unpad (pad amounts don't change since downsample=1)
        lr_u = unpad(lr_u, pad_x, pad_y, pad_z)
        lr_v = unpad(lr_v, pad_x, pad_y, pad_z)
        lr_w = unpad(lr_w, pad_x, pad_y, pad_z)
        mag  = unpad(mag,  pad_x, pad_y, pad_z)

        # Pressure: no zoom needed (crop_ratio=1.0, same spatial grid)
        p_out  = unpad(hr_p,  pad_x, pad_y, pad_z)
        px_out = unpad(hr_px, pad_x, pad_y, pad_z)
        py_out = unpad(hr_py, pad_x, pad_y, pad_z)
        pz_out = unpad(hr_pz, pad_x, pad_y, pad_z)

        h5utils.save_to_h5(output_path, "u", lr_u)
        h5utils.save_to_h5(output_path, "v", lr_v)
        h5utils.save_to_h5(output_path, "w", lr_w)
        h5utils.save_to_h5(output_path, "mag", mag)
        h5utils.save_to_h5(output_path, "p",  p_out)
        h5utils.save_to_h5(output_path, "px", px_out)
        h5utils.save_to_h5(output_path, "py", py_out)
        h5utils.save_to_h5(output_path, "pz", pz_out)

        if not non_temporal_saved:
            h5utils.save_to_h5(output_path, "mask",         case_mask)
            h5utils.save_to_h5(output_path, "dx",           dx)  # unchanged — same resolution
            h5utils.save_to_h5(output_path, "venc",         venc)
            h5utils.save_to_h5(output_path, "template_idx", template_idx)
            non_temporal_saved = True

    print(f"  Done — {time.time()-start_time:.1f}s  →  {output_path}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("prepare_HRLR_sv.py — HR-resolution LR generation for all cases")
    print(f"downsample={DOWNSAMPLE}, crop_ratio={CROP_RATIO}, tSNR={TARGET_SNR}")
    print(f"Template file: {TEMPLATE_FILE}")

    if not os.path.exists(TEMPLATE_FILE):
        raise FileNotFoundError(f"Magnitude template not found: {TEMPLATE_FILE}")

    # Load all templates once
    with h5py.File(TEMPLATE_FILE, "r") as hf:
        templates = {
            "mag":     np.asarray(hf["mag"]),
            "vessels": np.asarray(hf["vessels"]),
            "mask":    np.asarray(hf["mask"]),
        }
    print(f"Loaded {len(templates['mag'])} magnitude templates")

    errors = []
    for case_def in CASES:
        ok = process_case(case_def, templates)
        if not ok:
            errors.append(case_def["case"])

    print(f"\n{'='*60}")
    print(f"Done: {len(CASES)-len(errors)}/{len(CASES)} cases processed successfully.")
    if errors:
        print(f"Failed: {errors}")


if __name__ == "__main__":
    main()
