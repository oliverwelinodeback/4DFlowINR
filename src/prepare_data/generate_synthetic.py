"""
Generate synthetic 4D Flow MRI data from high-resolution CFD reference data.

Supported options
-----------------
- Single-VENC or dual-VENC downsampling
- Optional spatial downsampling by central k-space cropping
- Optional temporal downsampling by retaining every Nth cardiac frame
- Pressure/pressure-gradient output at either the velocity resolution, the
  original HR resolution, or omitted 
"""

from pathlib import Path
import time
import h5py
import numpy as np
import scipy.ndimage as ndimage
from skimage.transform import downscale_local_mean
import fft_downsampling as fft
import h5_utils as h5utils

# =============================================================================
# Configuration

# Input/output
BASE_PATH = Path("../../data/stenosis_70")
CASE_NAME = "ICAD21_05mm3_20ms"
INPUT_FILE = BASE_PATH / f"{CASE_NAME}.h5"
OUTPUT_FILE = BASE_PATH / f"{CASE_NAME}_LR_check.h5"

# Magnitude template
TEMPLATE_FILE = Path("../../data/mag_templates.h5")
TEMPLATE_IDX = 0  # Set explicitly for reproducibility. Use None for random selection.
MAG_THRESHOLD = 30

# Downsampling parameters
SPATIAL_DOWNSAMPLE = 2   # 1 = keep HR spatial resolution, 2 = 2x downsampling
TEMPORAL_DOWNSAMPLE = 2  # 1 = keep all frames, 2 = keep every second frame
TARGET_SNR = 10           # Set to None to disable k-space noise

# VENC mode
VENC_MODE = "single"      # "single" or "dual"

# Used when VENC_MODE == "single"
SINGLE_VENC = 2.6         # m/s

# Used when VENC_MODE == "dual"
DUAL_VENC_HIGH = 2.6      # m/s
DUAL_VENC_LOW = 1.3       # m/s

# Pressure handling:
#   "match_velocity"  - spatially downsample p, px, py, pz to velocity resolution
#   "full_resolution" - retain p, px, py, pz at the original HR spatial resolution
#   "omit"            - do not write p, px, py, pz
PRESSURE_MODE = "match_velocity"

RANDOM_SEED = 1234        # Set to None for non-deterministic noise/template selection
OVERWRITE_OUTPUT = False

# =============================================================================

# Utility functions
def validate_settings():

    if not isinstance(SPATIAL_DOWNSAMPLE, int) or SPATIAL_DOWNSAMPLE < 1:
        raise ValueError("SPATIAL_DOWNSAMPLE must be a positive integer")

    if not isinstance(TEMPORAL_DOWNSAMPLE, int) or TEMPORAL_DOWNSAMPLE < 1:
        raise ValueError("TEMPORAL_DOWNSAMPLE must be a positive integer")

    if VENC_MODE not in {"single", "dual"}:
        raise ValueError("VENC_MODE must be 'single' or 'dual'")

    if PRESSURE_MODE not in {"match_velocity", "full_resolution", "omit"}:
        raise ValueError(
            "PRESSURE_MODE must be 'match_velocity', 'full_resolution', or 'omit'"
        )

    if VENC_MODE == "single" and SINGLE_VENC <= 0:
        raise ValueError("SINGLE_VENC must be > 0.")

    if VENC_MODE == "dual":
        if DUAL_VENC_LOW <= 0 or DUAL_VENC_HIGH <= 0:
            raise ValueError("Dual-VENC values must be > 0")
        if DUAL_VENC_LOW >= DUAL_VENC_HIGH:
            raise ValueError("DUAL_VENC_LOW must be smaller than DUAL_VENC_HIGH")

    if TARGET_SNR is not None and TARGET_SNR <= 0:
        raise ValueError("TARGET_SNR must be > 0 or None.")

def calculate_pad(arr_shape, downsample_factor):
    """Return right-side padding that makes each dimension crop-compatible."""
    divisor = downsample_factor * 2

    pad_x = arr_shape[0] % divisor
    pad_y = arr_shape[1] % divisor
    pad_z = arr_shape[2] % divisor

    pad_x = 0 if pad_x == 0 else divisor - pad_x
    pad_y = 0 if pad_y == 0 else divisor - pad_y
    pad_z = 0 if pad_z == 0 else divisor - pad_z

    return pad_x, pad_y, pad_z


def pad(arr, pad_x, pad_y, pad_z):
    """Pad a 3D array with zeros on the positive side of each axis."""
    return np.pad(arr, ((0, pad_x), (0, pad_y), (0, pad_z)), mode="constant")


def unpad(arr, pad_x, pad_y, pad_z):
    """Remove right-side padding from a 3D array."""
    return arr[:-pad_x or None, :-pad_y or None, :-pad_z or None]


def scale_and_repeat(img, target_img):
    """Resize and tile a magnitude-template volume to the target spatial shape."""
    scale_x = target_img.shape[0] / img.shape[0]
    scale_y = target_img.shape[1] / img.shape[1]
    scale_z = target_img.shape[2] / img.shape[2]

    img = ndimage.zoom(img, (scale_x, scale_y, 1))
    img = np.tile(img, int(np.ceil(scale_z)))
    return img[:, :, : target_img.shape[2]]


def prepare_magnitude(template, vessel, template_mask, case_mask, threshold=0):
    """Construct an aligned synthetic magnitude image for the CFD geometry."""
    vessel_values = vessel[vessel > 0]
    if vessel_values.size == 0:
        raise ValueError("Magnitude-template vessel mask contains no positive voxels.")

    mean_vessel_signal = np.mean(vessel_values)

    new_template = scale_and_repeat(template, case_mask)
    vessel = scale_and_repeat(vessel, case_mask)
    template_mask = scale_and_repeat(template_mask, case_mask)

    no_signal_mask = template_mask < 1
    noisy_vessel = vessel + mean_vessel_signal * no_signal_mask

    new_template[case_mask > 0] = 0
    new_vessel = case_mask * noisy_vessel

    magnitude = new_template + new_vessel
    magnitude[magnitude < threshold] = 0

    return magnitude


def flow_dualvenc_reconstruction(vel_lv, vel_hv, venc_l, venc_h):
    """Reconstruct dual-VENC velocity"""
    del venc_h  # Kept in the signature for readability - thresholds depend on low VENC.

    diff = vel_hv - vel_lv

    fold1 = venc_l * 1.2
    fold1plus = venc_l * 3.0
    fold2 = venc_l * 3.0
    fold2plus = venc_l * 5.0
    fold3 = venc_l * 5.0
    fold3plus = venc_l * 7.0

    idx_pos_1 = np.where((diff > fold1) & (diff < fold1plus))
    idx_neg_1 = np.where((diff < -fold1) & (diff > -fold1plus))

    diff2 = diff.copy()
    diff2[idx_pos_1] = 0
    diff2[idx_neg_1] = 0

    idx_pos_2 = np.where((diff2 >= fold2) & (diff2 < fold2plus))
    idx_neg_2 = np.where((diff2 <= -fold2) & (diff2 > -fold2plus))

    diff3 = diff.copy()
    diff3[idx_pos_1] = 0
    diff3[idx_neg_1] = 0
    diff3[idx_pos_2] = 0
    diff3[idx_neg_2] = 0

    idx_pos_3 = np.where((diff3 >= fold3) & (diff3 < fold3plus))
    idx_neg_3 = np.where((diff3 <= -fold3) & (diff3 > -fold3plus))

    reconstructed = vel_lv.copy()
    reconstructed[idx_pos_1] += 2 * venc_l
    reconstructed[idx_neg_1] -= 2 * venc_l
    reconstructed[idx_pos_2] += 4 * venc_l
    reconstructed[idx_neg_2] -= 4 * venc_l
    reconstructed[idx_pos_3] += 6 * venc_l
    reconstructed[idx_neg_3] -= 6 * venc_l

    return reconstructed


def simulate_velocity(velocity_components, magnitude_image, crop_ratio, snr_power_ratio, pad_amounts):
    """Generate synthetic velocity components using single- or dual-VENC encoding."""
    output_pad = tuple(amount // SPATIAL_DOWNSAMPLE for amount in pad_amounts)

    if VENC_MODE == "single":
        simulated = []
        magnitude_lr = None

        for component in velocity_components:
            component_lr, component_mag = fft.downsample_phase_img(
                component,
                magnitude_image,
                SINGLE_VENC,
                crop_ratio,
                snr_power_ratio,
            )
            simulated.append(component_lr)
            if magnitude_lr is None:
                magnitude_lr = component_mag

    else:
        # Preserve the ordering of the legacy dual-VENC script:
        # all low-VENC components first, then all high-VENC components.
        low_venc = []
        magnitude_lr = None

        for component in velocity_components:
            component_lr, component_mag = fft.downsample_phase_img(
                component,
                magnitude_image,
                DUAL_VENC_LOW,
                crop_ratio,
                snr_power_ratio,
            )
            low_venc.append(component_lr)
            if magnitude_lr is None:
                magnitude_lr = component_mag

        high_venc = []
        for component in velocity_components:
            component_lr, _ = fft.downsample_phase_img(
                component,
                magnitude_image,
                DUAL_VENC_HIGH,
                crop_ratio,
                snr_power_ratio,
            )
            high_venc.append(component_lr)

        simulated = [
            flow_dualvenc_reconstruction(
                low_component,
                high_component,
                DUAL_VENC_LOW,
                DUAL_VENC_HIGH,
            )
            for low_component, high_component in zip(low_venc, high_venc)
        ]

    simulated = [unpad(component, *output_pad) for component in simulated]
    magnitude_lr = unpad(magnitude_lr, *output_pad)

    return simulated[0], simulated[1], simulated[2], magnitude_lr


def prepare_pressure_field(field, crop_ratio, pad_amounts):
    """Prepare a pressure/pressure-gradient field according to PRESSURE_MODE."""
    pad_x, pad_y, pad_z = pad_amounts

    if PRESSURE_MODE == "match_velocity" and SPATIAL_DOWNSAMPLE > 1:
        field = ndimage.zoom(field, crop_ratio, order=3)
        return unpad(
            field,
            pad_x // SPATIAL_DOWNSAMPLE,
            pad_y // SPATIAL_DOWNSAMPLE,
            pad_z // SPATIAL_DOWNSAMPLE,
        )

    # For factor 1, match_velocity and full_resolution are equivalent.
    return unpad(field, pad_x, pad_y, pad_z)


def prepare_mask(case_mask, pad_amounts):
    """Prepare the binary mask at the synthetic velocity resolution."""
    if SPATIAL_DOWNSAMPLE == 1:
        return case_mask.astype(np.uint8)

    pad_x, pad_y, pad_z = pad_amounts
    mask_padded = pad(case_mask, pad_x, pad_y, pad_z)

    mask_soft = downscale_local_mean(
        mask_padded,
        (SPATIAL_DOWNSAMPLE,) * 3,
    )
    mask_lr = (mask_soft >= 0.25).astype(np.uint8)

    return unpad(
        mask_lr,
        pad_x // SPATIAL_DOWNSAMPLE,
        pad_y // SPATIAL_DOWNSAMPLE,
        pad_z // SPATIAL_DOWNSAMPLE,
    )


def save_metadata(output_file, template_idx):
    """Store generation settings as HDF5 attributes."""
    with h5py.File(output_file, "a") as hf:
        hf.attrs["generator"] = "generate_synthetic.py"
        hf.attrs["case_name"] = CASE_NAME
        hf.attrs["venc_mode"] = VENC_MODE
        hf.attrs["spatial_downsample"] = SPATIAL_DOWNSAMPLE
        hf.attrs["temporal_downsample"] = TEMPORAL_DOWNSAMPLE
        hf.attrs["target_snr"] = -1 if TARGET_SNR is None else TARGET_SNR
        hf.attrs["pressure_mode"] = PRESSURE_MODE
        hf.attrs["template_idx"] = template_idx
        hf.attrs["random_seed"] = -1 if RANDOM_SEED is None else RANDOM_SEED

        if VENC_MODE == "single":
            hf.attrs["venc"] = SINGLE_VENC
        else:
            hf.attrs["high_venc"] = DUAL_VENC_HIGH
            hf.attrs["low_venc"] = DUAL_VENC_LOW


# Main generation routine
def main():
    validate_settings()

    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    if not TEMPLATE_FILE.exists():
        raise FileNotFoundError(f"Magnitude template not found: {TEMPLATE_FILE}")

    if OUTPUT_FILE.exists():
        if not OVERWRITE_OUTPUT:
            raise FileExistsError(
                f"Output file already exists: {OUTPUT_FILE}\n"
                "Set OVERWRITE_OUTPUT = True to replace it."
            )
        OUTPUT_FILE.unlink()

    template_idx = TEMPLATE_IDX
    if template_idx is None:
        template_idx = np.random.randint(0, 5)

    with h5py.File(TEMPLATE_FILE, "r") as hf:
        template = np.asarray(hf["mag"][template_idx])
        vessel = np.asarray(hf["vessels"][template_idx])
        template_mask = np.asarray(hf["mask"][template_idx])

    with h5py.File(INPUT_FILE, "r") as hf:
        required = {"u", "v", "w", "mask", "dx"}
        if PRESSURE_MODE != "omit":
            required.update({"p", "px", "py", "pz"})

        missing = sorted(required.difference(hf.keys()))
        if missing:
            raise KeyError(
                f"Input file is missing required datasets: {', '.join(missing)}"
            )

        dx = np.asarray(hf["dx"])
        data_count = len(hf["u"])
        case_mask = np.asarray(hf["mask"])

        if case_mask.ndim == 4:
            case_mask = case_mask[0]

        magnitude_hr = prepare_magnitude(
            template,
            vessel,
            template_mask,
            case_mask,
            threshold=MAG_THRESHOLD,
        )

        pad_amounts = calculate_pad(magnitude_hr.shape, SPATIAL_DOWNSAMPLE)
        magnitude_padded = pad(magnitude_hr, *pad_amounts)
        crop_ratio = 1.0 / SPATIAL_DOWNSAMPLE

        # Existing fft_downsampling.py expects the SNR power ratio
        snr_power_ratio = None if TARGET_SNR is None else TARGET_SNR ** 2

        frame_indices = range(0, data_count, TEMPORAL_DOWNSAMPLE)
        output_frame_count = len(range(0, data_count, TEMPORAL_DOWNSAMPLE))
        mask_out = prepare_mask(case_mask, pad_amounts)

        print("="*20)
        print("Synthetic 4D Flow MRI generation")
        print("="*20)
        print(f"Input:                {INPUT_FILE}")
        print(f"Output:               {OUTPUT_FILE}")
        print(f"VENC mode:            {VENC_MODE}")
        if VENC_MODE == "single":
            print(f"VENC:                 {SINGLE_VENC:.3g} m/s")
        else:
            print(f"Low / high VENC:      {DUAL_VENC_LOW:.3g} / {DUAL_VENC_HIGH:.3g} m/s")
        print(f"Spatial factor:       {SPATIAL_DOWNSAMPLE}")
        print(f"Temporal factor:      {TEMPORAL_DOWNSAMPLE}")
        print(f"Target SNR:           {TARGET_SNR}")
        print(f"Pressure mode:        {PRESSURE_MODE}")
        print(f"Template index:       {template_idx}")
        print(f"Random seed:          {RANDOM_SEED}")
        print(f"HR frames:            {data_count}")
        print(f"Output frames:        {output_frame_count}")
        print(f"Padding:              {pad_amounts}")
        print("="*20)

        if PRESSURE_MODE == "full_resolution" and SPATIAL_DOWNSAMPLE > 1:
            print(
                "WARNING: pressure fields will retain HR spatial dimensions "
                "while velocity fields are spatially downsampled."
            )

        # Save non-temporal quantities once.
        h5utils.save_to_h5(OUTPUT_FILE, "dx", dx * SPATIAL_DOWNSAMPLE)
        h5utils.save_to_h5(OUTPUT_FILE, "template_idx", template_idx)
        h5utils.save_to_h5(OUTPUT_FILE, "mask", mask_out)
        h5utils.save_to_h5(OUTPUT_FILE, "mag_image", magnitude_hr)

        if VENC_MODE == "single":
            h5utils.save_to_h5(OUTPUT_FILE, "venc", SINGLE_VENC)
        else:
            h5utils.save_to_h5(OUTPUT_FILE, "high_venc", DUAL_VENC_HIGH)
            h5utils.save_to_h5(OUTPUT_FILE, "low_venc", DUAL_VENC_LOW)

        start_time = time.time()

        for frame_number, idx in enumerate(frame_indices, start=1):
            print(
                f"Processing source frame {idx + 1}/{data_count} "
                f"(output frame {frame_number}/{output_frame_count})"
            )

            hr_u = pad(np.asarray(hf["u"][idx]), *pad_amounts)
            hr_v = pad(np.asarray(hf["v"][idx]), *pad_amounts)
            hr_w = pad(np.asarray(hf["w"][idx]), *pad_amounts)

            lr_u, lr_v, lr_w, magnitude_lr = simulate_velocity(
                (hr_u, hr_v, hr_w),
                magnitude_padded,
                crop_ratio,
                snr_power_ratio,
                pad_amounts,
            )

            h5utils.save_to_h5(OUTPUT_FILE, "u", lr_u)
            h5utils.save_to_h5(OUTPUT_FILE, "v", lr_v)
            h5utils.save_to_h5(OUTPUT_FILE, "w", lr_w)
            h5utils.save_to_h5(OUTPUT_FILE, "mag", magnitude_lr)

            if PRESSURE_MODE != "omit":
                for key in ("p", "px", "py", "pz"):
                    field = pad(np.asarray(hf[key][idx]), *pad_amounts)
                    field_out = prepare_pressure_field(
                        field,
                        crop_ratio,
                        pad_amounts,
                    )
                    h5utils.save_to_h5(OUTPUT_FILE, key, field_out)

        save_metadata(OUTPUT_FILE, template_idx)

    elapsed = time.time() - start_time

    print("="*20)
    print(f"Done in {elapsed:.1f} s")
    print(f"Saved: {OUTPUT_FILE}")
    print("="*20)


if __name__ == "__main__":
    main()


