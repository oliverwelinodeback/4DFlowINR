# Imports
import numpy as np
from utils.preprocessing_utils import (
    standardize, compute_outer_boundary_mask,
    generate_collocation_points, generate_collocation_points_in_fluid_region,
    generate_boundary_points, min_max_normalize
)
import h5py

def load_data(config):
    
    # Load data
    with h5py.File(config["data_file"], mode='r') as hf:
        
        # Crop the data
        if config["setup"]["include_time"]:
            u = np.asarray(hf['u'][config["domain"]["t_start"]:config["domain"]["t_end"], 
                                config["domain"]["x_start"]:config["domain"]["x_end"], 
                                config["domain"]["y_start"]:config["domain"]["y_end"], 
                                config["domain"]["z_start"]:config["domain"]["z_end"]])
            v = np.asarray(hf['v'][config["domain"]["t_start"]:config["domain"]["t_end"], 
                                config["domain"]["x_start"]:config["domain"]["x_end"], 
                                config["domain"]["y_start"]:config["domain"]["y_end"], 
                                config["domain"]["z_start"]:config["domain"]["z_end"]])
            w = np.asarray(hf['w'][config["domain"]["t_start"]:config["domain"]["t_end"], 
                                config["domain"]["x_start"]:config["domain"]["x_end"], 
                                config["domain"]["y_start"]:config["domain"]["y_end"], 
                                config["domain"]["z_start"]:config["domain"]["z_end"]])
            
            p = np.asarray(hf['p'][config["domain"]["t_start"]:config["domain"]["t_end"], 
                                config["domain"]["x_start"]:config["domain"]["x_end"], 
                                config["domain"]["y_start"]:config["domain"]["y_end"], 
                                config["domain"]["z_start"]:config["domain"]["z_end"]]
                                ) if config["setup"]["include_pressure"] else None

            px = np.asarray(hf['px'][config["domain"]["t_start"]:config["domain"]["t_end"], 
                                config["domain"]["x_start"]:config["domain"]["x_end"], 
                                config["domain"]["y_start"]:config["domain"]["y_end"], 
                                config["domain"]["z_start"]:config["domain"]["z_end"]]
                                )*1000 if (config["setup"]["include_pressure"] and config["training"]["predict_gradients"]) else None

            py = np.asarray(hf['py'][config["domain"]["t_start"]:config["domain"]["t_end"], 
                                config["domain"]["x_start"]:config["domain"]["x_end"], 
                                config["domain"]["y_start"]:config["domain"]["y_end"], 
                                config["domain"]["z_start"]:config["domain"]["z_end"]]
                                )*1000 if (config["setup"]["include_pressure"] and config["training"]["predict_gradients"]) else None

            pz = np.asarray(hf['pz'][config["domain"]["t_start"]:config["domain"]["t_end"], 
                                config["domain"]["x_start"]:config["domain"]["x_end"], 
                                config["domain"]["y_start"]:config["domain"]["y_end"], 
                                config["domain"]["z_start"]:config["domain"]["z_end"]]
                                )*1000 if (config["setup"]["include_pressure"] and config["training"]["predict_gradients"]) else None

            #px *= 1000 # Pa /mm --> Pa /m
            #py *= 1000
            #pz *= 1000
        else:
            t_index = config["domain"]["t_start"]

            u = np.asarray(hf['u'][t_index, 
                                config["domain"]["x_start"]:config["domain"]["x_end"], 
                                config["domain"]["y_start"]:config["domain"]["y_end"], 
                                config["domain"]["z_start"]:config["domain"]["z_end"]])
            v = np.asarray(hf['v'][t_index, 
                                config["domain"]["x_start"]:config["domain"]["x_end"], 
                                config["domain"]["y_start"]:config["domain"]["y_end"], 
                                config["domain"]["z_start"]:config["domain"]["z_end"]])
            w = np.asarray(hf['w'][t_index, 
                                config["domain"]["x_start"]:config["domain"]["x_end"], 
                                config["domain"]["y_start"]:config["domain"]["y_end"], 
                                config["domain"]["z_start"]:config["domain"]["z_end"]])
            
            p = np.asarray(hf['p'][t_index,
                                config["domain"]["x_start"]:config["domain"]["x_end"], 
                                config["domain"]["y_start"]:config["domain"]["y_end"], 
                                config["domain"]["z_start"]:config["domain"]["z_end"]]
                                ) if config["setup"]["include_pressure"] else None
            
            px = np.asarray(hf['px'][config["domain"]["t_start"]:config["domain"]["t_end"], 
                                config["domain"]["x_start"]:config["domain"]["x_end"], 
                                config["domain"]["y_start"]:config["domain"]["y_end"], 
                                config["domain"]["z_start"]:config["domain"]["z_end"]]
                                )*1000 if (config["setup"]["include_pressure"] and config["training"]["predict_gradients"]) else None

            py = np.asarray(hf['py'][config["domain"]["t_start"]:config["domain"]["t_end"], 
                                config["domain"]["x_start"]:config["domain"]["x_end"], 
                                config["domain"]["y_start"]:config["domain"]["y_end"], 
                                config["domain"]["z_start"]:config["domain"]["z_end"]]
                                )*1000 if (config["setup"]["include_pressure"] and config["training"]["predict_gradients"]) else None

            pz = np.asarray(hf['pz'][config["domain"]["t_start"]:config["domain"]["t_end"], 
                                config["domain"]["x_start"]:config["domain"]["x_end"], 
                                config["domain"]["y_start"]:config["domain"]["y_end"], 
                                config["domain"]["z_start"]:config["domain"]["z_end"]]
                                )*1000 if (config["setup"]["include_pressure"] and config["training"]["predict_gradients"]) else None

            ## TODO - fix pressure gradients loading
            
        # T×h×w×d = (126, 81, 57, 50)

        mask = np.asarray(hf['mask'])
        if len(mask.shape) == 4: 
            mask = mask[0]

        mask = mask[config["domain"]["x_start"]:config["domain"]["x_end"], 
                    config["domain"]["y_start"]:config["domain"]["y_end"], 
                    config["domain"]["z_start"]:config["domain"]["z_end"]]
        # h×w×d = (81, 57, 50)

        if config["resolution"]["from_file"]:
            config["resolution"]["dx"] = hf.attrs['spacing'][0]
            config["resolution"]["dy"] = hf.attrs['spacing'][1]
            config["resolution"]["dz"] = hf.attrs['spacing'][2]
            config["resolution"]["dt"] = hf.attrs['dt']
            print(f"Loaded resolution from file: {config['resolution']['dx']}, {config['resolution']['dy']}, {config['resolution']['dz']}, {config['resolution']['dt']}")

    return u, v, w, p, px, py, pz, mask, config

def load_ref_data(config):
    
    # Load data
    with h5py.File(config["data_file_ref"], mode='r') as hf:
        
        # Crop the data
        if config["setup"]["include_time"]:
            u = np.asarray(hf['u'][config["domain"]["t_start"]*config["ref_temporal_factor"]:config["domain"]["t_end"]*config["ref_temporal_factor"],
                                config["domain"]["x_start"]*config["ref_spatial_factor"]:config["domain"]["x_end"]*config["ref_spatial_factor"], 
                                config["domain"]["y_start"]*config["ref_spatial_factor"]:config["domain"]["y_end"]*config["ref_spatial_factor"], 
                                config["domain"]["z_start"]*config["ref_spatial_factor"]:config["domain"]["z_end"]*config["ref_spatial_factor"]])
            v = np.asarray(hf['v'][config["domain"]["t_start"]*config["ref_temporal_factor"]:config["domain"]["t_end"]*config["ref_temporal_factor"],
                                config["domain"]["x_start"]*config["ref_spatial_factor"]:config["domain"]["x_end"]*config["ref_spatial_factor"], 
                                config["domain"]["y_start"]*config["ref_spatial_factor"]:config["domain"]["y_end"]*config["ref_spatial_factor"], 
                                config["domain"]["z_start"]*config["ref_spatial_factor"]:config["domain"]["z_end"]*config["ref_spatial_factor"]])
            w = np.asarray(hf['w'][config["domain"]["t_start"]*config["ref_temporal_factor"]:config["domain"]["t_end"]*config["ref_temporal_factor"],
                                config["domain"]["x_start"]*config["ref_spatial_factor"]:config["domain"]["x_end"]*config["ref_spatial_factor"], 
                                config["domain"]["y_start"]*config["ref_spatial_factor"]:config["domain"]["y_end"]*config["ref_spatial_factor"], 
                                config["domain"]["z_start"]*config["ref_spatial_factor"]:config["domain"]["z_end"]*config["ref_spatial_factor"]])
            
            p = np.asarray(hf['p'][config["domain"]["t_start"]*config["ref_temporal_factor"]:config["domain"]["t_end"]*config["ref_temporal_factor"], 
                                config["domain"]["x_start"]*config["ref_spatial_factor"]:config["domain"]["x_end"]*config["ref_spatial_factor"], 
                                config["domain"]["y_start"]*config["ref_spatial_factor"]:config["domain"]["y_end"]*config["ref_spatial_factor"], 
                                config["domain"]["z_start"]*config["ref_spatial_factor"]:config["domain"]["z_end"]*config["ref_spatial_factor"]]
                                ) if config["setup"]["include_pressure"] else None

            px = np.asarray(hf['px'][config["domain"]["t_start"]*config["ref_temporal_factor"]:config["domain"]["t_end"]*config["ref_temporal_factor"], 
                config["domain"]["x_start"]*config["ref_spatial_factor"]:config["domain"]["x_end"]*config["ref_spatial_factor"], 
                config["domain"]["y_start"]*config["ref_spatial_factor"]:config["domain"]["y_end"]*config["ref_spatial_factor"], 
                config["domain"]["z_start"]*config["ref_spatial_factor"]:config["domain"]["z_end"]*config["ref_spatial_factor"]]
                )*1000 if (config["setup"]["include_pressure"] and config["training"]["reference_gradients"]) else None
            py = np.asarray(hf['py'][config["domain"]["t_start"]*config["ref_temporal_factor"]:config["domain"]["t_end"]*config["ref_temporal_factor"],
                config["domain"]["x_start"]*config["ref_spatial_factor"]:config["domain"]["x_end"]*config["ref_spatial_factor"], 
                config["domain"]["y_start"]*config["ref_spatial_factor"]:config["domain"]["y_end"]*config["ref_spatial_factor"], 
                config["domain"]["z_start"]*config["ref_spatial_factor"]:config["domain"]["z_end"]*config["ref_spatial_factor"]]
                )*1000 if (config["setup"]["include_pressure"] and config["training"]["reference_gradients"]) else None
            pz = np.asarray(hf['pz'][config["domain"]["t_start"]*config["ref_temporal_factor"]:config["domain"]["t_end"]*config["ref_temporal_factor"],
                config["domain"]["x_start"]*config["ref_spatial_factor"]:config["domain"]["x_end"]*config["ref_spatial_factor"], 
                config["domain"]["y_start"]*config["ref_spatial_factor"]:config["domain"]["y_end"]*config["ref_spatial_factor"], 
                config["domain"]["z_start"]*config["ref_spatial_factor"]:config["domain"]["z_end"]*config["ref_spatial_factor"]]
                )*1000 if (config["setup"]["include_pressure"] and config["training"]["reference_gradients"]) else None
            
            #px *= 1000 # Pa /mm --> Pa /m
            #py *= 1000
            #pz *= 1000

        else:
            t_index = config["domain"]["t_start"]

            u = np.asarray(hf['u'][t_index, 
                                config["domain"]["x_start"]*config["ref_spatial_factor"]:config["domain"]["x_end"]*config["ref_spatial_factor"], 
                                config["domain"]["y_start"]*config["ref_spatial_factor"]:config["domain"]["y_end"]*config["ref_spatial_factor"], 
                                config["domain"]["z_start"]*config["ref_spatial_factor"]:config["domain"]["z_end"]*config["ref_spatial_factor"]])
            v = np.asarray(hf['v'][t_index, 
                                config["domain"]["x_start"]*config["ref_spatial_factor"]:config["domain"]["x_end"]*config["ref_spatial_factor"], 
                                config["domain"]["y_start"]*config["ref_spatial_factor"]:config["domain"]["y_end"]*config["ref_spatial_factor"], 
                                config["domain"]["z_start"]*config["ref_spatial_factor"]:config["domain"]["z_end"]*config["ref_spatial_factor"]])
            w = np.asarray(hf['w'][t_index, 
                                config["domain"]["x_start"]*config["ref_spatial_factor"]:config["domain"]["x_end"]*config["ref_spatial_factor"], 
                                config["domain"]["y_start"]*config["ref_spatial_factor"]:config["domain"]["y_end"]*config["ref_spatial_factor"], 
                                config["domain"]["z_start"]*config["ref_spatial_factor"]:config["domain"]["z_end"]*config["ref_spatial_factor"]])
            
            p = np.asarray(hf['p'][t_index,
                                config["domain"]["x_start"]*config["ref_spatial_factor"]:config["domain"]["x_end"]*config["ref_spatial_factor"], 
                                config["domain"]["y_start"]*config["ref_spatial_factor"]:config["domain"]["y_end"]*config["ref_spatial_factor"], 
                                config["domain"]["z_start"]*config["ref_spatial_factor"]:config["domain"]["z_end"]*config["ref_spatial_factor"]]
                                ) if config["setup"]["include_pressure"] else None
            
            px = np.asarray(hf['px'][config["domain"]["t_start"]*config["ref_temporal_factor"]:config["domain"]["t_end"]*config["ref_temporal_factor"], 
                config["domain"]["x_start"]*config["ref_spatial_factor"]:config["domain"]["x_end"]*config["ref_spatial_factor"], 
                config["domain"]["y_start"]*config["ref_spatial_factor"]:config["domain"]["y_end"]*config["ref_spatial_factor"], 
                config["domain"]["z_start"]*config["ref_spatial_factor"]:config["domain"]["z_end"]*config["ref_spatial_factor"]]
                )*1000 if (config["setup"]["include_pressure"] and config["training"]["reference_gradients"]) else None
            py = np.asarray(hf['py'][config["domain"]["t_start"]*config["ref_temporal_factor"]:config["domain"]["t_end"]*config["ref_temporal_factor"],
                config["domain"]["x_start"]*config["ref_spatial_factor"]:config["domain"]["x_end"]*config["ref_spatial_factor"], 
                config["domain"]["y_start"]*config["ref_spatial_factor"]:config["domain"]["y_end"]*config["ref_spatial_factor"], 
                config["domain"]["z_start"]*config["ref_spatial_factor"]:config["domain"]["z_end"]*config["ref_spatial_factor"]]
                )*1000 if (config["setup"]["include_pressure"] and config["training"]["reference_gradients"]) else None
            pz = np.asarray(hf['pz'][config["domain"]["t_start"]*config["ref_temporal_factor"]:config["domain"]["t_end"]*config["ref_temporal_factor"],
                config["domain"]["x_start"]*config["ref_spatial_factor"]:config["domain"]["x_end"]*config["ref_spatial_factor"], 
                config["domain"]["y_start"]*config["ref_spatial_factor"]:config["domain"]["y_end"]*config["ref_spatial_factor"], 
                config["domain"]["z_start"]*config["ref_spatial_factor"]:config["domain"]["z_end"]*config["ref_spatial_factor"]]
                )*1000 if (config["setup"]["include_pressure"] and config["training"]["reference_gradients"]) else None

        mask = np.asarray(hf['mask'])
        if len(mask.shape) == 4: 
            mask = mask[0]

        mask = mask[config["domain"]["x_start"]*config["ref_spatial_factor"]:config["domain"]["x_end"]*config["ref_spatial_factor"], 
                    config["domain"]["y_start"]*config["ref_spatial_factor"]:config["domain"]["y_end"]*config["ref_spatial_factor"], 
                    config["domain"]["z_start"]*config["ref_spatial_factor"]:config["domain"]["z_end"]*config["ref_spatial_factor"]]

    return u, v, w, p, px, py, pz, mask

def compute_template_parameters(config): 
    """Compute the template parameters for baseline normalization."""
    
    print('Computing template parameters for baseline normalization...')
    
    # Extract resolutions
    dx, dy, dz, dt = (
        config["template"]["dx"], config["template"]["dy"], config["template"]["dz"], config["template"]["dt"]
    )
    x_len, y_len, z_len, t_len = (
        config["template"]["x_len"], config["template"]["y_len"],
        config["template"]["z_len"], config["template"]["t_len"]
    )

    # Create linspaces
    t = np.linspace(dt, t_len * dt, t_len)
    x = np.linspace(dx, x_len * dx, x_len)
    y = np.linspace(dy, y_len * dy, y_len)
    z = np.linspace(dz, z_len * dz, z_len)

    # Normalize coordinates
    if config["coords_characteristic"]:
        L, T = config["constants"]["L"], config["constants"]["T"]
        t = t / T
        x = x / L
        y = y / L
        z = z / L

    tf = {}  # template_factors dict to return

    if config["coords_normalization"] == "standardize":
        # per-axis factors
        _, tf['mean_x'], tf['std_x'] = standardize(x)
        _, tf['mean_y'], tf['std_y'] = standardize(y)
        _, tf['mean_z'], tf['std_z'] = standardize(z)
        if config["setup"]["include_time"]:
            _, tf['mean_t'], tf['std_t'] = standardize(t)

        # optional global factors (replicates your runtime "largest span" policy)
        if config["global_normalization"]:
            # choose reference axis by largest span (using template spans)
            ranges = [np.ptp(arr) for arr in (x, y, z)]
            idx_largest = np.argmax(ranges)

            if idx_largest == 0:
                tf['global_mean'] = tf['mean_x']
                tf['global_std']  = tf['std_x']
            elif idx_largest == 1:
                tf['global_mean'] = tf['mean_y']
                tf['global_std']  = tf['std_y']
            else:
                tf['global_mean'] = tf['mean_z']
                tf['global_std']  = tf['std_z']

    elif config["coords_normalization"] == "min_max":
        # per-axis bounds
        _, tf['min_x'], tf['max_x'] = min_max_normalize(x)
        _, tf['min_y'], tf['max_y'] = min_max_normalize(y)
        _, tf['min_z'], tf['max_z'] = min_max_normalize(z)
        if config["setup"]["include_time"]:
            _, tf['min_t'], tf['max_t'] = min_max_normalize(t)

        # optional global bounds (shared across x,y,z)
        if config["global_normalization"]:
            tf['global_min'] = min(tf['min_x'], tf['min_y'], tf['min_z'])
            tf['global_max'] = max(tf['max_x'], tf['max_y'], tf['max_z'])

    else:
        raise ValueError("Unknown coords_normalization in config.")
    # Print the computed factors
    print("Template factors:")
    for key, value in tf.items():
        print(f"{key}: {value}")
    return tf

def create_and_normalize_coords(config, t_len, x_len, y_len, z_len):
    
    # Extract resolutions
    dx, dy, dz = config["resolution"]["dx"], config["resolution"]["dy"], config["resolution"]["dz"]
    dt = config["resolution"]["dt"]

    # Create linspaces
    t = np.linspace(dt, t_len * dt, t_len)
    x = np.linspace(dx, x_len * dx, x_len) # (h,) = (81, ) , [0.0005 0.001 ... 0.0405] (voxel centers)
    y = np.linspace(dy, y_len * dy, y_len)
    z = np.linspace(dz, z_len * dz, z_len)

    # FOV starts at 0.00025 or dx/2
    # FOV ends at 0.04075 = x_len * dx + dx/2

    # Normalize coordinates
    if config["coords_characteristic"]:
        L, T = config["constants"]["L"], config["constants"]["T"]
        t = t / T
        x = x / L
        y = y / L
        z = z / L

    t_normalized = None
    standardization_factors = None

    template_factors = None
    if config["use_baseline_normalization"]:
        template_factors = compute_template_parameters(config)
    
    if config["coords_normalization"] == "standardize":

        if config["global_normalization"]:
            if config["use_baseline_normalization"]:
                global_mean = template_factors['global_mean']
                global_std = template_factors['global_std']

                # Standardize all coordinate arrays using the global factors:
                x_normalized, mean_x, std_x = standardize(x, global_mean, global_std)
                y_normalized, mean_y, std_y = standardize(y, global_mean, global_std)
                z_normalized, mean_z, std_z = standardize(z, global_mean, global_std)
            else:
                ranges = [np.ptp(arr) for arr in (x, y, z)]
                print(ranges)
                idx_largest = np.argmax(ranges)

                if idx_largest == 0:
                    ref_data = x
                elif idx_largest == 1:
                    ref_data = y
                else:
                    ref_data = z

                # Compute global mean and std from the largest array:
                global_mean = np.mean(ref_data)
                global_std = np.std(ref_data)

                # Standardize all coordinate arrays using the global factors:
                x_normalized, mean_x, std_x = standardize(x, global_mean, global_std)
                y_normalized, mean_y, std_y = standardize(y, global_mean, global_std)
                z_normalized, mean_z, std_z = standardize(z, global_mean, global_std)

        else:
            if config["use_baseline_normalization"]:
                # Standardize all coordinate arrays using the global factors:
                x_normalized, mean_x, std_x = standardize(x, template_factors['mean_x'], template_factors['std_x'])
                y_normalized, mean_y, std_y = standardize(y, template_factors['mean_y'], template_factors['std_y'])
                z_normalized, mean_z, std_z = standardize(z, template_factors['mean_z'], template_factors['std_z'])
            else:
                x_normalized, mean_x, std_x = standardize(x)
                y_normalized, mean_y, std_y = standardize(y)
                z_normalized, mean_z, std_z = standardize(z)

        if config["setup"]["include_time"]:
            if config["use_baseline_normalization"]:
                t_normalized, mean_t, std_t = standardize(t, template_factors['mean_t'], template_factors['std_t'])
            else:
                t_normalized, mean_t, std_t = standardize(t)
            standardization_factors = [
                mean_t, std_t, mean_x, std_x, 
                mean_y, std_y, mean_z, std_z, 
            ]
        else:
            standardization_factors = [
                mean_x, std_x,
                mean_y, std_y,
                mean_z, std_z
            ]

    elif config["coords_normalization"] == "min_max":

        if config["global_normalization"]:
            if config["use_baseline_normalization"]:
                max_C = template_factors['global_max']
                min_C = template_factors['global_min']
                
            else:
                max_x, min_x = x.max(), x.min()
                max_y, min_y = y.max(), y.min()
                max_z, min_z = z.max(), z.min()
                
                max_C = max(max_x, max_y, max_z)
                min_C = min(min_x, min_y, min_z)

            x_normalized, min_x, max_x = min_max_normalize(x, min_C, max_C)
            y_normalized, min_y, max_y = min_max_normalize(y, min_C, max_C)
            z_normalized, min_z, max_z = min_max_normalize(z, min_C, max_C)

        else:
            if config["use_baseline_normalization"]:
                x_normalized, min_x, max_x = min_max_normalize(x, template_factors['min_x'], template_factors['max_x'])
                y_normalized, min_y, max_y = min_max_normalize(y, template_factors['min_y'], template_factors['max_y'])
                z_normalized, min_z, max_z = min_max_normalize(z, template_factors['min_z'], template_factors['max_z'])
            else:
                x_normalized, min_x, max_x = min_max_normalize(x)
                y_normalized, min_y, max_y = min_max_normalize(y)
                z_normalized, min_z, max_z = min_max_normalize(z)

        if config["setup"]["include_time"]:
            if config["use_baseline_normalization"]:
                t_normalized, min_t, max_t = min_max_normalize(t, template_factors['min_t'], template_factors['max_t'])
            else:
                t_normalized, min_t, max_t = min_max_normalize(t)
            standardization_factors = [
                min_t, max_t, min_x, max_x, 
                min_y, max_y, min_z, max_z, 
            ]
        else:
            standardization_factors = [
                min_x, max_x,
                min_y, max_y,
                min_z, max_z
            ]
    else:
        raise ValueError("Unknown coordinate normalization.")

    ## print(x_normalized)
    ## print(upsample_1d(x_normalized))
    ## print(upsample_1d(x_normalized, mode='centered'))

    return t_normalized, x_normalized, y_normalized, z_normalized, standardization_factors

def upsample_1d(arr, factor=2, mode='extend'):
    """
    mode : str
        'extend' or 'voxel_centered'.

        - 'extend': Anchors at arr[0], extends the domain on the right 
                    by + one new step.
        - 'centered': Interprets arr as voxel centers, extends 
                    domain edges on both sides by half an old voxel,
                    then subdivides.
    """
    if factor <= 1 or len(arr) <= 1:
        return arr

    dx = arr[1] - arr[0]

    if mode == 'extend':
        N_new = len(arr) * factor-1
        step = dx / factor
        start = arr[0]
        stop  = start + (N_new - 1) * step
        return np.linspace(start, stop, N_new)

    elif mode == 'centered':
        # old approach B
        dx_hr = dx / factor
        start = arr[0] - dx/2 + dx_hr/2
        end   = arr[-1] + dx/2 - dx_hr/2
        N_new = len(arr) * factor
        return np.linspace(start, end, N_new)

    else:
        raise ValueError("Unknown mode. Use 'extend' or 'centered'.")

def prepare_data(config, u, v, w, p, px, py, pz, mask):

    # Prepare coordinates
    if config["setup"]["include_time"]:
        t_len, x_len, y_len, z_len = u.shape # (T, h, w, d)
    else:
        x_len, y_len, z_len = u.shape # (h, w, d)
        t_len = 1

    t_normalized, x_normalized, y_normalized, z_normalized, standardization_factors = create_and_normalize_coords(config, t_len, x_len, y_len, z_len)

    # Create coordinate grid
    if config["setup"]["include_time"]:
        grids = np.meshgrid(t_normalized, x_normalized, y_normalized, z_normalized, indexing='ij')
    else:
        grids = np.meshgrid(x_normalized, y_normalized, z_normalized, indexing='ij')

# print(grids[0], last)
###  [[ 1.71835849  1.71835849  1.71835849 ...  1.71835849  1.71835849
###     1.71835849]
###   [ 1.71835849  1.71835849  1.71835849 ...  1.71835849  1.71835849
###     1.71835849]
###   ...
###   [ 1.71835849  1.71835849  1.71835849 ...  1.71835849  1.71835849
###     1.71835849]
###   [ 1.71835849  1.71835849  1.71835849 ...  1.71835849  1.71835849
###     1.71835849]]]]

# print(grids[3], last)
###  [[-1.69774938 -1.62845348 -1.55915759 ...  1.55915759  1.62845348
###     1.69774938]
###   [-1.69774938 -1.62845348 -1.55915759 ...  1.55915759  1.62845348
###     1.69774938]
###   ...
###   [-1.69774938 -1.62845348 -1.55915759 ...  1.55915759  1.62845348
###     1.69774938]
###   [-1.69774938 -1.62845348 -1.55915759 ...  1.55915759  1.62845348
###     1.69774938]]]]

    flat_coordinates = [grid.ravel() for grid in grids] # T×h×w×d --> T*h*w*d = (29087100,)
    xyz_data = np.stack(flat_coordinates, axis=1) # (T*h*w*d, 4) = (29087100, 4)

    # xyz_data:
### [[-1.71835849 -1.71079785 -1.70192589 -1.69774938]
###  [-1.71835849 -1.71079785 -1.70192589 -1.62845348]
###  ...
###  [ 1.71835849  1.71079785  1.70192589  1.62845348]
###  [ 1.71835849  1.71079785  1.70192589  1.69774938]]

    # Extract boundaries
    boundary_mask = compute_outer_boundary_mask(mask) # h×w×d = (81, 57, 50)

    if config["setup"]["include_time"]:
        # Tile the masks
        mask_flat = np.tile(mask.ravel(), t_len)
        boundary_mask_flat = np.tile(boundary_mask.ravel(), t_len)
    else:
        mask_flat = mask.ravel()
        boundary_mask_flat = boundary_mask.ravel()

    U_max = max(u.max(), v.max(), w.max())

    # Normalize velocity data
    if config["vel_normalization"] == "characteristic":
        U = config["constants"]["U"]
        u_normalized = u / U
        v_normalized = v / U
        w_normalized = w / U

    elif config["vel_normalization"] == "max_velocity":

        u_normalized = u / U_max
        v_normalized = v / U_max
        w_normalized = w / U_max

    # Flatten data into pointwise prediction
    u_flat = u_normalized.ravel()   # T×h×w×d --> T*h*w*d = (29087100,)
    v_flat = v_normalized.ravel()
    w_flat = w_normalized.ravel()

    velocities = [u_flat, v_flat, w_flat]

    if (config["setup"]["include_pressure"] and config["training"]["predict_gradients"]):
        rho, U, L = config["constants"]["rho"], config["constants"]["U"], config["constants"]["L"]
        p_normalized = p / (rho*(U**2))
        p_flat = p_normalized.reshape(-1)
        velocities.append(p_flat)

        _, _, _, std_x, _, std_y, _, std_z = standardization_factors

        px_normalized = px * L * std_x / (rho*(U**2))
        py_normalized = py * L * std_y / (rho*(U**2))
        pz_normalized = pz * L * std_z / (rho*(U**2))

        px_flat = px_normalized.reshape(-1)
        py_flat = py_normalized.reshape(-1)
        pz_flat = pz_normalized.reshape(-1)
        velocities.append(px_flat)
        velocities.append(py_flat)
        velocities.append(pz_flat)

    elif config["setup"]["include_pressure"]:
        rho, U = config["constants"]["rho"], config["constants"]["U"]
        p_normalized = p / (rho*(U**2))
        p_flat = p_normalized.reshape(-1)
        velocities.append(p_flat)

    # Ground truth data
    uvw_data = np.stack(velocities, axis=1) # (T*h*w*d, 4) = (29087100, 4)

    return uvw_data, xyz_data, mask_flat, boundary_mask_flat, standardization_factors, U_max

def prepare_ref_data(config, u, u_ref, v_ref, w_ref, p_ref, px_ref, py_ref, pz_ref, mask, U_max):

    # Prepare coordinates
    if config["setup"]["include_time"]:
        t_len, x_len, y_len, z_len = u.shape # (T, h, w, d)
    else:
        x_len, y_len, z_len = u.shape # (h, w, d)
        t_len = 1

    t_normalized, x_normalized, y_normalized, z_normalized, _ = create_and_normalize_coords(config, t_len, x_len, y_len, z_len)

    # Upsample coordinates
    t_ups = upsample_1d(t_normalized, config["ref_temporal_factor"], 'extend') if config["setup"]["include_time"] else []
    x_ups = upsample_1d(x_normalized, config["ref_spatial_factor"], mode='centered')
    y_ups = upsample_1d(y_normalized, config["ref_spatial_factor"], mode='centered')
    z_ups = upsample_1d(z_normalized, config["ref_spatial_factor"], mode='centered')

    # Create coordinate grid
    if config["setup"]["include_time"]:
        grids = np.meshgrid(t_ups, x_ups, y_ups, z_ups, indexing='ij')
    else:
        grids = np.meshgrid(x_ups, y_ups, z_ups, indexing='ij')

    flat_coordinates = [grid.ravel() for grid in grids] # T×h×w×d --> T*h*w*d = (29087100,)
    xyz_data = np.stack(flat_coordinates, axis=1) # (T*h*w*d, 4) = (29087100, 4)

    # Extract boundaries
    boundary_mask = compute_outer_boundary_mask(mask) # h×w×d = (81, 57, 50)

    if config["setup"]["include_time"]:
        # Tile the masks
        mask_flat = np.tile(mask.ravel(), len(t_ups))
        boundary_mask_flat = np.tile(boundary_mask.ravel(), len(t_ups))
    else:
        mask_flat = mask.ravel()
        boundary_mask_flat = boundary_mask.ravel()

    # Normalize velocity data
    if config["vel_normalization"] == "characteristic":
        U = config["constants"]["U"]
        u_normalized = u_ref / U
        v_normalized = v_ref / U
        w_normalized = w_ref / U

    elif config["vel_normalization"] == "max_velocity":
        u_normalized = u_ref / U_max
        v_normalized = v_ref / U_max
        w_normalized = w_ref / U_max

    # Flatten data into pointwise prediction
    u_flat = u_normalized.ravel()   # T×h×w×d --> T*h*w*d = (29087100,)
    v_flat = v_normalized.ravel()
    w_flat = w_normalized.ravel()

    velocities = [u_flat, v_flat, w_flat]

    if config["setup"]["include_pressure"]:
        rho, U, L = config["constants"]["rho"], config["constants"]["U"], config["constants"]["L"]
        p_normalized = p_ref / (rho*(U**2))
        p_flat = p_normalized.reshape(-1)
        velocities.append(p_flat)

    if config["training"]["reference_gradients"]:

        #_, _, _, std_x, _, std_y, _, std_z = standardization_factors
        # px_normalized = px_ref * L * std_x / (rho*(U**2))
        # py_normalized = py_ref * L * std_y / (rho*(U**2))
        # pz_normalized = pz_ref * L * std_z / (rho*(U**2))

        px_flat = px_ref.reshape(-1)
        py_flat = py_ref.reshape(-1)
        pz_flat = pz_ref.reshape(-1)
        velocities.append(px_flat)
        velocities.append(py_flat)
        velocities.append(pz_flat)

    # Ground truth data
    uvw_data_ref = np.stack(velocities, axis=1) # (T*h*w*d, 4) = (29087100, 4)
    
    return uvw_data_ref, xyz_data, mask_flat, boundary_mask_flat

def extract_fluid_region(uvw_data, xyz_data, mask_flat, print_fluid_points=False):

    fluid_indices = mask_flat == 1

    if print_fluid_points:
        print(f"Number of fluid-containing points per timestep: {np.sum(fluid_indices)}")
        print(f"Out of: {mask_flat.size}")

    uvw_fluid = uvw_data[fluid_indices]
    xyz_fluid = xyz_data[fluid_indices]

    return uvw_fluid, xyz_fluid

def sample_collocation_points(config, xyz_data, mask):

    # np.random.seed(123)

    if not config["collocation_in_fluid"]:

        # Sample random points
        for dim in range(xyz_data.shape[1]):

            # Initialize output array
            coll_points = np.empty((config["collocation_points"], xyz_data.shape[1])) # (N, 4)

            # Extract min & max values
            mins = xyz_data.min(axis=0)
            maxs = xyz_data.max(axis=0)

            # Sample points random
            coll_points[:, dim] = np.random.uniform(low=mins[dim], high=maxs[dim], size=config["collocation_points"])

        return coll_points
    
    else:

        # Sample random points in fluid region
        xyz_fluid = xyz_data[mask == 1]
        
        # Sample without replacement
        #indices = np.random.choice(len(xyz_fluid), size=config.collocation_points, replace=True) 
        #sampled_points = xyz_fluid[indices]

        # Sample with replacement
        data_voxels = len(xyz_fluid)
        N_c = config["collocation_points"]

        # Compute repeats and leftovers
        repeat_times = N_c // data_voxels
        remaining_points = N_c - (repeat_times * data_voxels)

        # Repeated indices
        repeated_indices = np.tile(np.arange(data_voxels), repeat_times)

        # Remaining indices sampled without replacement
        if remaining_points > 0:
            random_indices = np.random.choice(data_voxels, size=remaining_points, replace=False)
            all_indices = np.concatenate([repeated_indices, random_indices])
        else:
            all_indices = repeated_indices

        # Shuffle indices so repetitions are mixed
        sampled_points = xyz_fluid[all_indices]        

        # Add noise to sampled fluid points
        unique_vals = [np.unique(xyz_data[:, i]) for i in range(xyz_data.shape[1])]
        min_distances = [vals[1] - vals[0] for vals in unique_vals]

        # Apply random noise
        random_vals = np.random.uniform(-0.5, 0.5, size=sampled_points.shape)
        noise = random_vals * min_distances
        coll_points = sampled_points + noise

        return coll_points
    
def sample_boundary_points(config, xyz_data, boundary_mask):

    # np.random.seed(123)

    # Sample random points in boundary region
    bound_points = xyz_data[boundary_mask == 1]

    if config["setup"]["include_time"]:

        # Sample random timesteps
        random_times = np.random.uniform(low=xyz_data[:, 0].min(), high=xyz_data[:, 0].max(), size=config["boundary_repetitions"])
        
        # Repeat boundary points at each random timestep
        boundary_spatial = np.unique(bound_points[:, 1:], axis=0)
        repeated_spatial = np.tile(boundary_spatial, (config["boundary_repetitions"], 1))
        repeated_times = np.repeat(random_times, boundary_spatial.shape[0])  
        bound_points = np.column_stack((repeated_times, repeated_spatial)) 

    return bound_points
