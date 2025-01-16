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
    with h5py.File(config.data_file, mode='r') as hf:
        
        # Crop the data
        if config.setup.include_time:
            u = np.asarray(hf['u'][config.domain.t_start:config.domain.t_end, 
                                config.domain.x_start:config.domain.x_end, 
                                config.domain.y_start:config.domain.y_end, 
                                config.domain.z_start:config.domain.z_end])
            v = np.asarray(hf['v'][config.domain.t_start:config.domain.t_end, 
                                config.domain.x_start:config.domain.x_end, 
                                config.domain.y_start:config.domain.y_end, 
                                config.domain.z_start:config.domain.z_end])
            w = np.asarray(hf['w'][config.domain.t_start:config.domain.t_end, 
                                config.domain.x_start:config.domain.x_end, 
                                config.domain.y_start:config.domain.y_end, 
                                config.domain.z_start:config.domain.z_end])
            
            p = np.asarray(hf['p'][config.domain.t_start:config.domain.t_end, 
                                config.domain.x_start:config.domain.x_end, 
                                config.domain.y_start:config.domain.y_end, 
                                config.domain.z_start:config.domain.z_end]
                                ) if config.setup.include_pressure else None
        else:
            t_index = config.domain.t_start

            u = np.asarray(hf['u'][t_index, 
                                config.domain.x_start:config.domain.x_end, 
                                config.domain.y_start:config.domain.y_end, 
                                config.domain.z_start:config.domain.z_end])
            v = np.asarray(hf['v'][t_index, 
                                config.domain.x_start:config.domain.x_end, 
                                config.domain.y_start:config.domain.y_end, 
                                config.domain.z_start:config.domain.z_end])
            w = np.asarray(hf['w'][t_index, 
                                config.domain.x_start:config.domain.x_end, 
                                config.domain.y_start:config.domain.y_end, 
                                config.domain.z_start:config.domain.z_end])
            
            p = np.asarray(hf['p'][t_index,
                                config.domain.x_start:config.domain.x_end, 
                                config.domain.y_start:config.domain.y_end, 
                                config.domain.z_start:config.domain.z_end]
                                ) if config.setup.include_pressure else None

        # T×h×w×d = (126, 81, 57, 50)

        mask = np.asarray(hf['mask'])
        if len(mask.shape) == 4: 
            mask = mask[0]

        mask = mask[config.domain.x_start:config.domain.x_end, 
                    config.domain.y_start:config.domain.y_end, 
                    config.domain.z_start:config.domain.z_end]
        # h×w×d = (81, 57, 50)

    return u, v, w, p, mask

def load_ref_data(config):
    
    # Load data
    with h5py.File(config.data_file_ref, mode='r') as hf:
        
        # Crop the data
        if config.setup.include_time:
            u = np.asarray(hf['u'][config.domain.t_start*config.ref_temporal_factor:config.domain.t_end*config.ref_temporal_factor,
                                config.domain.x_start*config.ref_spatial_factor:config.domain.x_end*config.ref_spatial_factor, 
                                config.domain.y_start*config.ref_spatial_factor:config.domain.y_end*config.ref_spatial_factor, 
                                config.domain.z_start*config.ref_spatial_factor:config.domain.z_end*config.ref_spatial_factor])
            v = np.asarray(hf['v'][config.domain.t_start*config.ref_temporal_factor:config.domain.t_end*config.ref_temporal_factor,
                                config.domain.x_start*config.ref_spatial_factor:config.domain.x_end*config.ref_spatial_factor, 
                                config.domain.y_start*config.ref_spatial_factor:config.domain.y_end*config.ref_spatial_factor, 
                                config.domain.z_start*config.ref_spatial_factor:config.domain.z_end*config.ref_spatial_factor])
            w = np.asarray(hf['w'][config.domain.t_start*config.ref_temporal_factor:config.domain.t_end*config.ref_temporal_factor,
                                config.domain.x_start*config.ref_spatial_factor:config.domain.x_end*config.ref_spatial_factor, 
                                config.domain.y_start*config.ref_spatial_factor:config.domain.y_end*config.ref_spatial_factor, 
                                config.domain.z_start*config.ref_spatial_factor:config.domain.z_end*config.ref_spatial_factor])
            
            p = np.asarray(hf['p'][config.domain.t_start*config.ref_temporal_factor:config.domain.t_end*config.ref_temporal_factor, 
                                config.domain.x_start*config.ref_spatial_factor:config.domain.x_end*config.ref_spatial_factor, 
                                config.domain.y_start*config.ref_spatial_factor:config.domain.y_end*config.ref_spatial_factor, 
                                config.domain.z_start*config.ref_spatial_factor:config.domain.z_end*config.ref_spatial_factor]
                                ) if config.setup.include_pressure else None
        else:
            t_index = config.domain.t_start

            u = np.asarray(hf['u'][t_index, 
                                config.domain.x_start*config.ref_spatial_factor:config.domain.x_end*config.ref_spatial_factor, 
                                config.domain.y_start*config.ref_spatial_factor:config.domain.y_end*config.ref_spatial_factor, 
                                config.domain.z_start*config.ref_spatial_factor:config.domain.z_end*config.ref_spatial_factor])
            v = np.asarray(hf['v'][t_index, 
                                config.domain.x_start*config.ref_spatial_factor:config.domain.x_end*config.ref_spatial_factor, 
                                config.domain.y_start*config.ref_spatial_factor:config.domain.y_end*config.ref_spatial_factor, 
                                config.domain.z_start*config.ref_spatial_factor:config.domain.z_end*config.ref_spatial_factor])
            w = np.asarray(hf['w'][t_index, 
                                config.domain.x_start*config.ref_spatial_factor:config.domain.x_end*config.ref_spatial_factor, 
                                config.domain.y_start*config.ref_spatial_factor:config.domain.y_end*config.ref_spatial_factor, 
                                config.domain.z_start*config.ref_spatial_factor:config.domain.z_end*config.ref_spatial_factor])
            
            p = np.asarray(hf['p'][t_index,
                                config.domain.x_start*config.ref_spatial_factor:config.domain.x_end*config.ref_spatial_factor, 
                                config.domain.y_start*config.ref_spatial_factor:config.domain.y_end*config.ref_spatial_factor, 
                                config.domain.z_start*config.ref_spatial_factor:config.domain.z_end*config.ref_spatial_factor]
                                ) if config.setup.include_pressure else None

        mask = np.asarray(hf['mask'])
        if len(mask.shape) == 4: 
            mask = mask[0]

        mask = mask[config.domain.x_start*config.ref_spatial_factor:config.domain.x_end*config.ref_spatial_factor, 
                    config.domain.y_start*config.ref_spatial_factor:config.domain.y_end*config.ref_spatial_factor, 
                    config.domain.z_start*config.ref_spatial_factor:config.domain.z_end*config.ref_spatial_factor]

    return u, v, w, p, mask

def create_and_normalize_coords(config, t_len, x_len, y_len, z_len):
    
    # Extract resolutions
    dx, dy, dz = config.resolution.dx, config.resolution.dy, config.resolution.dz
    dt = config.resolution.dt

    # Create linspaces
    t = np.linspace(dt, t_len * dt, t_len)
    x = np.linspace(dx, x_len * dx, x_len) # (h,) = (81, ) , [0.0005 0.001 ... 0.0405] (voxel centers)
    y = np.linspace(dy, y_len * dy, y_len)
    z = np.linspace(dz, z_len * dz, z_len)

    # FOV starts at 0.00025 or dx/2
    # FOV ends at 0.04075 = x_len * dx + dx/2

    # Normalize coordinates
    if config.coords_characteristic:
        L, T = config.constants.L, config.constants.T
        t = t / T
        x = x / L
        y = y / L
        z = z / L

    t_normalized = None

    standardization_factors = None
    if config.coords_normalization == "standardize":
        x_normalized, mean_x, std_x = standardize(x)
        y_normalized, mean_y, std_y = standardize(y)
        z_normalized, mean_z, std_z = standardize(z)

        if config.setup.include_time:
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

    elif config.coords_normalization == "min_max":
        x_normalized, min_x, max_x = min_max_normalize(x)
        y_normalized, min_y, max_y = min_max_normalize(y)
        z_normalized, min_z, max_z = min_max_normalize(z)
        if config.setup.include_time:
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
        step = dx / factor
        # old approach A
        N_new = len(arr) * factor
        return np.linspace(arr[0], arr[-1] + step, N_new)

    elif mode == 'centered':
        # old approach B
        dx_hr = dx / factor
        start = arr[0] - dx/2 + dx_hr/2
        end   = arr[-1] + dx/2 - dx_hr/2
        N_new = len(arr) * factor
        return np.linspace(start, end, N_new)

    else:
        raise ValueError("Unknown mode. Use 'extend' or 'centered'.")

def prepare_data(config, u, v, w, p, mask):

    U_max = max(u.max(), v.max(), w.max())

    # Normalize velocity data
    if config.vel_normalization == "characteristic":
        U = config.constants.U
        u_normalized = u / U
        v_normalized = v / U
        w_normalized = w / U

    elif config.vel_normalization == "max_velocity":
        u_normalized = u / U_max
        v_normalized = v / U_max
        w_normalized = w / U_max

    # Flatten data into pointwise prediction
    u_flat = u_normalized.ravel()   # T×h×w×d --> T*h*w*d = (29087100,)
    v_flat = v_normalized.ravel()
    w_flat = w_normalized.ravel()

    velocities = [u_flat, v_flat, w_flat]

    if config.setup.include_pressure:
        rho, U = config.constants.rho, config.constants.U
        p_normalized = p / (rho*(U**2))
        p_flat = p_normalized.reshape(-1)
        velocities.append(p_flat)

    # Ground truth data
    uvw_data = np.stack(velocities, axis=1) # (T*h*w*d, 4) = (29087100, 4)

    # Prepare coordinates
    if config.setup.include_time:
        t_len, x_len, y_len, z_len = u.shape # (T, h, w, d)
    else:
        x_len, y_len, z_len = u.shape # (h, w, d)
        t_len = 1

    t_normalized, x_normalized, y_normalized, z_normalized, standardization_factors = create_and_normalize_coords(config, t_len, x_len, y_len, z_len)

    # Create coordinate grid
    if config.setup.include_time:
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

    if config.setup.include_time:
        # Tile the masks
        mask_flat = np.tile(mask.ravel(), t_len)
        boundary_mask_flat = np.tile(boundary_mask.ravel(), t_len)
    else:
        mask_flat = mask.ravel()
        boundary_mask_flat = boundary_mask.ravel()

    return uvw_data, xyz_data, mask_flat, boundary_mask_flat, standardization_factors, U_max

def prepare_ref_data(config, u, mask):

    # Prepare coordinates
    if config.setup.include_time:
        t_len, x_len, y_len, z_len = u.shape # (T, h, w, d)
    else:
        x_len, y_len, z_len = u.shape # (h, w, d)
        t_len = 1

    t_normalized, x_normalized, y_normalized, z_normalized, _ = create_and_normalize_coords(config, t_len, x_len, y_len, z_len)

    # Upsample coordinates
    t_ups = upsample_1d(t_normalized, config.ref_temporal_factor, 'extend') if config.setup.include_time else []
    x_ups = upsample_1d(x_normalized, config.ref_spatial_factor, mode='centered')
    y_ups = upsample_1d(y_normalized, config.ref_spatial_factor, mode='centered')
    z_ups = upsample_1d(z_normalized, config.ref_spatial_factor, mode='centered')

    # Create coordinate grid
    if config.setup.include_time:
        grids = np.meshgrid(t_ups, x_ups, y_ups, z_ups, indexing='ij')
    else:
        grids = np.meshgrid(x_ups, y_ups, z_ups, indexing='ij')

    flat_coordinates = [grid.ravel() for grid in grids] # T×h×w×d --> T*h*w*d = (29087100,)
    xyz_data = np.stack(flat_coordinates, axis=1) # (T*h*w*d, 4) = (29087100, 4)

    # Extract boundaries
    boundary_mask = compute_outer_boundary_mask(mask) # h×w×d = (81, 57, 50)

    if config.setup.include_time:
        # Tile the masks
        mask_flat = np.tile(mask.ravel(), t_len)
        boundary_mask_flat = np.tile(boundary_mask.ravel(), t_len)
    else:
        mask_flat = mask.ravel()
        boundary_mask_flat = boundary_mask.ravel()

    return xyz_data, mask_flat, boundary_mask_flat

def extract_fluid_region(uvw_data, xyz_data, mask_flat):

    fluid_indices = mask_flat == 1

    uvw_fluid = uvw_data[fluid_indices]
    xyz_fluid = xyz_data[fluid_indices]

    return uvw_fluid, xyz_fluid

def sample_collocation_points(config, xyz_data, mask):

    # np.random.seed(123)

    if not config.collocation_in_fluid:

        # Sample random points
        for dim in range(xyz_data.shape[1]):

            # Initialize output array
            coll_points = np.empty((config.collocation_points, xyz_data.shape[1])) # (N, 4)

            # Extract min & max values
            mins = xyz_data.min(axis=0)
            maxs = xyz_data.max(axis=0)

            # Sample points random
            coll_points[:, dim] = np.random.uniform(low=mins[dim], high=maxs[dim], size=config.collocation_points)

        return coll_points
    
    else:

        # Sample random points in fluid region
        xyz_fluid = xyz_data[mask == 1]
        indices = np.random.choice(len(xyz_fluid), size=config.collocation_points, replace=True)
        sampled_points = xyz_fluid[indices]

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

    if config.setup.include_time:

        # Sample random timesteps
        random_times = np.random.uniform(low=xyz_data[:, 0].min(), high=xyz_data[:, 0].max(), size=config.boundary_repetitions)
        
        # Repeat boundary points at each random timestep
        boundary_spatial = np.unique(bound_points[:, 1:], axis=0)
        repeated_spatial = np.tile(boundary_spatial, (config.boundary_repetitions, 1))
        repeated_times = np.repeat(random_times, boundary_spatial.shape[0])  
        bound_points = np.column_stack((repeated_times, repeated_spatial)) 

    return bound_points
