import numpy as np
import h5py

def standardize(data):
    mean = np.mean(data)
    std = np.std(data)
    normalized_data = (data - mean) / std
    return normalized_data, mean, std

def min_max_normalize(data):
    """Normalize the data to a [0, 1] range using min-max scaling."""
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val), min_val, max_val

def compute_boundary_mask(mask):
    t, x, y, z = mask.shape
    boundary_mask = np.zeros_like(mask, dtype=int)

    for time_step in range(t):
        current_mask = mask[time_step]
        
        # Check for boundary in each direction
        for i in range(1, x-1):
            for j in range(1, y-1):
                for k in range(1, z-1):
                    if current_mask[i, j, k] == 1:
                        if (current_mask[i+1, j, k] == 0 or current_mask[i-1, j, k] == 0 or
                            current_mask[i, j+1, k] == 0 or current_mask[i, j-1, k] == 0 or
                            current_mask[i, j, k+1] == 0 or current_mask[i, j, k-1] == 0):
                            boundary_mask[time_step, i, j, k] = 1
    return boundary_mask

def compute_outer_boundary_mask(mask):
    x, y, z = mask.shape
    outer_boundary_mask = np.zeros_like(mask, dtype=int)

    current_mask = mask
    
    # Check for outer boundary in each direction
    for i in range(x):
        for j in range(y):
            for k in range(z):
                if current_mask[i, j, k] == 0:
                    if ((i > 0 and current_mask[i-1, j, k] == 1) or
                        (i < x-1 and current_mask[i+1, j, k] == 1) or
                        (j > 0 and current_mask[i, j-1, k] == 1) or
                        (j < y-1 and current_mask[i, j+1, k] == 1) or
                        (k > 0 and current_mask[i, j, k-1] == 1) or
                        (k < z-1 and current_mask[i, j, k+1] == 1)):
                        outer_boundary_mask[i, j, k] = 1
    return outer_boundary_mask

def generate_collocation_points(collocation_points, t_min, t_max, x_min, x_max, y_min, y_max, z_min, z_max):

    # Initialize output array
    points = np.empty((collocation_points, 4))
    
    # Column 0: Time (t) values
    points[:, 0] = np.random.uniform(low=t_min, high=t_max, size=collocation_points)
    points[:, 1] = np.random.uniform(low=x_min, high=x_max, size=collocation_points)
    points[:, 2] = np.random.uniform(low=y_min, high=y_max, size=collocation_points)
    points[:, 3] = np.random.uniform(low=z_min, high=z_max, size=collocation_points)
    
    return points

def generate_collocation_points_in_fluid_region(n_timepoints, t_min, t_max, x_min, x_max, y_min, y_max, z_min, z_max, mask, n_collocation):

    # Initialize output array
    point_coord = np.empty((n_collocation, 4))
    
    # Column 0: Time (t) values
    point_coord[:, 0] = np.random.uniform(low=t_min, high=t_max, size=n_collocation)
    point_coord[:, 1] = np.random.uniform(low=x_min, high=x_max, size=n_collocation)
    point_coord[:, 2] = np.random.uniform(low=y_min, high=y_max, size=n_collocation)
    point_coord[:, 3] = np.random.uniform(low=z_min, high=z_max, size=n_collocation)

    if mask.ndim == 4:
        orig_shape = mask.shape
    elif mask.ndim == 3:
        orig_shape = (n_timepoints, *mask.shape)

    collo_indices, _ = coords_to_matrix_indices(point_coord, 1, 1, dt, dx, dy, dz,  mean_t, std_t, mean_x, std_x, mean_y, std_y, mean_z, std_z, L, T,  orig_shape)
    
    # for each collocation point this array is 1 if its in the fluid region and 0 otherwise
    check_fluiregion = np.zeros(n_collocation)
    check_fluiregion[np.where(mask[collo_indices[:, 0], collo_indices[:, 1], collo_indices[:, 2], collo_indices[:, 3]] >=0.5)] = 1

    # get all indices that are within the fluid region
    collo_indices_fluid = collo_indices[np.where(check_fluiregion == 1)]

    # visual check
    collect_points = np.zeros((collo_indices_fluid.shape[0], 4))
    collect_points = point_coord[np.where(check_fluiregion == 1), :]
    collect_points = collect_points.squeeze()

    return collect_points

def generate_boundary_points(boundary_mask, time_boundary_points, mean_x, std_x, mean_y, std_y, mean_z, std_z):

    x_coord, y_coord, z_coord = np.where(boundary_mask >= 0.5)
    x_coord = (((x_coord+1)*dx)/L - mean_x)/std_x
    y_coord = (((y_coord+1)*dx)/L - mean_y)/std_y
    z_coord = (((z_coord+1)*dx)/L - mean_z)/std_z

    t_coord = np.random.uniform(low=t_min, high=t_max, size=time_boundary_points)
    xyz_coords = np.vstack([x_coord, y_coord, z_coord]).T

    # Repeat the coordinates for each time point
    repeated_coords = np.tile(xyz_coords, (len(t_coord), 1))

    # Repeat the time points to match the number of coordinate sets
    time_column = np.repeat(t_coord, xyz_coords.shape[0])

    # Combine the time points with the coordinates
    txyz_boundary = np.column_stack((time_column, repeated_coords))

    return txyz_boundary

def coords_to_matrix_indices(txyz_coords, spatial_increase, time_increase, dt, dx, dy, dz, mean_t, std_t, mean_x, std_x, mean_y, std_y, mean_z, std_z, L,T, orig_shape):

    new_matrix_shape = orig_shape[0]*time_increase, orig_shape[1]*spatial_increase, orig_shape[2]*spatial_increase, orig_shape[3]*spatial_increase

    dx_factor = 1.0/spatial_increase
    dt_factor = 1.0/time_increase

    # Update transformations according to new matrix dimensions
    t_indices = np.round((txyz_coords[:, 0] * std_t + mean_t) * (T/((dt*dt_factor))) - 1).astype(int) 
    x_indices = np.round((txyz_coords[:, 1] * std_x + mean_x) * (L/((dx*dx_factor))) - 1).astype(int) 
    y_indices = np.round((txyz_coords[:, 2] * std_y + mean_y) * (L/((dy*dx_factor))) - 1).astype(int) 
    z_indices = np.round((txyz_coords[:, 3] * std_z + mean_z) * (L/((dz*dx_factor))) - 1).astype(int) 

    txyz_indices = np.vstack([t_indices, x_indices, y_indices, z_indices]).T

    #print(f'Coordinates are transformed into matrix with spatial resolution increase of {spatial_increase}x and temporal increase of {time_increase}x and resulting matrix of {new_matrix_shape}')
    
    return txyz_indices, new_matrix_shape

def matrix_to_coordinates(mask, dt, dx, dy, dz, mean_t, mean_x, mean_y, mean_z, std_t,   std_x, std_y, std_z, L, T):

    if mask.ndim == 4:
         # spatial coordinantes of fluid region
        t_coord, x_coord, y_coord, z_coord = np.where(mask >= 0.5)
        t_coord = (((t_coord+1)*dt)/T - mean_t)/std_t
        x_coord = (((x_coord+1)*dx)/L - mean_x)/std_x
        y_coord = (((y_coord+1)*dy)/L - mean_y)/std_y
        z_coord = (((z_coord+1)*dz)/L - mean_z)/std_z

        return np.vstack([t_coord, x_coord, y_coord, z_coord]).T
    elif mask.ndim == 3:

        # spatial coordinantes of fluid region
        x_coord, y_coord, z_coord = np.where(mask >= 0.5)
        x_coord = (((x_coord+1)*dx)/L - mean_x)/std_x
        y_coord = (((y_coord+1)*dy)/L - mean_y)/std_y
        z_coord = (((z_coord+1)*dz)/L - mean_z)/std_z

        return np.vstack([x_coord, y_coord, z_coord]).T

    else:
        raise ValueError("mask.ndim must be 3 or 4")
    
