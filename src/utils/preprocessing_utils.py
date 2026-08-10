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


def S(x, omega, af):
    """
    Compute the activation function results for a given input and weight matrix.

    This function applies a specified activation function to the matrix product
    of the input data and a weight matrix, along with a bias term. The
    operation computes the linear combination of features in the input and
    then passes it through the provided activation function.

    :param x: A 2D array containing the input data of shape (N, d), where N is
        the number of samples and d is the number of features.
    :param omega: A 2D array of shape (d+1, M) representing the weight matrix (frequencies)
        and bias, where d is the number of features and M is the number of
        output nodes or activation outputs.
    :param af: A callable activation function that is applied element-wise to
        the linear transformation of the input data.
    :return: The result of applying the activation function to the computed
        weighted input as a 2D array of shape (N, M).
    """
    N = x.shape[0]
    sv = af(np.matmul(np.c_[x, np.ones(N)], omega))
    return sv

def get_c(x, y, lambda_reg, omega, K, af):
    """
    Solves the linear least squares problem in the amplitudes 'c' for given frequencies 'omega'.

    :param x: Input data matrix of shape (N, d), where N is the number of samples and d
        is the number of features.
    :param y: Target vector of shape (N,) which corresponds to the observed or target
        values for each sample.
    :param lambda_reg: Regularization parameter to penalize the model complexity and
        prevent overfitting.
    :param omega: Frequencies and biases matrix of shape (d + 1, K).
    :param K: Number of frequencies to sample.
    :param af: Activation function to use in the sampling. (Cosine)
    :return: Amplitudes matrix 'c' of shape (K,).
    """
    N = x.shape[0]
    St = S(x, omega, af)

    # Normal equations
    cm = np.matmul(np.transpose(St), St) + N * lambda_reg * np.identity(K)
    return np.linalg.solve(cm, np.matmul(np.transpose(St), y))

def am_resample_im_reg(x, y, xvalid, yvalid, M, K, N, delta, lambda_reg,
                       gamma, af, resampling=True, DO_METROPOLIS_TEST=True):
    """
    Samples frequencies for the random Fourier features layer adaptively using random walk with
    resampling and/or Metropolis sampling.

    This function implements the RFF sampling algorithm from https://arxiv.org/abs/2410.06399.


    :param x: Input training data.
    :param y: Target values for the training data.
    :param xvalid: Validation dataset.
    :param yvalid: Target values for the validation dataset.
    :param M: Number of iterations to perform.
    :param K: Number of frequencies to sample.
    :param N: Number of data points in the training dataset.
    :param delta: Standard deviation used in the random walk.
    :param lambda_reg: Regularization parameter for the linear least squares problem. (Tikhonov parameter)
    :param gamma: Exponent parameter for the Metropolis test.
    :param af: Activation function to use in for the sampling. (Cosine)
    :param resampling: Boolean flag indicating whether resampling is to be performed.
        Defaults to True.
    :param DO_METROPOLIS_TEST: Boolean flag enabling or disabling the Metropolis
        update mechanism. Defaults to True.
    :return: A tuple containing the following:
        (1) Sampled random Fourier features frequencies and biases matrix `omega` after `M` iterations.
        (2) Amplitudes matrix `c` after `M` iterations.
        (3) Array of training error values over all iterations.
        (4) Array of validation error values over all iterations.
        (5) Array of time taken to reach each iteration.
    """
    d = x.shape[1]
    ve = np.zeros(M)
    te = np.zeros(M)
    start_time = time.time()
    time_arr = np.zeros(shape=M)

    omega = np.zeros(shape=(d + 1, K))
    c = get_c(x, y, lambda_reg, omega, K, af)

    for t in range(0, M):
        if DO_METROPOLIS_TEST:
            omega_prime = omega + delta * np.random.normal(0, 1, size=(d + 1, K))
            c_prime = get_c(x, y, lambda_reg, K, af)

            for k in range(0, K):
                if (np.linalg.norm(c_prime[k,:]) / np.linalg.norm(c[k,:])) ** gamma >= np.random.random():
                    omega[:, k] = omega_prime[:, k]
        else:
            omega = omega + delta * np.random.normal(0, 1, size=(d + 1, K))

        c = get_c(x, y, lambda_reg, omega, K, af)

        if resampling:
            amp_pmf = np.linalg.norm(c, axis=1) / np.sum(np.linalg.norm(c, axis=1))
            omega = omega[:, np.random.choice(c.shape[0], K, p=amp_pmf)]

        c = get_c(x, y, lambda_reg, omega, K, af)

        St = S(x, omega, af)
        beta_train = np.matmul(St, c)
        te[t] = np.sum(np.linalg.norm(beta_train - y)**2)/N
        St = S(xvalid, omega, af)
        beta_valid = np.matmul(St, c)
        ve[t] = np.sum(np.linalg.norm(beta_valid - yvalid) ** 2) / len(xvalid)
        time_arr[t] = time.time() - start_time


        if np.mod(t, 100) == 0:
            print('t = ', t)
            print('K = ', K)
            print('Training error = ', te[t])

    c = get_c(x, y, lambda_reg, omega, K, af)

    return omega, c, te, ve, time_arr

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
    
