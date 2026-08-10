import numpy as np
import torch

def mse_loss(uvw_data, uvw_pred, config):
    
    u_pred, v_pred, w_pred = uvw_pred[..., 0], uvw_pred[..., 1], uvw_pred[..., 2]
    u, v, w = uvw_data[..., 0], uvw_data[..., 1], uvw_data[..., 2]

    mse_u = (u_pred - u) ** 2
    mse_v = (v_pred - v) ** 2
    mse_w = (w_pred - w) ** 2

    mse_vel = torch.mean(config.training.u_weight*mse_u + config.training.v_weight*mse_v + config.training.w_weight*mse_w)

    if config.training.pressure_in_data_loss:
        p_pred = uvw_pred[..., 3]
        p = uvw_data[..., 3]

        mse_p = (p_pred - p) ** 2

        mse_vel += config.training.p_weight*torch.mean(mse_p)

    return mse_vel

def cosine_loss(uvw_data, uvw_pred, config):
    
    u_pred, v_pred, w_pred = uvw_pred[..., 0]*config.U_max, uvw_pred[..., 1]*config.U_max, uvw_pred[..., 2]*config.U_max
    u, v, w = uvw_data[..., 0]*config.U_max, uvw_data[..., 1]*config.U_max, uvw_data[..., 2]*config.U_max

    Kv = torch.pi / config.constants.venc
    cosine_u = 1 - torch.cos(Kv * (u_pred - u))
    cosine_v = 1 - torch.cos(Kv * (v_pred - v))
    cosine_w = 1 - torch.cos(Kv * (w_pred - w))

    cosine_vel = torch.mean(config.training.u_weight*cosine_u + config.training.v_weight*cosine_v + config.training.w_weight*cosine_w)

    if config.training.pressure_in_data_loss:
        p_pred = uvw_pred[..., 3]
        p = uvw_data[..., 3]

        cosine_p = 1 - torch.cos(Kv * (p_pred - p))
        
        cosine_vel += config.training.p_weight*torch.mean(cosine_p)
    
    return cosine_vel

def fluid_weighted_mse_loss(uvw_data, uvw_pred, mask, config):
    
    u_pred, v_pred, w_pred = uvw_pred[..., 0], uvw_pred[..., 1], uvw_pred[..., 2]
    u, v, w = uvw_data[..., 0], uvw_data[..., 1], uvw_data[..., 2]

    mse_u = (u_pred - u) ** 2
    mse_v = (v_pred - v) ** 2
    mse_w = (w_pred - w) ** 2

    # Get non-fluid mask
    mask_flattened = mask.flatten()
    non_fluid_mask = 1 - mask_flattened

    mse_u_fluid = mse_u * mask_flattened
    mse_u_non_fluid = mse_u * non_fluid_mask

    mse_v_fluid = mse_v * mask_flattened
    mse_v_non_fluid = mse_v * non_fluid_mask

    mse_w_fluid = mse_w * mask_flattened
    mse_w_non_fluid = mse_w * non_fluid_mask

    mse_u_total = config.training.u_weight*(mse_u_fluid.sum()/mask_flattened.sum() + mse_u_non_fluid.sum()/non_fluid_mask.sum())
    mse_v_total = config.training.v_weight*(mse_v_fluid.sum()/mask_flattened.sum() + mse_v_non_fluid.sum()/non_fluid_mask.sum())
    mse_w_total = config.training.w_weight*(mse_w_fluid.sum()/mask_flattened.sum() + mse_w_non_fluid.sum()/non_fluid_mask.sum())

    mse_vel = torch.mean(mse_u_total + mse_v_total + mse_w_total)

    if config.training.pressure_in_data_loss:

        p_pred = uvw_pred[..., 3]
        p = uvw_data[..., 3]

        mse_p = (p_pred - p) ** 2
        mse_p_fluid = mse_p * mask
        mse_p_non_fluid = mse_p * non_fluid_mask

        mse_p = mse_p_fluid.sum()/mask.sum() + mse_p_non_fluid.sum()/non_fluid_mask.sum()
        mse_vel += config.training.p_weight*mse_p

    return mse_vel

def vector_potential_fn(phi_pred, xyz_hr):

   # Separate out each predicted component
    phi_x = phi_pred[:, 0]  # shape (N,)
    phi_y = phi_pred[:, 1]
    phi_z = phi_pred[:, 2]

    # Compute partial derivatives
    dphix = torch.autograd.grad(phi_x, xyz_hr,
                                grad_outputs=torch.ones_like(phi_x),
                                create_graph=True, retain_graph=True)[0]  # shape (N,3)

    dphiy = torch.autograd.grad(phi_y, xyz_hr,
                                grad_outputs=torch.ones_like(phi_y),
                                create_graph=True, retain_graph=True)[0]

    dphiz = torch.autograd.grad(phi_z, xyz_hr,
                                grad_outputs=torch.ones_like(phi_z),
                                create_graph=True, retain_graph=True)[0]

    # Extract partial derivatives for each dimension:
    _, dphix_dy, dphix_dz = dphix[:, 0], dphix[:, 1], dphix[:, 2]
    dphiy_dx, _, dphiy_dz = dphiy[:, 0], dphiy[:, 1], dphiy[:, 2]
    dphiz_dx, dphiz_dy, _ = dphiz[:, 0], dphiz[:, 1], dphiz[:, 2]

    # Compute velocities
    u = dphiz_dy - dphiy_dz  # shape (N,)
    v = dphix_dz - dphiz_dx
    w = dphiy_dx - dphix_dy

    # Stack into (N, 3)
    uvw_pred = torch.stack([u, v, w], dim=1)

    return uvw_pred

def boundary_mse_loss(uvw_boundary_pred, config):
    
    u_pred, v_pred, w_pred = uvw_boundary_pred[..., 0], uvw_boundary_pred[..., 1], uvw_boundary_pred[..., 2]

    mse_u = (u_pred) ** 2
    mse_v = (v_pred) ** 2
    mse_w = (w_pred) ** 2

    mse_bound = torch.mean(config.training.u_weight*mse_u + config.training.v_weight*mse_v + config.training.w_weight*mse_w)

    if config.training.pressure_in_boundary_loss:
        p_pred = uvw_boundary_pred[..., 3]
        mse_p = (p_pred) ** 2
        mse_bound_p = torch.mean(mse_p)
        mse_bound += config.training.p_weight*mse_bound_p

    return mse_bound

def compute_gradient(outputs, inputs, grad_dim):
    # Function to compute derivatives
    grad_outputs = torch.ones_like(outputs)
    return torch.autograd.grad(outputs=outputs, inputs=inputs,
                                grad_outputs=grad_outputs, # Specify dimension - compute gradient of each element
                                create_graph=True,  # Allows for higher order gradients
                                retain_graph=True,  # Allows to compute more gradients on the same graph
                                only_inputs=True)[0][..., grad_dim]

def navier_stokes_loss(uvw_pred, xyz_collocation, standardization_factors, config):

    # Unpack velocity and pressure
    u, v, w, p = uvw_pred[..., 0], uvw_pred[..., 1], uvw_pred[..., 2], uvw_pred[..., 3]

    # First derivatives
    du_dt = compute_gradient(u, xyz_collocation, 0)
    du_dx = compute_gradient(u, xyz_collocation, 1)
    du_dy = compute_gradient(u, xyz_collocation, 2)
    du_dz = compute_gradient(u, xyz_collocation, 3)

    dv_dt = compute_gradient(v, xyz_collocation, 0)
    dv_dx = compute_gradient(v, xyz_collocation, 1)
    dv_dy = compute_gradient(v, xyz_collocation, 2)
    dv_dz = compute_gradient(v, xyz_collocation, 3)

    dw_dt = compute_gradient(w, xyz_collocation, 0)
    dw_dx = compute_gradient(w, xyz_collocation, 1)
    dw_dy = compute_gradient(w, xyz_collocation, 2)
    dw_dz = compute_gradient(w, xyz_collocation, 3)

    dp_dx = compute_gradient(p, xyz_collocation, 1)
    dp_dy = compute_gradient(p, xyz_collocation, 2)
    dp_dz = compute_gradient(p, xyz_collocation, 3)

    # Second derivatives
    d2u_dx2 = compute_gradient(du_dx, xyz_collocation, 1)
    d2u_dy2 = compute_gradient(du_dy, xyz_collocation, 2)
    d2u_dz2 = compute_gradient(du_dz, xyz_collocation, 3)

    d2v_dx2 = compute_gradient(dv_dx, xyz_collocation, 1)
    d2v_dy2 = compute_gradient(dv_dy, xyz_collocation, 2)
    d2v_dz2 = compute_gradient(dv_dz, xyz_collocation, 3)

    d2w_dx2 = compute_gradient(dw_dx, xyz_collocation, 1)
    d2w_dy2 = compute_gradient(dw_dy, xyz_collocation, 2)
    d2w_dz2 = compute_gradient(dw_dz, xyz_collocation, 3)
    
    # Extract constants
    U = config.constants.U
    L = config.constants.L
    rho = config.constants.rho
    mu = config.constants.mu
    Re = (rho*U*L)/mu # Reynolds number

    # density: 1.06, dynamic_viscosity: 0.035
    # rho: 1.06, mu: 0.035

    # Re = 

    if config.coords_normalization == "standardize":
        # Extract standardization factors
        _, std_t, _, std_x, _, std_y, _, std_z = standardization_factors
    else:
        std_t, std_x, std_y, std_z = 1.0, 1.0, 1.0, 1.0

    # Calculate residuals based on Navier-Stokes Equations
    momentum_u = (
        (1 / std_t) * du_dt 
        + (1 / std_x) * (u * du_dx) + (1 / std_y) * (v * du_dy) + (1 / std_z) * (w * du_dz) 
        + (1 / std_x) * dp_dx 
        - (1/Re) * ((1 / std_x**2) * d2u_dx2 + (1 / std_y**2) * d2u_dy2 + (1 / std_z**2) * d2u_dz2)
    )    

    momentum_v = (
        (1 / std_t) * dv_dt 
        + (1 / std_x) * (u * dv_dx) + (1 / std_y) * (v * dv_dy) + (1 / std_z) * (w * dv_dz) 
        + (1 / std_y) * dp_dy 
        - (1/Re) * ((1 / std_x**2) * d2v_dx2 + (1 / std_y**2) * d2v_dy2 + (1 / std_z**2) * d2v_dz2)
    ) 

    momentum_w = (
        (1 / std_t) * dw_dt 
        + (1 / std_x) * (u * dw_dx) + (1 / std_y) * (v * dw_dy) + (1 / std_z) * (w * dw_dz) 
        + (1 / std_z) * dp_dz 
        - (1/Re) * ((1 / std_x**2) * d2w_dx2 + (1 / std_y**2) * d2w_dy2 + (1 / std_z**2) * d2w_dz2)
    ) 

    # Calculate divergence
    div = (du_dx / std_x) + (dv_dy / std_y) + (dw_dz / std_z)

    # Calculate MSE
    momentum_loss_u = momentum_u ** 2
    momentum_loss_v = momentum_v ** 2
    momentum_loss_w = momentum_w ** 2

    momentum_loss = torch.mean(momentum_loss_u + momentum_loss_v + momentum_loss_w)
    div_loss = torch.mean(div ** 2)

    return momentum_loss, div_loss

def divergence_loss(uvw_pred, xyz_collocation, standardization_factors, config):

    # Unpack velocity and pressure
    u, v, w = uvw_pred[..., 0], uvw_pred[..., 1], uvw_pred[..., 2]

    if config.setup.include_time:
        # [t, x, y, z].
        x_dim, y_dim, z_dim = 1, 2, 3
    else:
        # [x, y, z].
        x_dim, y_dim, z_dim = 0, 1, 2

    # First derivatives
    du_dx = compute_gradient(u, xyz_collocation, x_dim)
    dv_dy = compute_gradient(v, xyz_collocation, y_dim)
    dw_dz = compute_gradient(w, xyz_collocation, z_dim)

    if config.coords_normalization == "standardize":
        if config.setup.include_time:
            _, _, _, std_x, _, std_y, _, std_z = standardization_factors
        else:
            _, std_x, _, std_y, _, std_z = standardization_factors
    else:
        std_x, std_y, std_z = 1.0, 1.0, 1.0

    # Calculate divergence
    div = (du_dx / std_x) + (dv_dy / std_y) + (dw_dz / std_z)

    # Calculate loss
    div_loss = torch.mean(div ** 2)

    return div_loss


# def five_point_encoding_cartesian(u, v, w, venc, m=None):
#     """
#     Convert velocity triplet (u, v, w) in physical units to five complex Cartesian
#     images using the balanced five-point encoding scheme of Johnson & Markl (2010),
#     as adopted by Fathi et al. (2020) Eqs. (15)-(17).
 
#     m: magnitude image (N,). If None, assumes m = 1 for all voxels.
 
#     Returns S_real, S_imag each of shape (N, 5).
#     """
#     kv = 1.0 / (2.0 * venc)
#     k_vecs = torch.tensor([
#         [ 0.0,  0.0,  0.0],   # r=0  reference
#         [-1.0, -1.0, -1.0],   # r=1
#         [-1.0, -1.0,  1.0],   # r=2
#         [-1.0,  1.0, -1.0],   # r=3
#         [ 1.0, -1.0, -1.0],   # r=4
#     ], dtype=u.dtype, device=u.device) * kv  # (5, 3)
 
#     uvw = torch.stack([u, v, w], dim=-1)     # (N, 3)
#     phi = 2.0 * torch.pi * (uvw @ k_vecs.T) # (N, 5)
 
#     if m is None:
#         mag = torch.ones(u.shape[0], 1, dtype=u.dtype, device=u.device)
#     else:
#         mag = m.unsqueeze(-1)                # (N, 1)
 
#     S_real = mag * torch.cos(phi)            # (N, 5)
#     S_imag = mag * (-torch.sin(phi))         # (N, 5)
 
#     return S_real, S_imag

def five_point_encoding_cartesian(u, v, w, venc, m=None):
    """
    Five-point balanced encoding per Johnson & Markl (2010), Table 1.
    Encoding matrix rows correspond to the four non-flow-compensated points
    (tetrahedral balanced) plus the flow-compensated reference (M1=0).

    Phase: phi_r = pi/venc * (k_r . v), where k_r in {-1,0,+1}^3.

    Returns S_real, S_imag each of shape (N, 5).
    """
    # Encoding directions from Table 1 (Johnson & Markl 2010)
    # Points 0-3: balanced tetrahedral (non-flow-compensated)
    # Point 4:    flow-compensated reference (M1 = 0)
    k_vecs = torch.tensor([
        [-1.0, -1.0, -1.0],   # r=0
        [+1.0, +1.0, -1.0],   # r=1
        [+1.0, -1.0, +1.0],   # r=2
        [-1.0, +1.0, +1.0],   # r=3
        [ 0.0,  0.0,  0.0],   # r=4  flow-compensated reference
    ], dtype=u.dtype, device=u.device)  # (5, 3)

    # Phase: phi = (pi / venc) * (k . v)
    # This follows from gamma * Delta_M1 * v = pi when v = venc and |k|=1
    uvw = torch.stack([u, v, w], dim=-1)          # (N, 3)
    phi = (torch.pi / venc) * (uvw @ k_vecs.T)   # (N, 5)

    if m is None:
        mag = torch.ones(u.shape[0], 1, dtype=u.dtype, device=u.device)
    else:
        mag = m.unsqueeze(-1)                      # (N, 1)

    S_real = mag * torch.cos(phi)                  # (N, 5)
    S_imag = mag * torch.sin(phi)                  # (N, 5)  note: +sin, not -sin

    return S_real, S_imag


 
def _get_gauss_legendre_points_weights(n_points: int, dtype, device):
    """
    Returns Gauss-Legendre quadrature points and weights on [-0.5, 0.5]
    (i.e. normalised to a unit voxel centred at the origin).
 
    For n_points in {1, 2, 3, 4, 5} the nodes and weights are hard-coded
    from standard tables for [-1,1] and then rescaled.
    """
    # Standard GL nodes and weights on [-1, 1]
    gl_tables = {
        1: ([0.0],
            [2.0]),
        2: ([-0.5773502691896258, 0.5773502691896258],
            [1.0, 1.0]),
        3: ([-0.7745966692414834, 0.0, 0.7745966692414834],
            [0.5555555555555556, 0.8888888888888888, 0.5555555555555556]),
        4: ([-0.8611363115940526, -0.3399810435848563,
              0.3399810435848563,  0.8611363115940526],
            [ 0.3478548451374538,  0.6521451548625461,
              0.6521451548625461,  0.3478548451374538]),
        5: ([-0.9061798459386640, -0.5384693101056831, 0.0,
              0.5384693101056831,  0.9061798459386640],
            [ 0.2369268850561891,  0.4786286704993665, 0.5688888888888889,
              0.4786286704993665,  0.2369268850561891]),
    }
    if n_points not in gl_tables:
        raise ValueError(f"Gauss-Legendre quadrature only implemented for "
                         f"n_points in {{1,2,3,4,5}}, got {n_points}.")
    nodes_11, weights_11 = gl_tables[n_points]
    # Rescale from [-1, 1] to [-0.5, 0.5]  (unit voxel centred at 0)
    nodes   = torch.tensor([n * 0.5 for n in nodes_11],   dtype=dtype, device=device)
    weights = torch.tensor([w * 0.5 for w in weights_11], dtype=dtype, device=device)
    return nodes, weights
 


def gaussian_quadrature_average(model, xyz_center, voxel_size_norm,
                                 n_quad_points: int, config):
    """
    Approximate the spatio-temporal average of the network prediction over
    each voxel using tensor-product Gauss-Legendre quadrature (Eq. 11-14 of
    Fathi et al. 2020).
 
    For a 3-D spatial voxel (time is NOT averaged — we evaluate at the voxel
    centre time as in the paper's single-timestep formulation):
 
        u_bar(r) ≈ sum_j w_j * u_NN(x_center + dx_j)
 
    where dx_j are the quadrature offsets scaled by half the voxel size and
    w_j are the corresponding quadrature weights (summing to 1 after normalisation).
 
    Args:
        model       : the network (already in eval or train mode).
        xyz_center  : (N, 4) tensor [t, x, y, z] of voxel centres (normalised).
        voxel_size_norm : (4,) tensor — voxel size in each normalised dimension
                          [dt_norm, dx_norm, dy_norm, dz_norm].
        n_quad_points   : number of 1-D Gauss-Legendre points (1–5). The total
                          number of model evaluations per voxel is n_quad_points^3
                          (spatial only; time is NOT integrated, matching Fathi).
        config      : training config.
 
    Returns:
        uvwp_avg : (N, out_dim) tensor — quadrature-averaged network output.
    """
    nodes, weights = _get_gauss_legendre_points_weights(
        n_quad_points, dtype=xyz_center.dtype, device=xyz_center.device)
 
    # Build tensor-product offsets over spatial dims (x, y, z) only.
    # Shape of offsets: (n^3, 3), shape of quad_weights: (n^3,)
    offsets_list = []
    quad_weights_list = []
    for xi, wi in zip(nodes, weights):
        for yi, wyi in zip(nodes, weights):
            for zi, wzi in zip(nodes, weights):
                offsets_list.append(torch.stack([xi, yi, zi]))
                quad_weights_list.append(wi * wyi * wzi)
 
    offsets      = torch.stack(offsets_list, dim=0)       # (n^3, 3)
    quad_weights = torch.stack(quad_weights_list, dim=0)  # (n^3,)
    # Normalise so weights sum to 1 (Gauss-Legendre already sums to 1 on [-1,1]
    # but after the [-0.5,0.5] rescale each 1-D sum = 1, so 3-D sum = 1 too)
    quad_weights = quad_weights / quad_weights.sum()
 
    # Scale offsets by normalised voxel sizes (spatial dims 1,2,3)
    dx_norm = voxel_size_norm[1:4]                        # (3,)
    offsets_scaled = offsets * dx_norm.unsqueeze(0)        # (n^3, 3)
 
    N = xyz_center.shape[0]
    n_total = offsets.shape[0]
 
    # Expand: (N, n^3, 4)
    xyz_expanded = xyz_center.unsqueeze(1).expand(N, n_total, 4).clone()
    # Add spatial offsets (dims 1,2,3); time dim (0) stays at voxel centre
    xyz_expanded[:, :, 1:4] += offsets_scaled.unsqueeze(0)
 
    # Flatten to (N*n^3, 4) for a single batched forward pass
    xyz_flat = xyz_expanded.reshape(N * n_total, 4)
    pred_flat = model(xyz_flat)                            # (N*n^3, out_dim)
    pred_per_point = pred_flat.reshape(N, n_total, -1)     # (N, n^3, out_dim)
 
    # Weighted average: sum_j w_j * pred_j
    uvwp_avg = (pred_per_point * quad_weights.view(1, n_total, 1)).sum(dim=1)  # (N, out_dim)
 
    return uvwp_avg

def four_point_encoding_cartesian(u, v, w, venc, m=None):
    """
    Four-point referenced encoding per Johnson & Markl (2010), Table 1.
    One reference point (M1=0) plus three orthogonal encodings.

    Phase: phi_r = pi/venc * (k_r . v), where k_r are unit axis vectors.

    Returns S_real, S_imag each of shape (N, 4).
    """
    # Encoding directions from Table 1 (Johnson & Markl 2010)
    # Point 0: flow-compensated reference (M1 = 0)
    # Points 1-3: orthogonal encodings along x, y, z
    k_vecs = torch.tensor([
        [ 0.0,  0.0,  0.0],   # r=0  flow-compensated reference
        [+1.0,  0.0,  0.0],   # r=1  x-encoding
        [ 0.0, +1.0,  0.0],   # r=2  y-encoding
        [ 0.0,  0.0, +1.0],   # r=3  z-encoding
    ], dtype=u.dtype, device=u.device)  # (4, 3)

    uvw = torch.stack([u, v, w], dim=-1)          # (N, 3)
    phi = (torch.pi / venc) * (uvw @ k_vecs.T)   # (N, 4)

    if m is None:
        mag = torch.ones(u.shape[0], 1, dtype=u.dtype, device=u.device)
    else:
        mag = m.unsqueeze(-1)                      # (N, 1)

    S_real = mag * torch.cos(phi)                  # (N, 4)
    S_imag = mag * torch.sin(phi)                  # (N, 4)

    return S_real, S_imag


def four_point_loss(uvw_data, uvw_pred, config, m_mr=None, m_nn=None):
    """
    Data-fidelity loss using four-point referenced encoding.

    Analogous to fathi_five_point_loss but with the weaker four-point
    referenced scheme. Each velocity component is encoded independently,
    so the alias minima are decoupled per direction:
        v_i^pred = v_i^orig + 2n*venc, n in Z
    This makes unwrapping harder — the physics regularization must work
    against independent periodic minima in x, y, z separately.

    Args:
        uvw_data : (N, 3+) measured velocities (normalised). First 3 cols = u,v,w.
        uvw_pred : (N, 3+) predicted velocities (normalised). First 3 cols = u,v,w.
        config   : training config. Must expose config.U_max, config.constants.venc.
        m_mr     : (N,) measured magnitude. If None, assumes m_MR = 1.
        m_nn     : (N,) predicted magnitude. If None, assumes m_NN = 1.
    """
    U_max = config.U_max
    venc  = config.constants.venc

    u_pred = uvw_pred[..., 0] * U_max
    v_pred = uvw_pred[..., 1] * U_max
    w_pred = uvw_pred[..., 2] * U_max

    u_data = uvw_data[..., 0] * U_max
    v_data = uvw_data[..., 1] * U_max
    w_data = uvw_data[..., 2] * U_max

    S_nn_real, S_nn_imag = four_point_encoding_cartesian(
        u_pred, v_pred, w_pred, venc, m=m_nn
    )
    S_mr_real, S_mr_imag = four_point_encoding_cartesian(
        u_data, v_data, w_data, venc, m=m_mr
    )

    diff_real = S_nn_real - S_mr_real   # (N, 4)
    diff_imag = S_nn_imag - S_mr_imag   # (N, 4)

    return torch.mean(diff_real ** 2 + diff_imag ** 2)


def fathi_five_point_loss(uvw_data, uvw_pred, config, m_mr=None, m_nn=None):
    """
    Fathi et al. (2020) data-fidelity loss using five-point balanced encoding.
 
    Both measured and predicted velocities are denormalized to physical units,
    converted to complex Cartesian images, and the MSE in Cartesian space is
    computed (Eq. 18, without Gaussian filter M, i.e. M = identity).
 
    Args:
        uvw_data : (N, 3+) measured velocities (normalised). First 3 cols = u,v,w.
        uvw_pred : (N, 3+) predicted velocities (normalised). First 3 cols = u,v,w.
        config   : training config.
        m_mr     : (N,) measured magnitude. If None, assumes m_MR = 1.
        m_nn     : (N,) predicted magnitude. If None, assumes m_NN = 1.
    """
    U_max = config.U_max
    venc  = config.constants.venc
 
    u_pred = uvw_pred[..., 0] * U_max
    v_pred = uvw_pred[..., 1] * U_max
    w_pred = uvw_pred[..., 2] * U_max
 
    u_data = uvw_data[..., 0] * U_max
    v_data = uvw_data[..., 1] * U_max
    w_data = uvw_data[..., 2] * U_max
 
    S_nn_real, S_nn_imag = five_point_encoding_cartesian(u_pred, v_pred, w_pred, venc, m=m_nn)
    S_mr_real, S_mr_imag = five_point_encoding_cartesian(u_data, v_data, w_data, venc, m=m_mr)
 
    diff_real = S_nn_real - S_mr_real   # (N, 5)
    diff_imag = S_nn_imag - S_mr_imag   # (N, 5)
 
    return torch.mean(diff_real ** 2 + diff_imag ** 2)


def data_loss_fn(model, xyz_data, uvw_data, mask, config):
    
    # Predict data points
    uvw_pred = model(xyz_data)

    if config.training.use_vector_potential:
        # Transform potential to velocities
        uvw_pred = vector_potential_fn(uvw_pred, xyz_data)

    # Calculate loss
    if config.training.use_mse:
        if config.setup.fluid_region:
            return mse_loss(uvw_pred, uvw_data, config)
        else:
            return fluid_weighted_mse_loss(uvw_pred, uvw_data, mask, config)    
    else:
        raise ValueError("No data loss specified, check config.training")

def physics_loss_fn(model, xyz_collocation, standardization_factors, config):

    # Predict collocation points
    uvw_pred = model(xyz_collocation)

    # Calculate loss
    if config.training.use_navier_stokes:
        momentum_loss, div_loss = navier_stokes_loss(uvw_pred, xyz_collocation, standardization_factors, config)
        physics_loss = momentum_loss + div_loss
    elif config.training.use_divergence:
        div_loss = divergence_loss(uvw_pred, xyz_collocation, standardization_factors, config)
        physics_loss = div_loss
    else:
        raise ValueError("No physics loss specified, check config.training")

    return config.training.physics_weight*physics_loss

def boundary_loss_fn(model, xyz_boundary, config):
    
    # Predict boundary points
    uvw_pred_boundary = model(xyz_boundary)
    
    # Calculate loss
    if config.training.use_boundary_mse:
        return boundary_mse_loss(uvw_pred_boundary, config)
    else:
        raise ValueError("No boundary loss specified, check config.training")

def compute_grad_norms(model, loss_fn, inputs, *args):
    
    # Reset gradients
    model.zero_grad()  

    # Compute loss
    loss = loss_fn(model, inputs, *args)
    loss.backward() 

    # Initialize list for gradients
    all_grads = []

    # Loop and collect gradients
    for param in model.parameters():
        if param.grad is not None:
            all_grads.append(param.grad.view(-1))

    # Concatenate gradients
    all_grads = torch.cat(all_grads)

    # Compute L2 norm
    grad_norm = all_grads.norm(p=2).item()

    return grad_norm

def update_loss_weights(config, model, loss_weights, it, xyz_data_batch, uvw_data_batch, mask, xyz_collocation_batch, xyz_boundary_batch, standardization_factors):
    
    # Initalize if loss_weights is None
    if loss_weights is None: 
        if config.sample_boundary:
            loss_weights = np.array([1.0, 1.0, 1.0])
        else:
            loss_weights = np.array([1.0, 1.0])
    
    # Return as is if data pre-training
    if it < config.training.epochs_before_PDE:
        return loss_weights
    
    # Compute gradient norms
    grad_norm_data = compute_grad_norms(model, data_loss_fn, xyz_data_batch, uvw_data_batch, mask, config)
    grad_norm_physics = compute_grad_norms(model, physics_loss_fn, xyz_collocation_batch, standardization_factors, config)
    if config.sample_boundary:
        grad_norm_boundary = compute_grad_norms(model, boundary_loss_fn, xyz_boundary_batch, config)

    # Collect gradient norms
    if config.sample_boundary:
        grad_norms = [grad_norm_data, grad_norm_physics, grad_norm_boundary]
    else:
        grad_norms = [grad_norm_data, grad_norm_physics]

    # Calculate and update weights
    weights_old = loss_weights
    weights_new = np.array([sum(grad_norms) / (grad_norm + 1e-7) for grad_norm in grad_norms])
    loss_weights = config.training.alpha*weights_old + (1-config.training.alpha)*weights_new

    return loss_weights

def compute_data_loss(config, model, xyz_data, uvw_data, mask):
    

    # if config.training.use_gaussian_quadrature:
    #     if config.voxel_size_norm is None:
    #         raise ValueError(
    #             "voxel_size_norm must be provided when use_gaussian_quadrature=True. "
    #             "Pass config.voxel_size_norm from the trainer."
    #         )
    #     n_quad = config.training.gaussian_quadrature_points
    #     uvw_pred = gaussian_quadrature_average(
    #         model, xyz_data, voxel_size_norm, n_quad, config
    #     )
    # else:

    # Predict data points
    uvw_pred = model(xyz_data)
    # Transform potential to velocity
    if config.training.use_vector_potential:
        uvw_pred = vector_potential_fn(uvw_pred, xyz_data)
    
        

    # m_nn = None
    # if config.training.use_magnitude_output:
    #     # Network outputs (u, v, w, p, m) — last column is magnitude,
    #     # plain linear output exactly as Fig. 3 of Fathi et al. (2020).
    #     m_nn     = uvw_pred[..., -1]    # (N,)  predicted magnitude
    #     uvw_pred = uvw_pred[..., :-1]   # (N, out_dim-1) = (N, 4)
 
    #     # m_MR is the binary fluid mask (1 = fluid, 0 = background)
    #     m_mr = mask.squeeze(-1)         # (N,)

    use_five = getattr(config.training, "use_fathi_five_point_loss", False)
    
    # Compute data loss
    if use_five:
        data_loss = fathi_five_point_loss(uvw_data, uvw_pred, config,m_mr=None,m_nn=None)
        # data_loss = four_point_loss(uvw_data, uvw_pred, config,m_mr=None,m_nn=None)

    elif config.training.use_mse:
        if config.setup.fluid_region:
            data_loss = mse_loss(uvw_pred, uvw_data, config)
        else:
            data_loss = fluid_weighted_mse_loss(uvw_pred, uvw_data, mask, config)
    
    elif config.training.use_cosine:
        if config.setup.fluid_region:
            data_loss = cosine_loss(uvw_pred, uvw_data, config)
        else:
            raise ValueError("No fluid_weighted_cosine_loss implemented yet.")
            #TODO
            # data_loss = fluid_weighted_cosine_loss(uvw_pred, uvw_data, mask, config)
    else:
        raise ValueError("No recognized data loss mode. Check config.training.use_mse or other flags.")

    return data_loss, uvw_pred

def compute_physics_loss(config, iter, model, xyz_collocation, xyz_data, standardization_factors):

    # Initialize dictionary for possible losses
    losses = {
        "physics_loss": torch.tensor(0.0),
        "momentum_loss": torch.tensor(0.0),
        "div_loss": torch.tensor(0.0),
        "physics_loss_data": torch.tensor(0.0),
        "momentum_loss_data": torch.tensor(0.0),
        "div_loss_data": torch.tensor(0.0),
    }

    # Skip if no physics loss
    if not config.training.use_physics_loss:
        return losses  # all remain None

    # Skip if PDE not active yet
    if iter < config.training.epochs_before_PDE:
        zero_t = torch.tensor(0.0)
        losses["physics_loss"] = zero_t
        losses["momentum_loss"] = zero_t
        losses["div_loss"] = zero_t
        losses["physics_loss_data"] = zero_t
        losses["momentum_loss_data"] = zero_t
        losses["div_loss_data"] = zero_t
        return losses

    # Predict collocation points
    uvw_pred_physics = model(xyz_collocation)
    if config.training.physics_loss_on_data_points:
        uvw_pred_data = model(xyz_data)

    # Navier-Stokes
    if config.training.use_navier_stokes:
        momentum_loss, div_loss = navier_stokes_loss(uvw_pred_physics, xyz_collocation, standardization_factors, config)
        physics_loss = momentum_loss + div_loss

        if config.training.physics_loss_on_data_points:
            momentum_loss_data, div_loss_data = navier_stokes_loss(uvw_pred_data, xyz_data, standardization_factors, config)
            physics_loss_data = momentum_loss_data + div_loss_data

        losses["physics_loss"] = physics_loss
        losses["momentum_loss"] = momentum_loss
        losses["div_loss"] = div_loss

        if config.training.physics_loss_on_data_points:
            losses["physics_loss_data"] = physics_loss_data
            losses["momentum_loss_data"] = momentum_loss_data
            losses["div_loss_data"] = div_loss_data

        return losses

    # Divergence only
    elif config.training.use_divergence:
        div_loss = divergence_loss(uvw_pred_physics, xyz_collocation, standardization_factors, config)
        if config.training.physics_loss_on_data_points:
            div_loss_data = divergence_loss(uvw_pred_data, xyz_data, standardization_factors, config)

        losses["physics_loss"] = div_loss
        losses["div_loss"] = div_loss

        if config.training.physics_loss_on_data_points:
            losses["physics_loss_data"] = div_loss_data
            losses["div_loss_data"] = div_loss_data
            
        return losses

    else:
        # Implement alternative loss here
        raise ValueError("No physics loss specified, check config.training")

def compute_boundary_loss(config, model, xyz_boundary):

    bound_loss = torch.tensor(0.0)

    # Skip if no boundary loss
    if not config.sample_boundary:
        return bound_loss  # all remain None
    
    # Predict boundary points
    uvw_pred = model(xyz_boundary)
    
    # Calculate loss
    if config.training.use_boundary_mse:
        return boundary_mse_loss(uvw_pred, config)
    else:
        # Implement alternative loss here
        raise ValueError("No boundary loss specified, check config.training")
    