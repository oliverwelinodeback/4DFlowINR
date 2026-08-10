import ml_collections
from datetime import datetime

def get_config(sweep_config=None):
    
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.sweep = False
    # Data
    config.data_file = "../data/Simulation_healthy/AoHealthy_noisy_4dsim_2mm_VENC150_n6.h5"
    config.include_ref = True
    config.data_file_ref = "../data/Simulation_healthy/AoHealthy_ref_2mm.h5"
    config.ref_spatial_factor = 1
    config.ref_temporal_factor = 1

    # Normalization and constants
    config.vel_normalization = "max_velocity" # max_velocity, characteristic
    config.coords_characteristic = False
    config.coords_normalization = "min_max" # min_max, standardize
    config.constants = ml_collections.ConfigDict()
    config.constants.U = 1.0
    config.constants.L = 0.025
    config.constants.T = 0.005
    config.constants.rho = 1060
    config.constants.mu = 0.004
    #######################################################
    config.constants.venc = 1.5  # CAMBIAR CUANDO SE CAMBIA DE VENC
    #######################################################
    # Model 
    networks_folder = "../models/260417_Aomodel_4dsim_V150_n6_FATHI-MRsignal_loss_MLP_300_MagOut"
    config.network_name = networks_folder.split("/")[-1] + '_FFN_1t'
    timestamp = datetime.now().strftime('%Y%m%d-%H%M')
    config.log_dir = f"{networks_folder}/{config.network_name}_{timestamp}"
    config.random_seed = 123

    # Domain
    config.domain = ml_collections.ConfigDict()
    config.domain.t_start = 0
    config.domain.t_end = 29
    config.domain.x_start = 0
    config.domain.x_end = 60
    config.domain.y_start = 0
    config.domain.y_end = 30
    config.domain.z_start = 0
    config.domain.z_end = 38
    config.global_normalization = True
    config.include_ref_loss = True

    # Resolution
    config.resolution = ml_collections.ConfigDict()
    config.resolution.from_file = True
    config.resolution.dx = 0.002
    config.resolution.dy = 0.002
    config.resolution.dz = 0.002
    config.resolution.dt = 0.01874

    # Setup / Options
    config.setup = ml_collections.ConfigDict()
    config.setup.include_pressure = False
    config.setup.include_time = True
    config.setup.fluid_region = False
    config.setup.expand_mask = False

    # Collocation & Boundary points sampling
    config.sample_collocation = True
    config.collocation_in_fluid = True
    config.collocation_points = 50_000
    config.sample_boundary = False
    config.boundary_repetitions = 1000

    # Network architecture
    config.network = ml_collections.ConfigDict()
    config.network.in_dim = 4
    config.network.out_dim = 4
    config.network.depth = 5
    config.network.hidden_features = 300
    config.network.arch = "FathiMLP" # FFN, FathiMLP, SIREN
    # SIREN parameters
    config.network.first_omega_0 = 30
    config.network.hidden_omega_0 = 30
    # Fourier Feature Encoding parameters
    config.network.fourier_mapping_size = 256
    config.network.fourier_scale = 1.0

    # Training parameters
    config.training = ml_collections.ConfigDict()
    config.training.iterations = 50_000 # 300000
    config.training.data_points_per_batch = 25000 # None to use all
    config.training.coll_points_per_batch = 25000 # None to use all
    config.training.boundary_points_per_batch = 5000 # None to use all
    config.training.alpha = 0.9
    # Optimizer
    config.training.lr = 1e-3
    config.training.lr_decay_iter = 2000
    config.training.lr_decay_factor = 0.9
    config.training.use_LBFGS = False
    config.training.BFGS_lr = 1e-1
    config.training.iterations_before_BFGS = 4000
    # Loss details
    config.training.epochs_before_PDE = 0
    config.training.grad_weight_scheme = False
    config.training.alpha = 0.9
    # Data loss options
    config.training.use_mse = False
    config.training.use_fathi_five_point_loss = True
    config.training.use_gaussian_quadrature = False     # Average network output over each voxel using Gauss-Legendre quadrature before computing data loss (Eq. 11-14). Requires use_fathi_five_point_loss=True.
    config.training.gaussian_quadrature_points = 4     # Number of 1-D GL points (1-5). Total evals per voxel = n^3. Paper used 4-point quadrature.
    config.training.use_magnitude_output = False       # Predict MR signal magnitude m as extra network output. Uses binary mask as m_MR. Requires fluid_region=False.

    config.training.use_cosine = False
    config.training.use_vector_potential = False
    config.training.pressure_in_data_loss = False
    config.training.u_weight = 1.0
    config.training.v_weight = 1.0
    config.training.w_weight = 1.0
    config.training.p_weight = 0.01
    # Physics loss options
    config.training.use_physics_loss = True
    config.training.physics_loss_on_data_points = False
    config.training.use_navier_stokes = True
    config.training.use_divergence = True
    config.training.physics_weight = 0.1
    # Boundary loss options
    config.training.pressure_in_boundary_loss = False
    config.training.use_boundary_mse = False
    config.training.boundary_weight = 0.1
    # Logging and performance evaluation
    config.training.summary_iter = 2500
    config.training.log_iter = 50
    config.training.error_iter = 1000
    config.training.denormalize = True
    # Plotting
    config.plot = ml_collections.ConfigDict()
    config.plot.iter = 5000
    config.plot.gt = True
    config.plot.t_step = 3
    config.plot.z_slice = 26
    config.plot.spatial_factor = 1
    config.plot.temporal_factor = 1
    config.plot.temp_upsampling_mode = 'extend'
    config.plot.spat_upsampling_mode = 'centered'
    config.plot.fluid_region = False
    config.plot.non_fluid_value = 0
    config.plot.expand_mask = False
    config.plot.denormalize = True

    # Prediction
    config.predictions = ml_collections.ConfigDict()
    config.predictions.peak_flow_idx = 0
    config.predictions.predict_reference_data = True
    config.predictions.predict_SR_data = True
    config.predictions.compare_noisy_vs_ref = True
    config.predictions.denormalize = True
    config.predictions.fluid_region = True

    return config