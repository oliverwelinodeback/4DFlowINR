import ml_collections
from datetime import datetime

def get_config():
    
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # Data
    config.data_file = "../../data/data_05mm_incl_pressure/healthy-05mm3_t24_26_250120_TSNR12_x1.h5"
    config.include_ref = True
    config.data_file_ref = "../../data/data_05mm_incl_pressure/healthy-05mm3.h5"
    config.ref_spatial_factor = 1
    config.ref_temporal_factor = 1

    # Model 
    networks_folder = "../models/250129_Testing"
    config.network_name = "SIREN_1t"
    timestamp = datetime.now().strftime('%Y%m%d-%H%M')
    config.log_dir = f"{networks_folder}/{config.network_name}_{timestamp}"
    config.random_seed = 123

    # Domain
    config.domain = ml_collections.ConfigDict()
    config.domain.t_start = 25
    config.domain.t_end = 126
    config.domain.x_start = 120
    config.domain.x_end = 201
    config.domain.y_start = 0
    config.domain.y_end = 57
    config.domain.z_start = 19
    config.domain.z_end = 69

    # Resolution
    config.resolution = ml_collections.ConfigDict()
    config.resolution.dx = 0.0005
    config.resolution.dy = 0.0005
    config.resolution.dz = 0.0005
    config.resolution.dt = 0.01

    # Setup / Options
    config.setup = ml_collections.ConfigDict()
    config.setup.include_pressure = False
    config.setup.include_time = False
    config.setup.fluid_region = True
    config.setup.expand_mask = True

    # Collocation & Boundary points sampling
    config.sample_collocation = True
    config.collocation_in_fluid = True
    config.collocation_points = 1500000
    config.sample_boundary = True
    config.boundary_repetitions = 1000

    # Normalization and constants
    config.vel_normalization = "characteristic" # max_velocity
    config.coords_characteristic = True 
    config.coords_normalization = "standardize" # min_max
    config.constants = ml_collections.ConfigDict()
    config.constants.U = 1.0
    config.constants.L = 0.005
    config.constants.T = 0.005
    config.constants.rho = 1060
    config.constants.mu = 0.004

    # Network architecture
    config.network = ml_collections.ConfigDict()
    config.network.in_dim = 3
    config.network.out_dim = 3
    config.network.depth = 6
    config.network.hidden_features = 128
    config.network.first_omega_0 = 30
    config.network.hidden_omega_0 = 30

    # Training parameters
    config.training = ml_collections.ConfigDict()
    config.training.iterations = 1000 # 300000
    config.training.data_points_per_batch = None # None to use all
    config.training.coll_points_per_batch = 20000 # None to use all
    config.training.boundary_points_per_batch = None # None to use all
    config.training.alpha = 0.9
    # Optimizer
    config.training.lr = 1e-4
    config.training.lr_decay_iter = 50000
    config.training.lr_decay_factor = 0.5
    config.training.use_LBFGS = False
    config.training.BFGS_lr = 1e-1
    config.training.iterations_before_BFGS = 600
    # Loss details
    config.training.epochs_before_PDE = 0
    config.training.grad_weight_scheme = False
    config.training.alpha = 0.9
    # Data loss options
    config.training.use_mse = True
    config.training.use_vector_potential = False
    config.training.pressure_in_data_loss = False
    config.training.u_weight = 1.0
    config.training.v_weight = 1.0
    config.training.w_weight = 1.0
    config.training.p_weight = 0.01
    # Physics loss options
    config.training.use_physics_loss = False
    config.training.physics_loss_on_data_points = False
    config.training.use_navier_stokes = False
    config.training.use_divergence = False
    config.training.physics_weight = 1.0
    # Boundary loss options
    config.training.pressure_in_boundary_loss = False
    config.training.use_boundary_mse = True
    config.training.boundary_weight = 1.0
    # Logging and performance evaluation
    config.training.summary_iter = 200
    config.training.log_iter = 1
    config.training.error_iter = 200
    config.training.denormalize = True
    # Plotting
    config.plot = ml_collections.ConfigDict()
    config.plot.iter = 200
    config.plot.gt = True
    config.plot.t_step = None
    config.plot.z_slice = 20
    config.plot.spatial_factor = 2
    config.plot.temporal_factor = 2
    config.plot.temp_upsampling_mode = 'extend'
    config.plot.spat_upsampling_mode = 'centered'
    config.plot.fluid_region = True
    config.plot.non_fluid_value = 0
    config.plot.expand_mask = True
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