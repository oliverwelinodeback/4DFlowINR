import ml_collections
from datetime import datetime

def get_config(sweep_config=None):
    
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.sweep = False
    # Data
    config.data_file = "../../data/KI_simulated_CFD/HV01_pg_05mm3_dv_highSNR_x1.h5"
    config.include_ref = True
    config.include_ref_loss = True
    config.data_file_ref = "../../data/KI_simulated_CFD/HV01_pg_05mm3.h5"
    #config.data_file_ref = "../../data/icad_sim/HV01_05mm.h5"
    config.ref_spatial_factor = 1
    config.ref_temporal_factor = 1

    # Model 
    networks_folder = "../models/251008_WIRE/"
    config.network_name = "HV01_WIRE_MSE_SIREN_Rep3" 
    timestamp = datetime.now().strftime('%Y%m%d-%H%M')
    config.log_dir = f"{networks_folder}/{config.network_name}_{timestamp}"
    config.random_seed = 123

    # Domain
    config.domain = ml_collections.ConfigDict()
    config.domain.t_start = 0
    config.domain.t_end = 33
    config.domain.x_start = 34
    config.domain.x_end = 48
    config.domain.y_start = 48
    config.domain.y_end = 60
    config.domain.z_start = 6
    config.domain.z_end = 16
    #config.domain.t_start = 14
    #config.domain.t_end =   int(120/2)
    #config.domain.x_start = int(0/2)
    #config.domain.x_end =   int(158/2)
    #config.domain.y_start = int(0/2)
    #config.domain.y_end =   int(126/2)
    #config.domain.z_start = int(0)
    #config.domain.z_end =   int(80/2)

    # Resolution
    config.resolution = ml_collections.ConfigDict()
    config.resolution.from_file = False
    config.resolution.dx = 0.0005
    config.resolution.dy = 0.0005
    config.resolution.dz = 0.0005
    config.resolution.dt = 0.01

    # Setup / Options
    config.setup = ml_collections.ConfigDict()
    config.setup.include_pressure = False
    config.setup.include_time = False
    config.setup.fluid_region = True
    config.setup.expand_mask = False

    # Collocation & Boundary points sampling
    config.sample_collocation = False
    config.collocation_in_fluid = True
    config.collocation_points = 500_000
    config.sample_boundary = False
    config.boundary_repetitions = 1000

    # Normalization and constants
    config.vel_normalization = "characteristic"
    config.coords_characteristic = False
    config.coords_normalization = "standardize" 
    config.global_normalization = True
    
    config.use_baseline_normalization = True
    config.template = ml_collections.ConfigDict()
    config.template.dx = 0.001 # m
    config.template.dy = config.template.dx
    config.template.dz = config.template.dx
    config.template.dt = 0.1 # s
    config.template.x_len = 200
    config.template.y_len = config.template.x_len
    config.template.z_len = 50
    config.template.t_len = 10*10

    config.constants = ml_collections.ConfigDict()
    config.constants.U = 1.0
    config.constants.L = 0.005
    config.constants.T = config.constants.L / config.constants.U
    config.constants.rho = 1060
    config.constants.mu = 0.004
    config.constants.venc = 1.2

    # Network architecture
    config.network = ml_collections.ConfigDict()
    config.network.in_dim = 3
    config.network.out_dim = 3
    config.network.depth = 6
    config.network.hidden_features = 128
    config.network.arch = "WIRE"
    # SIREN parameters
    config.network.omega_0 = 0 #17
    # Fourier Feature Encoding parameters
    config.network.fourier_mapping_size = 128
    config.network.fourier_scale = 2.5
    # WIRE parameters
    config.network.omega_0 = 0 #17
    config.network.sigma_0 = 10

    # Training parameters
    config.training = ml_collections.ConfigDict()
    config.training.iterations = 2000
    config.training.data_points_per_batch = None # None to use all
    config.training.coll_points_per_batch = 20000 # None to use all
    config.training.boundary_points_per_batch = 10000 # None to use all
    # Optimizer
    config.training.lr = 1e-4
    config.training.lr_decay_iter = 50000
    config.training.lr_decay_factor = 0.5
    config.training.use_LBFGS = False
    config.training.BFGS_lr = 1e-1
    config.training.iterations_before_BFGS = 1000
    # Loss details
    config.training.epochs_before_PDE = 0
    config.training.grad_weight_scheme = False
    config.training.alpha = 0.95
    # Data loss options
    config.training.use_mse = True
    config.training.use_cosine = False # TODO - name loss instead of True False options?
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
    config.training.use_PPE = False
    config.training.PPE_weight = 0.001
    config.training.predict_gradients = False
    config.training.reference_gradients = False
    config.training.physics_weight = 1
    # Boundary loss options
    config.training.pressure_in_boundary_loss = False
    config.training.use_boundary_mse = True
    config.training.boundary_weight = 1.0
    # Logging and performance evaluation
    config.training.summary_iter = 100
    config.training.log_iter = 10
    config.training.error_iter = 100
    config.training.denormalize = True
    # Plotting
    config.plot = ml_collections.ConfigDict()
    config.plot.iter = 100
    config.plot.gt = True
    config.plot.t_step = 0
    config.plot.z_slice = 2
    config.plot.spatial_factor = 2
    config.plot.temporal_factor = 2
    config.plot.temp_upsampling_mode = 'extend'
    config.plot.spat_upsampling_mode = 'centered'
    config.plot.fluid_region = True
    config.plot.non_fluid_value = 0
    config.plot.expand_mask = False
    config.plot.denormalize = True

    # Prediction
    config.predictions = ml_collections.ConfigDict()
    config.predictions.peak_flow_idx = 0
    config.predictions.predict_reference_data = True
    config.predictions.predict_SR_data = False
    config.predictions.compare_noisy_vs_ref = False
    config.predictions.denormalize = True
    config.predictions.fluid_region = True

    return config