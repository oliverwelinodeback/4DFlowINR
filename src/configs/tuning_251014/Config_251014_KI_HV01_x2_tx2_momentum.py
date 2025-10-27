import ml_collections
from datetime import datetime

def get_sweep_config():
    """Sweep configuration for wandb."""
    
    timestamp = datetime.now().strftime('%Y%m%d-%H%M')
    return {
        'name': f'NS_Tunings_sweep_{timestamp}', 
        'method': 'grid',
        'metric': {'name': "FINAL Pressure Gradient Relative Error [Fluid]", 'goal': 'minimize'},
        'parameters': {
            'template.t_len': {'values': [100, 5000]},
            'network.arch': {'values': ['WIRE', 'SIREN']},
            'network.hidden_features': {'values': [64, 128]},
            'training.BFGS_lr': {'values': [1, 1e-1, 1e-2]},
            'training.iterations_before_BFGS': {'values': [999999, 12500, 7500]},
            'training.lr_decay_iter': {'values': [99999999, 4000]},
        },
    }

def get_config(sweep_config=None):
    
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.sweep = True
    # Data
    config.data_file = "../data/healthy/HV01_05mm3_20ms_LR_dv_tSNR8.h5"
    config.include_ref = True
    config.include_ref_loss = True
    config.data_file_ref = "../data/healthy/HV01_05mm3_20ms.h5"
    config.ref_spatial_factor = 2
    config.ref_temporal_factor = 2

    # Model 
    config.networks_folder = "../models/251014_NS_tuning/"
    config.networks_folder = "../models/251014_NS_tuning/"
    config.network_name = "HV01_WIRE_momentum_REP" 
    timestamp = datetime.now().strftime('%Y%m%d-%H%M')
    config.log_dir = f"{config.networks_folder}/{config.network_name}_{timestamp}"
    config.random_seed = 1234

    # Domain
    config.domain = ml_collections.ConfigDict()
    config.domain.t_start = 0
    config.domain.t_end = -(-25 // 2) # Ceiling division
    config.domain.x_start = int(0/2)
    config.domain.x_end = int(140/2)
    config.domain.y_start = int(0/2)
    config.domain.y_end = int(92/2)
    config.domain.z_start = int(0/2)
    config.domain.z_end = int(52/2)
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
    config.resolution.dx = 0.0005*2
    config.resolution.dy = 0.0005*2
    config.resolution.dz = 0.0005*2
    config.resolution.dt = 0.02*2

    # Setup / Options
    config.setup = ml_collections.ConfigDict()
    config.setup.include_pressure = True
    config.setup.include_time = True
    config.setup.fluid_region = True
    config.setup.expand_mask = False

    # Collocation & Boundary points sampling
    config.sample_collocation = True
    config.collocation_in_fluid = True
    config.collocation_points = 1_000_000
    config.sample_boundary = False
    config.boundary_repetitions = 1000

    # Normalization and constants
    config.vel_normalization = "characteristic"
    config.coords_characteristic = True
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
    config.template.t_len = 100 #!#

    config.constants = ml_collections.ConfigDict()
    config.constants.U = 1.0
    config.constants.L = 0.005
    config.constants.T = config.constants.L / config.constants.U
    config.constants.rho = 1060
    config.constants.mu = 0.004
    config.constants.venc = 1.2

    # Network architecture
    config.network = ml_collections.ConfigDict()
    config.network.in_dim = 4
    config.network.out_dim = 6
    config.network.depth = 6
    config.network.hidden_features = 64
    config.network.arch = "WIRE"
    # SIREN parameters
    #config.network.omega_0 = 30 #17
    ## Fourier Feature Encoding parameters
    #config.network.fourier_mapping_size = 128
    #config.network.fourier_scale = 1.0
    # WIRE parameters
    config.network.sigma_0 = 30
    config.network.omega_0 = 30 #17
    config.network.complex = False

    # Training parameters
    config.training = ml_collections.ConfigDict()
    config.training.iterations = 15000 #!#
    config.training.data_points_per_batch = 20000 # None to use all #20000
    config.training.coll_points_per_batch = 20000 # None to use all #20000
    config.training.boundary_points_per_batch = 10000 # None to use all #10000
    # Optimizer
    config.training.lr = 1e-4
    config.training.lr_decay_iter = 25000
    config.training.lr_decay_factor = 0.5
    config.training.use_LBFGS = True
    config.training.BFGS_lr = 1e-1
    config.training.iterations_before_BFGS = 12500 #!#
    # Loss details
    config.training.epochs_before_PDE = 0
    config.training.grad_weight_scheme = True
    config.training.alpha = 0.95
    # Data loss options
    config.training.use_mse = False
    config.training.use_cosine = True # TODO - name loss instead of True False options?
    config.training.use_vector_potential = False
    config.training.pressure_in_data_loss = False
    config.training.u_weight = 1.0
    config.training.v_weight = 1.0
    config.training.w_weight = 1.0
    config.training.p_weight = 0.01
    # Physics loss options
    config.training.use_physics_loss = True
    config.training.physics_loss_on_data_points = True
    config.training.use_navier_stokes = True
    config.training.use_divergence = False
    config.training.use_PPE = False
    config.training.PPE_weight = 0.001
    config.training.predict_gradients = True
    config.training.reference_gradients = True
    config.training.physics_weight = 1
    # Boundary loss options
    config.training.pressure_in_boundary_loss = False
    config.training.use_boundary_mse = True
    config.training.boundary_weight = 1.0
    # Logging and performance evaluation
    config.training.summary_iter = 5000
    config.training.log_iter = 150
    config.training.error_iter = 5000
    config.training.denormalize = True
    # Plotting
    config.plot = ml_collections.ConfigDict()
    config.plot.iter = 5000
    config.plot.gt = True
    config.plot.t_step = 2
    config.plot.t_step_2 = 13
    config.plot.z_slice = 20
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
    config.predictions.peak_flow_idx = 12
    config.predictions.flow_idx2 = 20
    config.predictions.predict_reference_data = True
    config.predictions.predict_SR_data = False
    config.predictions.compare_noisy_vs_ref = False
    config.predictions.denormalize = True
    config.predictions.fluid_region = True

    return config