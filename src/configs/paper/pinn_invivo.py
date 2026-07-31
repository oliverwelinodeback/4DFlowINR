import ml_collections
from datetime import datetime

def get_sweep_config():
    
    """Sweep configuration for wandb."""
    timestamp = datetime.now().strftime('%Y%m%d-%H%M')
    return {
        'name': "paper-pinn-invivo", 
        'method': 'grid',
        'metric': {
            'name': 'Final/Relative error [Fluid]', 
            'goal': 'minimize'
        },
        'parameters': {
            'data_file': {'values': [
                            "../data/invivo/HV01.h5", 
                            "../data/invivo/HV03.h5", 
                            "../data/invivo/HV06.h5", 
                            "../data/invivo/ICAD28.h5", 
                            "../data/invivo/ICAD48.h5", 
                            "../data/invivo/ICAD98.h5", 
                            "../data/invivo/ICAD17.h5", 
                            "../data/invivo/ICAD21.h5",
                            "../data/invivo/ICAD146.h5",    
                        ]
            },
        },
    }

def get_config(sweep_config=None):
    
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.sweep = True
    # Data
    config.data_file = "../data/XXX.h5"
    config.include_ref = False
    config.include_ref_loss = False
    config.load_pressure_from_data = False
    config.data_file_ref = "../data/XXX.h5"
    config.ref_spatial_factor = 2
    config.ref_temporal_factor = 2

    # Model 
    config.networks_folder = "../models/paper_pinn-invivo/"
    config.network_name = "paper_pinn-invivo"
    timestamp = datetime.now().strftime('%Y%m%d-%H%M')
    config.log_dir = f"{config.networks_folder}/{config.network_name}_{timestamp}"
    config.random_seed = 1234

    # Weights & Biases
    config.wandb = ml_collections.ConfigDict()
    config.wandb.project = "4DFlowINR"
    config.wandb.group = "paper-pinn-invivo"
    config.wandb.tags = ["paper", "pinn", "invivo"]

    # Domain
    config.domain = ml_collections.ConfigDict()
    config.domain.crop = False
    config.domain.t_start = None
    config.domain.t_end = None
    config.domain.x_start = None
    config.domain.x_end = None
    config.domain.y_start = None
    config.domain.y_end = None
    config.domain.z_start = None
    config.domain.z_end = None

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
    config.collocation_points = 1_500_000
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
    config.template.t_len = 100

    config.constants = ml_collections.ConfigDict()
    config.constants.U = 2.0
    config.constants.L = 0.005
    config.constants.T = config.constants.L / config.constants.U
    config.constants.rho = 1060
    config.constants.mu = 0.004
    config.constants.venc = 1.2

    # Network architecture
    config.network = ml_collections.ConfigDict()
    config.network.in_dim = 4
    config.network.out_dim = 4
    config.network.depth = 6
    config.network.hidden_features = 128
    config.network.arch = "WIRE"
    # WIRE parameters
    config.network.sigma_0 = 30
    config.network.omega_0 = 60
    config.network.complex = False

    # Meta-learning initialization
    config.meta_learning = ml_collections.ConfigDict()
    config.meta_learning.enabled = False
    config.load_meta_init = False
    config.meta_init_path =  "../models/paper_pinn-meta/paper_pinn-meta_20260731-1425/meta_best.pth"

    # Training parameters
    config.training = ml_collections.ConfigDict()
    config.training.iterations = 15000
    config.training.data_points_per_batch = 6000
    config.training.coll_points_per_batch = 6000
    config.training.boundary_points_per_batch = 6000
    # Optimizer
    config.training.lr = 1e-4
    config.training.lr_decay_iter = 25000
    config.training.lr_decay_factor = 0.5
    config.decay_type = "none"  
    config.training.use_LBFGS = True
    config.training.BFGS_lr = 5e-2
    config.training.iterations_before_BFGS = 10000
    config.training.BFGS_max_iter = 3
    config.training.BFGS_history_size = 50
    config.training.BFGS_tolerance_grad = 1e-7
    config.training.BFGS_tolerance_change = 1e-6
    # Loss details
    config.training.epochs_before_PDE = 0
    config.training.grad_weight_scheme = True
    config.training.alpha = 0.95

    # Self-adaptive
    config.training.self_adaptive = False
    config.training.adaptive_sampling = False
    config.training.tau = 0.02
    config.training.weight_clip = [6, 0.2]
    config.training.beta = 0.2
    config.training.K_initial = 10_000
    config.training.K = 20
    config.training.points_to_update = 750_000
    config.training.chunk_size = 2_000

    # Data loss options
    config.training.use_mse = False
    config.training.use_cosine = True
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
    config.training.use_divergence = True
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
    config.training.summary_iter = 5000
    config.training.log_iter = 250
    config.training.error_iter = 5000
    config.training.denormalize = True

    # Visualization
    config.visualization = ml_collections.ConfigDict()
    config.visualization.enabled = True
    config.visualization.time_index_lr = 1
    config.visualization.z_index_lr = 20

    # Prediction
    config.predictions = ml_collections.ConfigDict()
    # Evaluation indices
    config.predictions.peak_flow_idx = 6
    # Arbitrary-grid INR prediction
    config.predictions.spatial_factor = 2
    config.predictions.temporal_factor = 2
    config.predictions.temporal_upsampling_mode = "extend"
    config.predictions.spatial_upsampling_mode = "centered"
    config.predictions.fluid_region = True
    config.predictions.non_fluid_value = 0
    config.predictions.expand_mask = False
    config.predictions.denormalize = True

    return config