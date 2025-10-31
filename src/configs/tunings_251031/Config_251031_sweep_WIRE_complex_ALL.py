import ml_collections
from datetime import datetime

def get_sweep_config():
    """Sweep configuration for wandb."""
    
    timestamp = datetime.now().strftime('%Y%m%d-%H%M')
    return {
        'name': f'WIRE-COMPLEX_ALL_{timestamp}', 
        'method': 'grid',
        'metric': {'name': 'FINAL Relative Error [Fluid]', 'goal': 'minimize'},
        'parameters': {
            'constants.U': {'values': [1.0]},

            'data_file': {'values': [

                #"../data/healthy/HV01_05mm3_20ms_LR_dv_hv17_tSNR8.h5", 
                #"../data/healthy/HV03_05mm3_20ms_LR_dv_hv13_tSNR8.h5", 
                #"../data/healthy/HV06_05mm3_20ms_LR_dv_hv12_tSNR8.h5", 
                #"../data/stenosis_50/ICAD28_05mm3_20ms_LR_dv_hv13_tSNR8.h5", 
                #"../data/stenosis_50/ICAD48_05mm3_20ms_LR_dv_hv13_tSNR8.h5", 
                "../data/stenosis_50/ICAD98_05mm3_20ms_LR_dv_hv51_tSNR8.h5", 
                "../data/stenosis_70/ICAD17_05mm3_20ms_LR_dv_hv41_tSNR8.h5", 
                "../data/stenosis_70/ICAD21_05mm3_20ms_LR_dv_hv26_tSNR8.h5",
                "../data/stenosis_70/ICAD146_05mm3_20ms_LR_dv_hv17_tSNR8.h5",
                
                ]},

        },
    }


def get_config():
    
    """Get the default hyperparameter configuration."""
    
    config = ml_collections.ConfigDict()
    config.sweep = True 

    # Data
    config.data_file = "../data/XXX.h5"
    config.include_ref = True
    config.include_ref_loss = True
    config.data_file_ref = "../data/XXX.h5"
    config.ref_spatial_factor = 2
    config.ref_temporal_factor = 2

    # Model 
    config.networks_folder = "../models/251031_WIRE_CMPLX_ALL/"
    config.network_name = "251031_WIRE_CMPLX_ALL" 
    timestamp = datetime.now().strftime('%Y%m%d-%H%M')
    config.log_dir = f"{config.networks_folder}/{config.network_name}_{timestamp}"
    config.random_seed = 123

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
    config.setup.include_pressure = False
    config.setup.include_time = True
    config.setup.fluid_region = True
    config.setup.expand_mask = False

    # Collocation & Boundary points sampling
    config.sample_collocation = False
    config.collocation_in_fluid = True
    config.collocation_points = 1_500_000
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
    config.template.t_len = 500

    config.constants = ml_collections.ConfigDict()
    config.constants.U = 1.0
    config.constants.L = 0.005
    config.constants.T = config.constants.L / config.constants.U
    config.constants.rho = 1060
    config.constants.mu = 0.004
    config.constants.venc = 1.3

    # Network architecture
    config.network = ml_collections.ConfigDict()
    config.network.in_dim = 4
    config.network.out_dim = 3
    config.network.depth = 6
    config.network.hidden_features = 64
    config.network.arch = "WIRE"
    
    # SIREN parameters
    #config.network.omega_0 = 0 #17

    # Fourier Feature Encoding parameters
    #config.network.fourier_mapping_size = 128
    #config.network.fourier_scale = 2.5
    
    # WIRE parameters
    config.network.complex = True
    config.network.omega_0 = 30
    config.network.sigma_0 = 30

    # Training parameters
    config.training = ml_collections.ConfigDict()
    config.training.iterations = 8000
    config.training.data_points_per_batch = 20000 # None to use all
    config.training.coll_points_per_batch = 20000 # None to use all
    config.training.boundary_points_per_batch = 10000 # None to use all
    # Optimizer
    config.training.lr = 1e-4
    config.training.lr_decay_iter = 9999
    config.training.lr_decay_factor = 0.5
    config.training.use_LBFGS = False
    config.training.BFGS_lr = 1e-1
    config.training.iterations_before_BFGS = 9999
    # Loss details
    config.training.epochs_before_PDE = 0
    config.training.grad_weight_scheme = False
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
    config.training.use_physics_loss = config.sample_collocation
    config.training.physics_loss_on_data_points = config.sample_collocation
    config.training.use_navier_stokes = False
    config.training.use_divergence = config.sample_collocation
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
    config.training.summary_iter = 4000
    config.training.log_iter = 250 #!#
    config.training.error_iter = 4000
    config.training.denormalize = True
    # Plotting
    config.plot = ml_collections.ConfigDict()
    config.plot.iter = 4000
    config.plot.gt = True
    config.plot.t_step = 4
    config.plot.t_step_2 = 6
    config.plot.z_slice = 10
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
    config.predictions.flow_idx2 = 2
    config.predictions.predict_reference_data = True
    config.predictions.predict_SR_data = False
    config.predictions.compare_noisy_vs_ref = False
    config.predictions.denormalize = True
    config.predictions.fluid_region = True

    return config