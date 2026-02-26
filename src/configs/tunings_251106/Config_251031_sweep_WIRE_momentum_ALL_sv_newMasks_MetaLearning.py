import ml_collections
from datetime import datetime

def get_sweep_config():
    
    timestamp = datetime.now().strftime('%Y%m%d-%H%M')
    return {
        'name': f'WIRE_SV_MetaLearn_loss1000it_{timestamp}', 
        'method': 'grid',
        'metric': {'name': 'FINAL Relative Error [Fluid]', 'goal': 'minimize'},
        'parameters': {

            'data_file': {'values': [

                #"../data/healthy/HV01_05mm3_20ms_LR_sv17_tSNR10_newMask.h5", 
                #"../data/healthy/HV03_05mm3_20ms_LR_sv13_tSNR10_newMask.h5", 
                "../data/healthy/HV06_05mm3_20ms_LR_sv12_tSNR10_newMask.h5", 
                #"../data/stenosis_50/ICAD28_05mm3_20ms_LR_sv13_tSNR10_newMask.h5", 
                #"../data/stenosis_50/ICAD48_05mm3_20ms_LR_sv13_tSNR10_newMask.h5", 
                "../data/stenosis_50/ICAD98_05mm3_20ms_LR_sv51_tSNR10_newMask.h5", 
                #"../data/stenosis_70/ICAD17_05mm3_20ms_LR_sv41_tSNR10_newMask.h5", 
                #"../data/stenosis_70/ICAD21_05mm3_20ms_LR_sv26_tSNR10_newMask.h5",
                "../data/stenosis_70/ICAD146_05mm3_20ms_LR_sv17_tSNR10_newMask.h5",    
                ]},
            
            'load_meta_init': {'values': [True, False]},
            #'training.iterations': {'values': [250, 500]},
            #'training.iterations': {'values': [500, 1000, 2500, 5000]},
            #'network.sigma_0': {'values': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]},
            #'network.omega_0': {'values': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
            #'training.lr': {'values': [ 2e-4, 1e-4]},
            #'decay_type': {'values': ['none','multi', 'exp', 'cosine']},
            #'decay_target': {'values': [0.1, 0.01, 0.001]}
            
            #'network.sigma_0': {'values': [0, 25, 50, 75, 100]},
            #'network.omega_0': {'values': [0, 25, 50, 75, 100]}
            #'training.iterations_before_BFGS': {
            #    'values': [5000, 12500]
            #},
            #'training.BFGS_lr': {
            #    'values': [0.1, 0.05, 0.01]
            #},
            #'training.BFGS_max_iter': {
            #    'values': [3, 5, 10, 20]
            #},
            #'training.BFGS_history_size': {
            #    'values': [10, 20, 50]
            #},
            #'training.BFGS_tolerance_grad': {
            #    'values': [1e-7, 1e-6, 1e-5]
            #},
            #'training.BFGS_tolerance_change': {
            #    'values': [1e-9, 1e-7, 1e-6]
            #},
            



        },
    }
""" def get_sweep_config():
    return {
        'name': 'Meta_SR_Hyperparameter_Tuning_Scheduler',
        'method': 'bayes',  # Bayesian optimization is faster than grid for finding "sweet spots"
        'metric': {
            'name': 'Val/Post_LR', 
            'goal': 'minimize'
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 1500,  # Minimum iterations before stopping (30% of max)
            's': 2,            # Controls aggressiveness (higher = more aggressive)
            'eta': 3,          # Proportion of runs to keep at each stage
            'max_iter': 5000   # Maximum iterations
        },
        'parameters': {
            # ===== CRITICAL: Inner Loop Parameters =====
            # Inner LR: How fast to adapt to each task
            # Too high: overfits to support set
            # Too low: doesn't adapt enough
            'meta_learning.inner_lr': {
                'distribution': 'log_uniform_values',
                'min': 5e-4,   # 0.0005
                'max': 5e-2    # 0.05
            },

            # Inner Steps: Depth of adaptation
            # More steps = better adaptation, but harder to meta-learn (gradient issues)
            'meta_learning.inner_steps': {
                'values': [3, 5, 10]
            },

            # ===== CRITICAL: Outer Loop Parameters =====
            # Outer LR: Meta-update learning rate
            # Too high: unstable meta-learning
            # Too low: slow convergence
            'meta_learning.outer_lr': {
                'distribution': 'log_uniform_values',
                'min': 5e-5,   # 0.00005
                'max': 5e-4    # 0.0005
            },

            # Meta Batch Size: Tasks per meta-update
            # Larger = more stable, but slower and higher memory
            'meta_learning.meta_batch_size': {
                'values': [2, 3, 4]
            }
            
        }
    } """

def get_config(sweep_config=None):
    
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.sweep = True
    # Data
    config.data_file = "../data/XXX.h5"
    config.include_ref = True
    config.include_ref_loss = True
    config.load_pressure_from_data = False
    config.data_file_ref = "../data/XXX.h5"
    config.ref_spatial_factor = 2
    config.ref_temporal_factor = 2

    # Model 
    config.networks_folder = "../models/260123_WIRE_MetaLearn_loss1000iter/"
    config.network_name = "260123_WIRE" 
    timestamp = datetime.now().strftime('%Y%m%d-%H%M')
    config.log_dir = f"{config.networks_folder}/{config.network_name}_{timestamp}"
    config.random_seed = 1234

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
    config.sample_collocation = True
    config.collocation_in_fluid = True
    config.collocation_points = 1_500_000
    config.sample_boundary = False #!#
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
    config.template.t_len = 100 #!#

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
    config.network.out_dim = 3 #!#
    config.network.depth = 6
    config.network.hidden_features = 128
    config.network.arch = "WIRE"
    # SIREN parameters
    #config.network.omega_0 = 30 #17
    ## Fourier Feature Encoding parameters
    #config.network.fourier_mapping_size = 128
    #config.network.fourier_scale = 1.0
    # WIRE parameters
    config.network.sigma_0 = 20
    config.network.omega_0 = 20 #17
    config.network.complex = False

    # meta-learning
    config.meta_learning = ml_collections.ConfigDict()
    config.meta_learning.enabled = False
    config.meta_learning.version = 'v2'  # 'v1' (wrong LR->HR) or 'v2' (correct LR->LR)
    config.meta_learning.method = 'MAML'
    config.meta_learning.inner_lr = 0.0010618534857988096
    config.meta_learning.inner_steps = 5
    config.meta_learning.outer_lr = 1e-4
    config.meta_learning.meta_batch_size = 4
    config.meta_learning.max_iters = 5000

    
    config.meta_learning.use_scheduler = True
    config.meta_learning.scheduler_gamma = 0.9991

    # V2-specific parameters
    config.meta_learning.support_fraction = 0.5    # Split LR: 50% support, 50% query
    config.meta_learning.use_physics_loss = False  # Enable Navier-Stokes constraints
    config.meta_learning.physics_weight = 0.1      # Weight for physics loss
    config.meta_learning.coll_points_inner = 1000  # Collocation points per inner step
    config.meta_learning.inner_points = 5000       # Max data points per inner step

    config.meta_learning.train_cases = [
        "../data/healthy/HV01_05mm3_20ms_LR_sv17_tSNR10_newMask.h5",
        "../data/healthy/HV03_05mm3_20ms_LR_sv13_tSNR10_newMask.h5",
        
        "../data/stenosis_50/ICAD28_05mm3_20ms_LR_sv13_tSNR10_newMask.h5",
        "../data/stenosis_50/ICAD48_05mm3_20ms_LR_sv13_tSNR10_newMask.h5",
        
        "../data/stenosis_70/ICAD17_05mm3_20ms_LR_sv41_tSNR10_newMask.h5",
        "../data/stenosis_70/ICAD21_05mm3_20ms_LR_sv26_tSNR10_newMask.h5",
    ]

    config.meta_learning.val_cases = [
        "../data/healthy/HV06_05mm3_20ms_LR_sv12_tSNR10_newMask.h5",
        "../data/stenosis_50/ICAD98_05mm3_20ms_LR_sv51_tSNR10_newMask.h5",
        "../data/stenosis_70/ICAD146_05mm3_20ms_LR_sv17_tSNR10_newMask.h5"
    ]
    
    config.meta_learning.case_venc = {
        "HV01_05mm3_20ms_LR_sv17_tSNR10_newMask": 1.7,
        "HV03_05mm3_20ms_LR_sv13_tSNR10_newMask": 1.3,
        "HV06_05mm3_20ms_LR_sv12_tSNR10_newMask": 1.2,
        "ICAD28_05mm3_20ms_LR_sv13_tSNR10_newMask": 1.3,
        "ICAD48_05mm3_20ms_LR_sv13_tSNR10_newMask": 1.3,
        "ICAD98_05mm3_20ms_LR_sv51_tSNR10_newMask": 5.1,
        "ICAD17_05mm3_20ms_LR_sv41_tSNR10_newMask": 4.1,
        "ICAD21_05mm3_20ms_LR_sv26_tSNR10_newMask": 2.6,
        "ICAD146_05mm3_20ms_LR_sv17_tSNR10_newMask": 1.7
    }

    # Fine-tuning
    config.load_meta_init = False
    config.meta_init_path = "../models/MetaLearningSweep/260115_WIRE_SV_NewMask_MetaLearn_innerLR0.0009159876127281078_innerSteps10_outerLR0.00011430108092301548_metaBatch2_20260115-1444/meta_learned_init_FINAL.pth"

    # Update save paths
    #config.networks_folder = "../models/MetaLearningSweep/"
    #config.network_name = "260115_MetaLearn_WIRE_sweep"
    #timestamp = datetime.now().strftime('%Y%m%d-%H%M')
    #config.log_dir = f"{config.networks_folder}/{config.network_name}_{timestamp}"
    

    # Training parameters
    config.training = ml_collections.ConfigDict()
    config.training.iterations = 1000 #8000 #!#
    config.training.data_points_per_batch = 20000 # None to use all #20000
    config.training.coll_points_per_batch = 20000 # None to use all #20000
    config.training.boundary_points_per_batch = 10000 # None to use all #10000
    # Optimizer
    config.training.lr = 1e-4
    config.training.lr_decay_iter = 25000
    config.training.lr_decay_factor = 0.5
    config.training.use_LBFGS = False
    config.training.BFGS_lr = 5e-2
    config.training.iterations_before_BFGS = 10000 #!#
    config.training.BFGS_max_iter = 3
    config.training.BFGS_history_size = 50
    config.training.BFGS_tolerance_grad = 1e-7
    config.training.BFGS_tolerance_change = 1e-6
    # Scheduler
    config.decay_type = 'exp' #'multi' #'exp' #'cosine' #
    # Loss details
    config.training.epochs_before_PDE = 0
    config.training.grad_weight_scheme = False
    config.training.alpha = 0.95 #!# 0.90

    config.training.self_adaptive = False
    config.training.adaptive_sampling = False
    config.training.tau = 0.02 # 0.03
    config.training.weight_clip = [6, 0.2] # [5, 0.2]
    config.training.beta = 0.2 # 0.2
    config.training.K_initial = 10_000 #10_000 # 500
    config.training.K = 20 # 250
    config.training.points_to_update = 750_000 # 500_000
    config.training.chunk_size = 2_000

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
    config.training.summary_iter = 5000
    config.training.log_iter = 20 #50 #!#
    config.training.error_iter = 100 #5000
    config.training.denormalize = True
    # Plotting
    config.plot = ml_collections.ConfigDict()
    config.plot.iter = 5000
    config.plot.gt = True
    config.plot.t_step = 2
    config.plot.t_step_2 = 6
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
    config.predictions.peak_flow_idx = 6
    config.predictions.flow_idx2 = 6
    config.predictions.predict_reference_data = True
    config.predictions.predict_SR_data = False
    config.predictions.compare_noisy_vs_ref = False
    config.predictions.denormalize = True
    config.predictions.fluid_region = True

    return config