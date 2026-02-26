import ml_collections
from datetime import datetime


def get_sweep_config_quick():
    """
    Quick test sweep for MAML + Physics with Curriculum Learning.

    Uses your best data-driven MAML hyperparameters as baseline,
    only sweeps physics-related parameters to validate the approach.

    Run this first to check if curriculum learning helps!
    """
    return {
        'name': 'Meta_MAML_Physics_QuickTest',
        'method': 'grid',
        'metric': {'name': 'Val/Post_LR', 'goal': 'minimize'},
        'parameters': {
            # Use best hyperparameters from data-driven MAML
            'meta_learning.inner_lr': {'values': [0.000611]},  # Best from data-driven
            'meta_learning.outer_lr': {'values': [1.51e-05]},  # Best from data-driven
            'meta_learning.inner_steps': {'values': [10]},     # Best from data-driven
            'meta_learning.meta_batch_size': {'values': [3]},  # Slightly lower for physics memory

            # Sweep physics parameters
            'meta_learning.physics_weight': {'values': [0.1, 0.5, 1.0]},
            'meta_learning.coll_points_outer': {'values': [1000, 2000]},

            # Curriculum: compare with vs without
            'meta_learning.physics_curriculum_start': {'values': [0, 500]},  # 0 = no curriculum
            'meta_learning.physics_curriculum_end': {'values': [1, 2000]},   # 1 = no curriculum (instant full weight)
        }
    }


""" def get_sweep_config():
    return {
        'name': 'Meta_MAML_PhysicsOuterOnly_Curriculum',
        'method': 'bayes',
        'metric': {'name': 'Val/Post_LR', 'goal': 'minimize'},
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 1000,  # Increased to allow curriculum to take effect
            'max_iter': 5000
        },
        'parameters': {
            'meta_learning.inner_lr': {
                'distribution': 'log_uniform_values',
                'min': 1e-4,
                'max': 1e-2
            },
            'meta_learning.outer_lr': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 1e-4
            },
            'meta_learning.inner_steps': {
                'values': [5, 7, 10]  # Can use more steps (no physics memory in inner)
            },
            'meta_learning.meta_batch_size': {
                'values': [2, 3]  # Can use larger batch than full PINN MAML
            },
            'meta_learning.physics_weight': {
                'distribution': 'log_uniform_values',
                'min': 0.01,  # Lower minimum - physics should be gentle regularization
                'max': 1.0   # Lower max - avoid physics dominating
            },
            'meta_learning.coll_points_outer': {
                'values': [1000, 2000, 3000]
            },
            # Curriculum parameters
            'meta_learning.physics_curriculum_start': {
                'values': [300, 500, 750]  # When to start adding physics
            },
            'meta_learning.physics_curriculum_end': {
                'values': [1500, 2000, 3000]  # When physics reaches full weight
            },
        }
    } """
def get_sweep_config():
    
    timestamp = datetime.now().strftime('%Y%m%d-%H%M')
    return {
        'name': f'MetaLearn_PINN_1000it_h5_{timestamp}', 
        'method': 'grid',
        'metric': {'name': 'FINAL Relative Error [Fluid]', 'goal': 'minimize'},
        'parameters': {

            'data_file': {'values': [

                #"../data/healthy/HV01_05mm3_20ms_LR_sv17_tSNR10_newMask.h5", 
                #"../data/healthy/HV03_05mm3_20ms_LR_sv13_tSNR10_newMask.h5", 
                "../data/healthy/HV06_05mm3_20ms_LR_sv12_tSNR10_newMask.h5", 
                #"../data/stenosis_50/ICAD28_05mm3_20ms_LR_sv13_tSNR10_newMask.h5", 
                #"../data/stenosis_50/ICAD48_05mm3_20ms_LR_sv13_tSNR10_newMask.h5", 
                ##"../data/stenosis_50/ICAD98_05mm3_20ms_LR_sv51_tSNR10_newMask.h5", 
                #"../data/stenosis_70/ICAD17_05mm3_20ms_LR_sv41_tSNR10_newMask.h5", 
                #"../data/stenosis_70/ICAD21_05mm3_20ms_LR_sv26_tSNR10_newMask.h5",
                ##"../data/stenosis_70/ICAD146_05mm3_20ms_LR_sv17_tSNR10_newMask.h5",    
                ]},
            
            'load_meta_init': {'values': [True, False]},
        },
    }

def get_config(sweep_config=None):
    """
    MAML (Second-Order) with Physics in OUTER LOOP ONLY

    This is the memory-efficient physics-informed MAML configuration:
    - Inner loop: Optimizes DATA LOSS only (velocities u, v, w)
    - Outer loop: Optimizes DATA + PHYSICS loss (Navier-Stokes)

    The model outputs 4 components (u, v, w, p):
    - Data loss: Uses (u, v, w) - ground truth velocities
    - Physics loss: Uses all 4 - Navier-Stokes needs pressure gradients

    Pressure (p) is a LATENT VARIABLE learned through physics constraints.

    Key settings:
    - meta_method: 'MAML' (full second-order)
    - use_first_order: False
    - use_physics_loss: True
    - use_physics_outer_only: True (MEMORY-EFFICIENT MODE)
    """
    config = ml_collections.ConfigDict()

    config.sweep = True
    # Data
    config.data_file = "../data/XXX.h5"
    config.include_ref = True
    config.include_ref_loss = True
    config.load_pressure_from_data = True
    config.data_file_ref = "../data/XXX.h5"
    config.ref_spatial_factor = 2
    config.ref_temporal_factor = 2

    # Model
    config.networks_folder = "../models/MetaLearn_PINN_1000it_h5/"
    config.network_name = "260203_MAML_PhysicsOuterOnly"
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
    config.coords_characteristic = True  # IMPORTANT: Must match PINN fine-tuning config!
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
    config.network.out_dim = 4  # u, v, w, p (pressure needed for Navier-Stokes)
    config.network.depth = 6
    config.network.hidden_features = 128
    config.network.arch = "WIRE"
    # WIRE parameters
    config.network.sigma_0 = 30
    config.network.omega_0 = 60
    config.network.complex = False

    # ==========================================
    # META-LEARNING CONFIGURATION (MAML + Physics Outer Only)
    # ==========================================
    config.meta_learning = ml_collections.ConfigDict()
    config.meta_learning.enabled = False

    # Meta-learning method: FULL SECOND-ORDER MAML
    config.meta_learning.meta_method = 'MAML'
    config.meta_learning.use_first_order = False  # Second-order gradients

    # Reptile-specific (not used)
    config.meta_learning.reptile_epsilon = 1.0

    # Inner loop settings (SGD optimizer)
    # Can use MORE steps than full PINN MAML (no physics memory in inner loop)
    config.meta_learning.inner_lr = 0.000611
    config.meta_learning.inner_steps = 10  # Higher than full PINN (data-only inner)
    config.meta_learning.inner_points = 5000
    config.meta_learning.coll_points_inner = 3000  # Not used (physics_outer_only)
    config.meta_learning.boundary_points_inner = 2000

    # Outer loop settings (Adam optimizer)
    config.meta_learning.outer_lr = 1.51e-05
    config.meta_learning.meta_batch_size = 3  # Can be higher than full PINN MAML
    config.meta_learning.max_iters = 10000

    # ==========================================
    # PHYSICS OUTER ONLY MODE (KEY SETTINGS)
    # ==========================================
    config.meta_learning.use_physics_loss = True        # Enable physics
    config.meta_learning.use_physics_outer_only = True  # OUTER LOOP ONLY!

    # Physics settings (applied ONLY in outer loop)
    config.meta_learning.physics_weight = 1.0           # Sweep this!
    config.meta_learning.coll_points_outer = 2000       # Collocation for outer loop
    config.meta_learning.use_boundary_loss = False
    config.meta_learning.div_weight = 1.0

    # ==========================================
    # CURRICULUM LEARNING FOR PHYSICS
    # ==========================================
    # Gradually introduce physics to avoid conflicting gradients early in training
    # Phase 1 (0 to curriculum_start): Pure data-driven (like working MAML)
    # Phase 2 (curriculum_start to curriculum_end): Physics weight ramps up linearly
    # Phase 3 (after curriculum_end): Full physics_weight applied
    config.meta_learning.physics_curriculum_start = 0   # Start adding physics at iter 500
    config.meta_learning.physics_curriculum_end = 0    # Full physics by iter 2000

    # Other settings
    config.meta_learning.support_fraction = 0.5
    config.meta_learning.use_grad_weights = False

    # Scheduler
    config.meta_learning.use_scheduler = False
    config.meta_learning.scheduler_gamma = 0.9995

    # Physical Pre-Conditioning (optional)
    config.meta_learning.use_physics_preconditioning = False
    config.meta_learning.preconditioning_iters = 100
    config.meta_learning.preconditioning_cases = 5
    config.meta_learning.preconditioning_lr = 1e-3

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

    # Fine-tuning (disabled for meta-learning sweep)
    config.load_meta_init = True
    config.meta_init_path = "../models/MetaLearning_MAML_PhysicsOuterOnly/260203_MAML_PhysicsOuterOnly_20260205-0755/meta_best.pth"

    # Training parameters (for fine-tuning after meta-learning)
    config.training = ml_collections.ConfigDict()
    config.training.iterations = 1000
    config.training.data_points_per_batch = 6000
    config.training.coll_points_per_batch = 6000
    config.training.boundary_points_per_batch = 60000
    # Optimizer
    config.training.lr = 1e-4
    config.training.lr_decay_iter = 25000
    config.training.lr_decay_factor = 0.5
    config.training.use_LBFGS = True
    config.training.BFGS_lr = 5e-2
    config.training.iterations_before_BFGS = 1000
    config.training.BFGS_max_iter = 3
    config.training.BFGS_history_size = 50
    config.training.BFGS_tolerance_grad = 1e-7
    config.training.BFGS_tolerance_change = 1e-6
    # Scheduler
    config.decay_type = 'none'
    # Loss details
    config.training.epochs_before_PDE = 0
    config.training.grad_weight_scheme = True
    config.training.alpha = 0.95

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
    # Physics loss options (for fine-tuning)
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
    config.training.error_iter = 500
    config.training.save_h5_iters = [10, 25, 50, 100, 250, 500, 1000]  # Iterations at which to save h5 predictions
    config.training.denormalize = True
    # Plotting
    config.plot = ml_collections.ConfigDict()
    config.plot.iter = 1000
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
