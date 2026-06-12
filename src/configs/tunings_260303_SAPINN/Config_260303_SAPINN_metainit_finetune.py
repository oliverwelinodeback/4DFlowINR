import ml_collections
from datetime import datetime


def get_sweep_config():
    """
    SA-PINN + Meta-Init Ablation Sweep - Phase 3b

    Ablation: 3 unseen patients × meta-init vs random-init × meta physics_weight vs 1.0
    = 12 runs total

    Hypothesis: meta-init + meta-informed physics_weight converges faster and to lower
    error than any other combination.

    Prerequisites:
    1. Run Config_MetaLearning_MAML_DataDriven_ForPINN.py (Phase 3a)
    2. Read final loss_weights from W&B logs:
         loss_weights/data  -> the data EMA weight (= grad_norm_physics / grad_norm_data)
         loss_weights/physics -> the physics EMA weight
    3. Set meta_physics_weight = loss_weights_physics / loss_weights_data
    4. Fill META_PHYSICS_WEIGHT below and update 'training.physics_weight' values

    SA params: Fill in best values found from Phase 2 sweep before running.
    """
    timestamp = datetime.now().strftime('%Y%m%d-%H%M')

    # !! FILL THIS IN after Phase 3a: read from W&B -> final loss_weights/physics / loss_weights/data
    META_PHYSICS_WEIGHT = 1.0  # placeholder - replace with value from Phase 3a

    return {
        'name': f'SAPINN_MetaInit_Ablation_{timestamp}',
        'method': 'grid',
        'metric': {'name': 'FINAL Relative Error [Fluid]', 'goal': 'minimize'},
        'parameters': {
            # 3 unseen validation patients (not seen during meta-training)
            'data_file': {'values': [
                "../data/healthy/HV06_05mm3_20ms_LR_sv12_tSNR10_newMask.h5",
                "../data/stenosis_50/ICAD98_05mm3_20ms_LR_sv51_tSNR10_newMask.h5",
                "../data/stenosis_70/ICAD146_05mm3_20ms_LR_sv17_tSNR10_newMask.h5",
            ]},
            # Meta-init vs random-init
            'load_meta_init': {'values': [True, False]},
            # Meta-informed physics weight vs default
            'training.physics_weight': {'values': [1.0, META_PHYSICS_WEIGHT]},
        },
    }


def get_config(sweep_config=None):
    """
    SA-PINN + Meta-Init Fine-Tuning - Phase 3

    Uses the best SA hyperparameter values found in Phase 2 and tests whether
    initialising from a meta-learned checkpoint AND using the meta-learned
    physics/data weight ratio improves SA-PINN performance on unseen patients.

    Key concept:
    - Phase 3a produces a meta_best.pth from MAML meta-learning across 6 patients
    - The grad_weight_scheme converges to a physics/data loss balance across patients
    - That converged ratio (physics_weight) is a better starting point than 1.0
    - SA-PINN then fine-tunes with adaptive collocation from this informed starting state
    - K_initial is reduced (1000) because meta-init is already near convergence

    Template: Config_MetaLearning_MAML_PhysicsOuterOnly.py

    !! BEFORE RUNNING:
    1. Complete Phase 2 sweep and fill best SA params below
    2. Complete Phase 3a meta-learning run
    3. Fill META_INIT_PATH with the actual meta_best.pth path
    4. Update META_PHYSICS_WEIGHT and the sweep config values above
    """
    config = ml_collections.ConfigDict()

    config.sweep = True

    # ==========================================
    # DATA
    # ==========================================
    config.data_file = "../data/XXX.h5"  # Overridden by sweep
    config.include_ref = True
    config.include_ref_loss = True
    config.load_pressure_from_data = True
    config.data_file_ref = "../data/XXX.h5"  # Overridden by trainer data routing
    config.ref_spatial_factor = 2
    config.ref_temporal_factor = 2

    # ==========================================
    # MODEL
    # ==========================================
    config.networks_folder = "../models/260303_SAPINN_metainit/"
    config.network_name = "260303_SAPINN_metainit_finetune"
    timestamp = datetime.now().strftime('%Y%m%d-%H%M')
    config.log_dir = f"{config.networks_folder}/{config.network_name}_{timestamp}"
    config.random_seed = 1234

    # ==========================================
    # DOMAIN
    # ==========================================
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

    # ==========================================
    # RESOLUTION
    # ==========================================
    config.resolution = ml_collections.ConfigDict()
    config.resolution.from_file = False
    config.resolution.dx = 0.0005 * 2
    config.resolution.dy = 0.0005 * 2
    config.resolution.dz = 0.0005 * 2
    config.resolution.dt = 0.02 * 2

    # ==========================================
    # SETUP / OPTIONS
    # ==========================================
    config.setup = ml_collections.ConfigDict()
    config.setup.include_pressure = True
    config.setup.include_time = True
    config.setup.fluid_region = True
    config.setup.expand_mask = False

    # ==========================================
    # COLLOCATION & BOUNDARY POINTS
    # ==========================================
    config.sample_collocation = True
    config.collocation_in_fluid = True
    config.collocation_points = 1_500_000
    config.sample_boundary = False
    config.boundary_repetitions = 1000

    # ==========================================
    # NORMALIZATION AND CONSTANTS
    # coords_characteristic must match the meta-learning config (True)
    # ==========================================
    config.vel_normalization = "characteristic"
    config.coords_characteristic = True  # MUST match Config_MetaLearning_MAML_PhysicsOuterOnly
    config.coords_normalization = "standardize"
    config.global_normalization = True

    config.use_baseline_normalization = True
    config.template = ml_collections.ConfigDict()
    config.template.dx = 0.001  # m
    config.template.dy = config.template.dx
    config.template.dz = config.template.dx
    config.template.dt = 0.1  # s
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
    config.constants.venc = 1.2  # Default; overridden by trainer data routing

    # ==========================================
    # NETWORK ARCHITECTURE
    # Must match meta-learning config exactly for checkpoint loading
    # ==========================================
    config.network = ml_collections.ConfigDict()
    config.network.in_dim = 4
    config.network.out_dim = 4         # [u, v, w, p] - must match meta-learning config
    config.network.depth = 6
    config.network.hidden_features = 128
    config.network.arch = "WIRE"
    config.network.sigma_0 = 30
    config.network.omega_0 = 60        # Must match meta-learning config
    config.network.complex = False

    # ==========================================
    # META-LEARNING (disabled for fine-tuning)
    # ==========================================
    config.meta_learning = ml_collections.ConfigDict()
    config.meta_learning.enabled = False

    config.meta_learning.meta_method = 'MAML'
    config.meta_learning.reptile_epsilon = 1.0
    config.meta_learning.inner_lr = 0.000611
    config.meta_learning.inner_steps = 10
    config.meta_learning.inner_points = 5000
    config.meta_learning.coll_points_inner = 3000
    config.meta_learning.boundary_points_inner = 2000
    config.meta_learning.outer_lr = 1.51e-05
    config.meta_learning.meta_batch_size = 3
    config.meta_learning.max_iters = 10000
    config.meta_learning.use_physics_loss = True
    config.meta_learning.use_physics_outer_only = True
    config.meta_learning.physics_weight = 1.0
    config.meta_learning.coll_points_outer = 2000
    config.meta_learning.use_boundary_loss = False
    config.meta_learning.div_weight = 1.0
    config.meta_learning.physics_curriculum_start = 0
    config.meta_learning.physics_curriculum_end = 0
    config.meta_learning.support_fraction = 0.5
    config.meta_learning.use_scheduler = False
    config.meta_learning.scheduler_gamma = 0.9995
    config.meta_learning.train_cases = []
    config.meta_learning.val_cases = []
    config.meta_learning.case_venc = {}

    # ==========================================
    # META-INIT
    # Set load_meta_init=True to use checkpoint; swept as [True, False] for ablation.
    # Fill META_INIT_PATH after Phase 3a completes.
    # ==========================================
    config.load_meta_init = True
    # !! FILL THIS IN after Phase 3a run completes:
    config.meta_init_path = "../models/MetaLearning_MAML_PhysicsOuterOnly/<RUN_NAME>/meta_best.pth"

    # ==========================================
    # TRAINING PARAMETERS
    # ==========================================
    config.training = ml_collections.ConfigDict()
    config.training.iterations = 40_000
    config.training.data_points_per_batch = 6000
    config.training.coll_points_per_batch = 6000
    config.training.boundary_points_per_batch = 60000

    # Optimizer
    config.training.lr = 1e-4
    config.training.lr_decay_iter = 25000
    config.training.lr_decay_factor = 0.5
    config.training.use_LBFGS = False  # SA + LBFGS interaction untested
    config.training.BFGS_lr = 5e-2
    config.training.iterations_before_BFGS = 40000
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

    # ==========================================
    # SELF-ADAPTIVE PINN SETTINGS
    # !! Fill best values from Phase 2 sweep before running Phase 3
    # K_initial reduced because meta-init is already near convergence
    # ==========================================
    config.training.self_adaptive = True
    config.training.adaptive_sampling = True
    config.training.tau = 0.01            # !! Replace with Phase 2 best value
    config.training.weight_clip = [10, 0.05]  # !! Replace with Phase 2 best [max, min]
    config.training.beta = 0.3            # !! Replace with Phase 2 best value
    config.training.K_initial = 1_000    # Reduced: meta-init is already near convergence
    config.training.K = 250              # !! Replace with Phase 2 best value
    config.training.points_to_update = 750_000
    config.training.chunk_size = 5_000

    # ==========================================
    # DATA LOSS OPTIONS
    # ==========================================
    config.training.use_mse = False
    config.training.use_cosine = True
    config.training.use_vector_potential = False
    config.training.pressure_in_data_loss = False
    config.training.u_weight = 1.0
    config.training.v_weight = 1.0
    config.training.w_weight = 1.0
    config.training.p_weight = 0.01

    # ==========================================
    # PHYSICS LOSS OPTIONS
    # physics_weight is swept as [1.0, meta_value] via trainer.py readback
    # !! Fill meta_value in get_sweep_config() after Phase 3a
    # ==========================================
    config.training.use_physics_loss = True
    config.training.physics_loss_on_data_points = True
    config.training.use_navier_stokes = True
    config.training.use_divergence = True
    config.training.use_PPE = False
    config.training.PPE_weight = 0.001
    config.training.predict_gradients = False
    config.training.reference_gradients = False
    config.training.physics_weight = 1  # Swept: [1.0, meta_physics_weight]

    # ==========================================
    # BOUNDARY LOSS OPTIONS
    # ==========================================
    config.training.pressure_in_boundary_loss = False
    config.training.use_boundary_mse = True
    config.training.boundary_weight = 1.0

    # ==========================================
    # LOGGING AND EVALUATION
    # ==========================================
    config.training.summary_iter = 5000
    config.training.log_iter = 250
    config.training.error_iter = 500
    config.training.save_h5_iters = [10, 25, 50, 100, 250, 500, 1000]
    config.training.denormalize = True

    # ==========================================
    # PLOTTING
    # ==========================================
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

    # ==========================================
    # PREDICTIONS
    # ==========================================
    config.predictions = ml_collections.ConfigDict()
    config.predictions.peak_flow_idx = 6
    config.predictions.flow_idx2 = 6
    config.predictions.predict_reference_data = True
    config.predictions.predict_SR_data = False
    config.predictions.compare_noisy_vs_ref = False
    config.predictions.denormalize = True
    config.predictions.fluid_region = True

    return config
