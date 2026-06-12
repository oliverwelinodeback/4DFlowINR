import ml_collections
from datetime import datetime


def get_sweep_config():
    """Dummy sweep config — this config runs a single meta-learning job (sweep=False)."""
    return {
        'name': 'unused_SAPINN_maml',
        'method': 'grid',
        'metric': {'name': 'FINAL Relative Error [Fluid]', 'goal': 'minimize'},
        'parameters': {},
    }


def get_config(sweep_config=None):
    """
    MAML (second-order) with Physics in Outer Loop Only — for out_dim=6 model.

    Produces meta-weights compatible with Config_260303_SAPINN_lbfgs_sweep.py.
    The key difference from Config_MetaLearning_MAML_PhysicsOuterOnly.py:
      - out_dim=6: network directly predicts [u, v, w, dp/dx, dp/dy, dp/dz]
      - predict_gradients=True: navier_stokes_loss uses predicted gradients directly
      - omega_0=30: matches repro_SA / lbfgs_sweep architecture
      - use_divergence=False: matches lbfgs_sweep

    The meta loop handles out_dim=6 correctly without code changes:
      - Inner loop data loss uses only channels [0,1,2] (u,v,w)
      - navier_stokes_loss branches on config.training.predict_gradients

    After meta-training, use the saved meta_best.pth as init for fine-tuning by
    setting load_meta_init=True and meta_init_path in the fine-tune config.

    MAML settings (identical to gold standard):
      inner_lr=0.000611, inner_steps=10, outer_lr=1.51e-05, meta_batch_size=3
      use_physics_outer_only=True, physics_weight=1.0, coll_points_outer=2000
    """
    config = ml_collections.ConfigDict()

    config.sweep = False

    # ==========================================
    # DATA  (HV06 — validation patient, used for fine-tune eval after meta)
    # ==========================================
    config.data_file = "../data/healthy/HV06_05mm3_20ms_LR_sv12_tSNR10_newMask.h5"
    config.include_ref = True
    config.include_ref_loss = True
    config.load_pressure_from_data = True
    config.data_file_ref = "../data/healthy/HV06_05mm3_20ms.h5"
    config.ref_spatial_factor = 2
    config.ref_temporal_factor = 2

    # ==========================================
    # MODEL
    # ==========================================
    config.networks_folder = "../models/260303_SAPINN_maml/"
    config.network_name = "260303_MAML_SAPINN_out6"
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
    # IMPORTANT: Must match lbfgs_sweep fine-tuning config exactly.
    # ==========================================
    config.vel_normalization = "characteristic"
    config.coords_characteristic = True
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
    config.constants.venc = 1.2  # HV06 default; overridden per case by case_venc

    # ==========================================
    # NETWORK ARCHITECTURE
    # out_dim=6 and omega_0=30 must match lbfgs_sweep config.
    # ==========================================
    config.network = ml_collections.ConfigDict()
    config.network.in_dim = 4
    config.network.out_dim = 6          # [u, v, w, dp/dx, dp/dy, dp/dz]
    config.network.depth = 6
    config.network.hidden_features = 128
    config.network.arch = "WIRE"
    config.network.sigma_0 = 30
    config.network.omega_0 = 30         # Must match lbfgs_sweep (repro_SA setting)
    config.network.complex = False

    # ==========================================
    # META-LEARNING CONFIGURATION
    # Identical to Config_MetaLearning_MAML_PhysicsOuterOnly.py (gold standard)
    # ==========================================
    config.meta_learning = ml_collections.ConfigDict()
    config.meta_learning.enabled = True  # Set to True by trainer when meta_learning runs

    config.meta_learning.meta_method = 'MAML'
    config.meta_learning.reptile_epsilon = 1.0

    # Inner loop (data loss only — memory-efficient)
    config.meta_learning.inner_lr = 0.000611
    config.meta_learning.inner_steps = 10
    config.meta_learning.inner_points = 5000
    config.meta_learning.coll_points_inner = 3000
    config.meta_learning.boundary_points_inner = 2000

    # Outer loop
    config.meta_learning.outer_lr = 1.51e-05
    config.meta_learning.meta_batch_size = 3
    config.meta_learning.max_iters = 10000

    # Physics outer only mode
    config.meta_learning.use_physics_loss = True
    config.meta_learning.use_physics_outer_only = True
    config.meta_learning.physics_weight = 1.0
    config.meta_learning.coll_points_outer = 2000
    config.meta_learning.use_boundary_loss = False
    config.meta_learning.div_weight = 1.0

    # Curriculum learning (disabled — matches gold standard)
    config.meta_learning.physics_curriculum_start = 0
    config.meta_learning.physics_curriculum_end = 0

    config.meta_learning.support_fraction = 0.5
    config.meta_learning.use_scheduler = False
    config.meta_learning.scheduler_gamma = 0.9995

    # Training cases (6 patients — same as gold standard)
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
        "../data/stenosis_70/ICAD146_05mm3_20ms_LR_sv17_tSNR10_newMask.h5",
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
        "ICAD146_05mm3_20ms_LR_sv17_tSNR10_newMask": 1.7,
    }

    # ==========================================
    # META-INIT (disabled — training from scratch)
    # After meta-training, use the saved meta_best.pth as init for fine-tuning
    # by pointing load_meta_init=True + meta_init_path in the fine-tune config.
    # ==========================================
    config.load_meta_init = False
    config.meta_init_path = ""

    # ==========================================
    # TRAINING PARAMETERS (for fine-tuning phase after meta-learning)
    # Matches lbfgs_sweep settings so the fine-tune is compatible.
    # ==========================================
    config.training = ml_collections.ConfigDict()
    config.training.iterations = 40_000
    config.training.data_points_per_batch = 10000
    config.training.coll_points_per_batch = 10000
    config.training.boundary_points_per_batch = 10000

    # Optimizer
    config.training.lr = 1e-4
    config.training.lr_decay_iter = 33_000   # placeholder; overridden by trainer if use_LBFGS
    config.training.lr_decay_factor = 0.5
    config.training.use_LBFGS = True
    config.training.BFGS_lr = 5e-2
    config.training.iterations_before_BFGS = 35_000
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

    # Self-adaptive (disabled for meta-learning run)
    config.training.self_adaptive = False
    config.training.adaptive_sampling = False
    config.training.tau = 0.02
    config.training.weight_clip = [6, 0.2]
    config.training.beta = 0.2
    config.training.K_initial = 10_000
    config.training.K = 20
    config.training.points_to_update = 750_000
    config.training.chunk_size = 10_000

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
    # predict_gradients=True: navier_stokes_loss uses network-predicted gradients.
    # Must match out_dim=6.
    # ==========================================
    config.training.use_physics_loss = True
    config.training.physics_loss_on_data_points = True
    config.training.use_navier_stokes = True
    config.training.use_divergence = False
    config.training.use_PPE = False
    config.training.PPE_weight = 0.001
    config.training.predict_gradients = True   # Must be True for out_dim=6
    config.training.reference_gradients = True
    config.training.physics_weight = 1

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
    config.training.error_iter = 5000
    config.training.save_h5_iters = []
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
    config.predictions.peak_flow_idx = 2   # HV06
    config.predictions.flow_idx2 = 2
    config.predictions.predict_reference_data = True
    config.predictions.predict_SR_data = False
    config.predictions.compare_noisy_vs_ref = False
    config.predictions.denormalize = True
    config.predictions.fluid_region = True

    return config
