import ml_collections
from datetime import datetime


def get_sweep_config():
    """Dummy sweep config — not used (sweep=False)."""
    return {
        'name': 'unused_repro_HRLR',
        'method': 'grid',
        'metric': {'name': 'FINAL Relative Error [Fluid]', 'goal': 'minimize'},
        'parameters': {}
    }


def get_config(sweep_config=None):
    """
    Reproduction of original SA-PINN config — Run C (HRLR input, SA enabled).

    Patient : HV01 (healthy, venc=1.7 m/s)
    Purpose : Identical to Config_260303_SAPINN_repro_SA.py except the input
              data uses HRLR (same spatial resolution as the HR reference).
              Tests whether a same-resolution input improves pressure gradient
              recovery compared to the standard 2× downsampled LR input.

    HRLR file: HV01_05mm3_20ms_HRLR_sv17_tSNR10.h5
      — same spatial grid as HR (0.5 mm voxels), but with MRI k-space noise.

    Changes vs repro_SA:
      data_file → HRLR file
      resolution.dx/dy/dz = 0.0005   (×1, not ×2)
      resolution.dt = 0.02            (×1, not ×2)
      ref_spatial_factor = 1          (no upsampling needed)
      ref_temporal_factor = 1
      plot.spatial_factor = 1
      plot.temporal_factor = 1
      network_name = "260303_repro_HRLR_HV01"
    """
    config = ml_collections.ConfigDict()

    config.sweep = False

    # ==========================================
    # DATA  (HRLR: same spatial resolution as reference)
    # ==========================================
    config.data_file = "../data/healthy/HV01_05mm3_20ms_HRLR_sv17_tSNR10.h5"
    config.include_ref = True
    config.include_ref_loss = True
    config.load_pressure_from_data = True
    config.data_file_ref = "../data/healthy/HV01_05mm3_20ms.h5"
    config.ref_spatial_factor = 1   # HRLR is already at reference resolution
    config.ref_temporal_factor = 1

    # ==========================================
    # MODEL
    # ==========================================
    config.networks_folder = "../models/260303_SAPINN_repro/"
    config.network_name = "260303_repro_HRLR_HV01"
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
    # RESOLUTION  (HRLR: 0.5 mm voxels, 20 ms — native HR resolution)
    # ==========================================
    config.resolution = ml_collections.ConfigDict()
    config.resolution.from_file = False
    config.resolution.dx = 0.0005   # ×1 — no factor-2 scaling
    config.resolution.dy = 0.0005
    config.resolution.dz = 0.0005
    config.resolution.dt = 0.02

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
    config.constants.venc = 1.7  # HV01 (sv17)

    # ==========================================
    # NETWORK ARCHITECTURE
    # ==========================================
    config.network = ml_collections.ConfigDict()
    config.network.in_dim = 4
    config.network.out_dim = 6          # [u, v, w, dp/dx, dp/dy, dp/dz]
    config.network.depth = 6
    config.network.hidden_features = 128
    config.network.arch = "WIRE"
    config.network.sigma_0 = 30
    config.network.omega_0 = 30
    config.network.complex = False

    # ==========================================
    # META-LEARNING (disabled)
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
    # META-INIT (disabled)
    # ==========================================
    config.load_meta_init = False
    config.meta_init_path = ""

    # ==========================================
    # TRAINING PARAMETERS
    # ==========================================
    config.training = ml_collections.ConfigDict()
    config.training.iterations = 40_000
    config.training.data_points_per_batch = 10000
    config.training.coll_points_per_batch = 10000
    config.training.boundary_points_per_batch = 10000

    # Optimizer
    config.training.lr = 1e-4
    config.training.lr_decay_iter = 25000
    config.training.lr_decay_factor = 0.5
    config.training.use_LBFGS = True
    config.training.BFGS_lr = 1e-1
    config.training.iterations_before_BFGS = 99_999  # Effectively disabled
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
    # SELF-ADAPTIVE PINN — ENABLED (same as repro_SA)
    # ==========================================
    config.training.self_adaptive = True
    config.training.adaptive_sampling = True
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
    # ==========================================
    config.training.use_physics_loss = True
    config.training.physics_loss_on_data_points = True
    config.training.use_navier_stokes = True
    config.training.use_divergence = False
    config.training.use_PPE = False
    config.training.PPE_weight = 0.001
    config.training.predict_gradients = True
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
    # PLOTTING  (no upsampling — HRLR already at reference resolution)
    # ==========================================
    config.plot = ml_collections.ConfigDict()
    config.plot.iter = 5000
    config.plot.gt = True
    config.plot.t_step = 2
    config.plot.t_step_2 = 6
    config.plot.z_slice = 20
    config.plot.spatial_factor = 1   # No spatial upsampling needed
    config.plot.temporal_factor = 1  # No temporal upsampling needed
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
    config.predictions.peak_flow_idx = 12  # HV01
    config.predictions.flow_idx2 = 12
    config.predictions.predict_reference_data = True
    config.predictions.predict_SR_data = False
    config.predictions.compare_noisy_vs_ref = False
    config.predictions.denormalize = True
    config.predictions.fluid_region = True

    return config
