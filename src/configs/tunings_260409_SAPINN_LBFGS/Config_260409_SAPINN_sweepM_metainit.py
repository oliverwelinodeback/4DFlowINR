import ml_collections
from datetime import datetime


def get_sweep_config():
    """
    Sweep M — Meta-init Adam/LBFGS split ratio + budget ablation

    SA confirmed dead with meta-init (Sweep 4 C/D ≈ A/B). Fixes SA=False and
    meta-init=True; explores the optimal Adam/LBFGS split and minimum budget.

    | Run | total_iters | itBFGS | Adam | LBFGS | Purpose                        |
    |-----|-------------|--------|------|-------|--------------------------------|
    | M1  | 15k         | 5k     | 5k   | 10k   | Short Adam, long LBFGS         |
    | M2  | 15k         | 7k     | 7k   | 8k    | Balanced                       |
    | M3  | 15k         | 10k    | 10k  | 5k    | Current best (Sweep 4 D)       |
    | M4  | 15k         | 12k    | 12k  | 3k    | Long Adam, short LBFGS (3k)    |
    | M5  | 20k         | 10k    | 10k  | 10k   | Does more LBFGS help?          |
    | M6  | 20k         | 15k    | 15k  | 5k    | Does more Adam help at 20k?    |
    | M7  | 10k         | 5k     | 5k   | 5k    | Minimum viable budget          |

    M1–M4: split ratio at fixed 15k budget.
    M5–M6: does extending to 20k total add anything?
    M7: is 10k budget sufficient with meta-init?

    Note: M4 LBFGS phase is only 3k — intentionally tests minimal LBFGS refinement.

    Method: grid (7 runs, ~60–120 min each).
    """
    timestamp = datetime.now().strftime('%Y%m%d-%H%M')
    return {
        'name': f'SAPINN_sweepM_metainit_{timestamp}',
        'method': 'grid',
        'metric': {'name': 'FINAL Relative Error [Fluid]', 'goal': 'minimize'},
        'parameters': {
            'run_id':      {'values': ["M1", "M2", "M3", "M4", "M5", "M6", "M7"]},
            'sweep_group': {'values': ["M"]},
        },
    }


def get_config(sweep_config=None):
    """
    Sweep M — meta-init split ratio + budget ablation.

    Base: ICAD48, meta-init=True (MAML checkpoint), SA=False, LBFGS=True.
    Per-run overrides (total_iters, itBFGS) applied by trainer.py SWEEP_M routing.

    self_adaptive=False: SA readback (line 169) is skipped before routing fires.
    disable_lr_decay=True: routing always sets lr_decay_iter=99_999.
    """
    config = ml_collections.ConfigDict()

    config.sweep = True

    # ==========================================
    # DATA — fixed to ICAD48
    # ==========================================
    config.data_file = "../data/stenosis_50/ICAD48_05mm3_20ms_LR_sv13_tSNR10_newMask.h5"
    config.include_ref = True
    config.include_ref_loss = True
    config.load_pressure_from_data = True
    config.data_file_ref = "../data/stenosis_50/ICAD48_05mm3_20ms.h5"
    config.ref_spatial_factor = 2
    config.ref_temporal_factor = 2

    # ==========================================
    # MODEL
    # ==========================================
    config.networks_folder = "../models/260409_SAPINN_sweepM_metainit/"
    config.network_name = "260409_SAPINN_sweepM"
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
    # RESOLUTION  (LR input: 1 mm voxels, 40 ms)
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
    config.constants.venc = 1.3  # ICAD48

    # ==========================================
    # NETWORK ARCHITECTURE
    # Must match MAML checkpoint: out_dim=6, omega_0=30, depth=6, hidden=128
    # ==========================================
    config.network = ml_collections.ConfigDict()
    config.network.in_dim = 4
    config.network.out_dim = 6
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
    # META-INIT — MAML final checkpoint (same as Sweep 4)
    # Trained on: HV03, HV06, ICAD28, ICAD98, ICAD17, ICAD146
    # ==========================================
    config.load_meta_init = True
    config.meta_init_path = "../models/260303_SAPINN_maml_newval/260303_MAML_SAPINN_HV01_ICAD48_ICAD21_val_20260330-1036/meta_learned_init_FINAL.pth"

    # ==========================================
    # TRAINING PARAMETERS
    # total_iters and itBFGS overridden per run_id by trainer.py SWEEP_M routing
    # ==========================================
    config.training = ml_collections.ConfigDict()
    config.training.iterations = 15_000        # default (M1–M4); overridden for M5–M7
    config.training.data_points_per_batch = 10000
    config.training.coll_points_per_batch = 10000
    config.training.boundary_points_per_batch = 10000

    # Optimizer — itBFGS overridden per run_id
    config.training.lr = 1e-4
    config.training.lr_decay_iter = 99_999
    config.training.lr_decay_factor = 0.5
    config.training.disable_lr_decay = True
    config.training.use_LBFGS = True
    config.training.BFGS_lr = 1e-1
    config.training.iterations_before_BFGS = 5_000    # overridden by routing
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
    # SELF-ADAPTIVE PINN — disabled for all M runs (SA confirmed dead with meta-init)
    # self_adaptive=False ensures SA readback is skipped before routing fires.
    # ==========================================
    config.training.self_adaptive = False
    config.training.adaptive_sampling = False
    config.training.tau = 0.02
    config.training.weight_clip = [6, 0.2]
    config.training.beta = 0.2
    config.training.K_initial = 10_000        # irrelevant for SA=False; kept for schema
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
    config.training.summary_iter = 2500
    config.training.log_iter = 250
    config.training.error_iter = 2500
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
    config.predictions.peak_flow_idx = 14   # ICAD48
    config.predictions.flow_idx2 = 14
    config.predictions.predict_reference_data = True
    config.predictions.predict_SR_data = False
    config.predictions.compare_noisy_vs_ref = False
    config.predictions.denormalize = True
    config.predictions.fluid_region = True

    return config
