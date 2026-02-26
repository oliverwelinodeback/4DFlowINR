"""
Meta-learning configuration for WIRE 4D Flow MRI Super-Resolution.

This config extends the base config with meta-learning specific parameters
for MAML and REPTILE training across multiple patient cases.
"""

import ml_collections
from datetime import datetime


def get_meta_config():
    """Get meta-learning hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # ============== Meta-Learning Parameters ==============
    config.meta = ml_collections.ConfigDict()

    # === Version Control ===
    config.meta.version = 'v2'  # 'v1' (original wrong LR->HR), 'v2' (corrected LR->LR)

    # Algorithm selection: 'MAML', 'FOMAML', or 'REPTILE'
    # MAML: Full second-order gradients (memory intensive)
    # FOMAML: First-order MAML (faster, less memory)
    # REPTILE: Simple weight averaging (most memory efficient)
    config.meta.method = 'MAML'

    # Inner loop (task-specific adaptation)
    config.meta.inner_lr = 0.01           # Learning rate for inner loop
    config.meta.inner_steps = 2           # Number of gradient steps per task
    config.meta.inner_points = 5000       # Points to sample per inner step

    # Outer loop (meta-update)
    config.meta.outer_lr = 1e-4           # Learning rate for meta-update
    config.meta.meta_batch_size = 3       # Number of tasks per meta-batch
    config.meta.max_iters = 5000          # Total meta-training iterations

    # === NEW V2: Support/Query Split ===
    config.meta.support_fraction = 0.5    # Fraction of LR data for support (rest for query)

    # === NEW V2: Physics Loss in MAML ===
    config.meta.use_physics_loss = False  # Flag to enable physics constraints
    config.meta.physics_weight = 0.1      # Weight for physics loss (if enabled)
    config.meta.coll_points_inner = 1000  # Collocation points for inner loop
    config.meta.coll_points_outer = 1000  # Collocation points for outer loop (unused in v2)

    # === NEW V2: NTK Analysis ===
    config.meta.compute_ntk_metrics = False  # Enable NTK energy analysis
    config.meta.ntk_eval_every = 500        # How often to compute NTK metrics
    config.meta.ntk_grid_size = 16          # Grid size for NTK (reduced for 4D: 16x16x8x4)
    config.meta.ntk_batch_size = 256        # Batch size for NTK computation

    # REPTILE-specific
    config.meta.reptile_epsilon = 1.0     # Interpolation factor for REPTILE

    # Validation
    config.meta.val_every = 100           # Validate every N iterations
    config.meta.val_inner_steps = 5       # Inner steps during validation

    # ============== Data Configuration ==============
    # Training cases (multiple .h5 files for meta-training)
    # Each case can have per-case parameters specified in case_params below
    config.meta.train_cases = [
        # --- HEALTHY (2 cases) ---
        "../data/healthy/HV01_05mm3_20ms_LR_sv17_tSNR10_newMask.h5",
        "../data/healthy/HV03_05mm3_20ms_LR_sv13_tSNR10_newMask.h5",

        # --- STENOSIS 50% (2 cases) ---
        "../data/stenosis_50/ICAD28_05mm3_20ms_LR_sv13_tSNR10_newMask.h5",
        "../data/stenosis_50/ICAD48_05mm3_20ms_LR_sv13_tSNR10_newMask.h5",

        # --- STENOSIS 70% (2 cases) ---
        "../data/stenosis_70/ICAD17_05mm3_20ms_LR_sv41_tSNR10_newMask.h5", 
        "../data/stenosis_70/ICAD21_05mm3_20ms_LR_sv26_tSNR10_newMask.h5",
    ]

    # Validation cases (Unseen data used to monitor generalization)
    # Contains 1 Healthy, 1 Stenosis 50%, and 1 Stenosis 70%
    # NOTE: These are NEVER used for training, only for monitoring metrics
    config.meta.val_cases = [
        "../data/healthy/HV06_05mm3_20ms_LR_sv12_tSNR10_newMask.h5",
        "../data/stenosis_50/ICAD98_05mm3_20ms_LR_sv51_tSNR10_newMask.h5",
        "../data/stenosis_70/ICAD146_05mm3_20ms_LR_sv17_tSNR10_newMask.h5",  # Moved from test_cases
    ]

    # Test cases (Currently unused in V2 implementation)
    # All evaluation is done using val_cases above
    config.meta.test_cases = []

    # Reference data for validation (high-res ground truth)
    config.meta.use_reference = True
    config.meta.ref_suffix = "_HR.h5"  # Naming convention for reference files

    # ============== Per-Case Parameters ==============
    # Map from case filename (stem) to case-specific parameters
    # If a case is not listed here, it will use default values from config.constants
    # venc: velocity encoding [m/s] - critical for cosine loss
    # peak_flow_idx: timestep index of peak flow (for evaluation only)
    config.meta.case_params = {
        # Healthy cases (example values - update with your actual values)
        "HV01_05mm3_20ms_LR_sv17_tSNR10_newMask": {"venc": 1.7, "peak_flow_idx": 12},
        "HV03_05mm3_20ms_LR_sv13_tSNR10_newMask": {"venc": 1.3, "peak_flow_idx": 4},
        "HV06_05mm3_20ms_LR_sv12_tSNR10_newMask": {"venc": 1.2, "peak_flow_idx": 2},
        # Stenosis 50% cases
        "ICAD28_05mm3_20ms_LR_sv13_tSNR10_newMask": {"venc": 1.3, "peak_flow_idx": 2},
        "ICAD48_05mm3_20ms_LR_sv13_tSNR10_newMask": {"venc": 1.3, "peak_flow_idx": 14},
        "ICAD98_05mm3_20ms_LR_sv51_tSNR10_newMask": {"venc": 5.1, "peak_flow_idx": 12},
        # Stenosis 70% cases
        "ICAD17_05mm3_20ms_LR_sv41_tSNR10_newMask": {"venc": 4.1, "peak_flow_idx": 8},
        "ICAD21_05mm3_20ms_LR_sv26_tSNR10_newMask": {"venc": 2.6, "peak_flow_idx": 12},
        "ICAD146_05mm3_20ms_LR_sv17_tSNR10_newMask": {"venc": 1.7, "peak_flow_idx": 8},
    }

    # Default values for cases not in case_params (uses config.constants.venc)
    config.meta.default_peak_flow_idx = 10

    # ============== Model Output Directory ==============
    timestamp = datetime.now().strftime('%Y%m%d-%H%M')
    config.networks_folder = f"../models/meta_{timestamp}/"
    config.network_name = f"WIRE_meta_{config.meta.method}"
    config.log_dir = f"{config.networks_folder}/{config.network_name}"
    config.random_seed = 1234

    # ============== Domain Configuration ==============
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

    # ============== Resolution ==============
    config.resolution = ml_collections.ConfigDict()
    config.resolution.from_file = False
    config.resolution.dx = 0.0005 * 2
    config.resolution.dy = 0.0005 * 2
    config.resolution.dz = 0.0005 * 2
    config.resolution.dt = 0.02 * 2

    # ============== Setup / Options ==============
    config.setup = ml_collections.ConfigDict()
    config.setup.include_pressure = False
    config.setup.include_time = True
    config.setup.fluid_region = True
    config.setup.expand_mask = False

    # ============== Normalization ==============
    config.vel_normalization = "characteristic"
    config.coords_characteristic = False
    config.coords_normalization = "standardize"
    config.global_normalization = True

    # Template-based normalization (ensures all cases have same coord range)
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

    # ============== Physical Constants ==============
    config.constants = ml_collections.ConfigDict()
    config.constants.U = 2.0
    config.constants.L = 0.005
    config.constants.T = config.constants.L / config.constants.U
    config.constants.rho = 1060
    config.constants.mu = 0.004
    config.constants.venc = 1.2

    # ============== Network Architecture ==============
    config.network = ml_collections.ConfigDict()
    config.network.in_dim = 4
    config.network.out_dim = 3
    config.network.depth = 6
    config.network.hidden_features = 128
    config.network.arch = "WIRE"
    config.network.sigma_0 = 30.0
    config.network.omega_0 = 30.0
    config.network.complex = True

    # ============== Training Parameters ==============
    config.training = ml_collections.ConfigDict()
    config.training.iterations = 8000  # For fine-tuning after meta-learning
    config.training.data_points_per_batch = 20000
    config.training.coll_points_per_batch = 20000
    config.training.boundary_points_per_batch = 10000

    # Optimizer (for fine-tuning)
    config.training.lr = 1e-4
    config.training.lr_decay_iter = 25000
    config.training.lr_decay_factor = 0.5
    config.training.use_LBFGS = False

    # Loss configuration
    config.training.use_mse = False
    config.training.use_cosine = True
    config.training.use_physics_loss = False
    config.training.physics_weight = 1.0
    config.training.u_weight = 1.0
    config.training.v_weight = 1.0
    config.training.w_weight = 1.0

    # Logging
    config.training.summary_iter = 500
    config.training.log_iter = 50
    config.training.error_iter = 500
    config.training.denormalize = True

    # ============== Collocation Points ==============
    config.sample_collocation = False  # Disable for basic meta-learning
    config.collocation_in_fluid = True
    config.collocation_points = 1_500_000
    config.sample_boundary = False

    # ============== Checkpointing ==============
    config.checkpoint = ml_collections.ConfigDict()
    config.checkpoint.save_every = 500
    config.checkpoint.keep_last_n = 3

    # ============== W&B Logging ==============
    config.wandb = ml_collections.ConfigDict()
    config.wandb.enabled = True
    config.wandb.project = "SRFlow-Meta"
    config.wandb.entity = None  # Set your W&B entity
    config.wandb.name = f"meta_{config.meta.method}_{timestamp}"

    return config


def get_finetune_config(meta_checkpoint_path: str, target_case: str):
    """
    Get configuration for fine-tuning a meta-learned model on a specific case.

    Args:
        meta_checkpoint_path: Path to meta-learned checkpoint
        target_case: Path to .h5 file for fine-tuning
    """
    config = get_meta_config()

    # Override for fine-tuning
    config.data_file = target_case
    config.meta.enabled = False  # Disable meta-learning for fine-tuning

    # Fine-tuning specific settings
    config.finetune = ml_collections.ConfigDict()
    config.finetune.meta_checkpoint = meta_checkpoint_path
    config.finetune.lr = 1e-4  # Can use higher LR with good initialization
    config.finetune.iterations = 2000  # Fewer iterations needed

    timestamp = datetime.now().strftime('%Y%m%d-%H%M')
    config.network_name = f"WIRE_finetune_{timestamp}"
    config.log_dir = f"{config.networks_folder}/{config.network_name}"

    return config
