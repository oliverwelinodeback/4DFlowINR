# Imports
import time
import torch
import wandb
from utils.loss_utils import compute_data_loss, compute_physics_loss, compute_boundary_loss, update_loss_weights, navier_stokes_loss
from utils.prepare_data import prepare_data, load_data, extract_fluid_region, sample_collocation_points, sample_boundary_points, load_ref_data, prepare_ref_data
from utils.utils import copy_cource_code, save_checkpoint, save_ckpt, save_ckpt_min, load_ckpt_min, load_ckpt, sample_to_device, sample_ref_to_device, sample_from_gpu, sample_ref_from_gpu, plot_predictions, evaluate_predictions, plot_predictions_vs_reference, set_seed
import networks
from configs.tunings_251106.Config_251031_sweep_WIRE_momentum_ALL_sv_newMasks import get_config, get_sweep_config
#try:
#    from configs.Config_251008_sweep_test import get_sweep_config
#except ImportError:
#    get_sweep_config = None
#from configs.Config_250923_KI_HV01_x2_tx2_momentum import get_config
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
import numpy as np

def train(config=None, run_name=None, use_sweep=False):

    print("Starting script")

    if use_sweep:
        # Initialize wandb for this run
        run = wandb.init(project="SRFlowNIR")
        sweep_config = wandb.config

        data_file = sweep_config["data_file"]
        if data_file == "../data/healthy/HV01_05mm3_20ms_LR_sv17_tSNR10_newMask.h5":
            config.data_file = data_file
            config.data_file_ref = "../data/healthy/HV01_05mm3_20ms.h5"
            file_type = "HV01_sv17"
            config.constants.venc = 1.7
            config.predictions.peak_flow_idx = 12
        if data_file == "../data/healthy/HV03_05mm3_20ms_LR_sv13_tSNR10_newMask.h5":
            config.data_file = data_file
            config.data_file_ref = "../data/healthy/HV03_05mm3_20ms.h5"
            file_type = "HV03_sv13"
            config.constants.venc = 1.3
            config.predictions.peak_flow_idx = 4
        if data_file == "../data/healthy/HV06_05mm3_20ms_LR_sv12_tSNR10_newMask.h5":
            config.data_file = data_file
            config.data_file_ref = "../data/healthy/HV06_05mm3_20ms.h5"
            file_type = "HV06_sv12"
            config.constants.venc = 1.2
            config.predictions.peak_flow_idx = 2
        if data_file == "../data/stenosis_50/ICAD28_05mm3_20ms_LR_sv13_tSNR10_newMask.h5":
            config.data_file = data_file
            config.data_file_ref = "../data/stenosis_50/ICAD28_05mm3_20ms.h5"
            file_type = "ICAD28_sv13"
            config.constants.venc = 1.3
            config.predictions.peak_flow_idx = 2
        if data_file == "../data/stenosis_50/ICAD48_05mm3_20ms_LR_sv13_tSNR10_newMask.h5":
            config.data_file = data_file
            config.data_file_ref = "../data/stenosis_50/ICAD48_05mm3_20ms.h5"
            file_type = "ICAD48_sv13"
            config.constants.venc = 1.3
            config.predictions.peak_flow_idx = 14
        if data_file == "../data/stenosis_50/ICAD98_05mm3_20ms_LR_sv51_tSNR10_newMask.h5":
            config.data_file = data_file
            config.data_file_ref = "../data/stenosis_50/ICAD98_05mm3_20ms.h5"
            file_type = "ICAD98_sv51"
            config.constants.venc = 5.1
            config.predictions.peak_flow_idx = 12
        if data_file == "../data/stenosis_70/ICAD17_05mm3_20ms_LR_sv41_tSNR10_newMask.h5":
            config.data_file = data_file
            config.data_file_ref = "../data/stenosis_70/ICAD17_05mm3_20ms.h5"
            file_type = "ICAD17_sv41"
            config.constants.venc = 4.1
            config.predictions.peak_flow_idx = 8
        if data_file == "../data/stenosis_70/ICAD21_05mm3_20ms_LR_sv26_tSNR10_newMask.h5":
            config.data_file = data_file
            config.data_file_ref = "../data/stenosis_70/ICAD21_05mm3_20ms.h5"
            file_type = "ICAD21_sv26"
            config.constants.venc = 2.6
            config.predictions.peak_flow_idx = 12
        if data_file == "../data/stenosis_70/ICAD146_05mm3_20ms_LR_sv17_tSNR10_newMask.h5":
            config.data_file = data_file
            config.data_file_ref = "../data/stenosis_70/ICAD146_05mm3_20ms.h5"
            file_type = "ICAD146_sv17"
            config.constants.venc = 1.7
            config.predictions.peak_flow_idx = 8


        data_file = sweep_config["data_file"]
        if data_file == "../data/invivo/HV01.h5":
            config.data_file = data_file
            file_type = "HV01_invivo"
            config.resolution.dx = 0.0009821 #0.0005*2
            config.resolution.dy =  0.0009821 # 0.0005*2
            config.resolution.dz =  0.001 
            config.resolution.dt = 0.0826 #0.02*2
        if data_file == "../data/invivo/HV03.h5":
            config.data_file = data_file
            file_type = "HV03_invivo"
            config.resolution.dx = 0.0009821 #0.0005*2
            config.resolution.dy =  0.0009821 # 0.0005*2
            config.resolution.dz =  0.001 
            config.resolution.dt = 0.0826 #0.02*2
        if data_file == "../data/invivo/HV06.h5":
            config.data_file = data_file
            file_type = "HV06_invivo"
            config.resolution.dx = 0.0009821 #0.0005*2
            config.resolution.dy =  0.0009821 # 0.0005*2
            config.resolution.dz =  0.001 
            config.resolution.dt = 0.0826 #0.02*2

        if data_file == "../data/invivo/ICAD28.h5":
            config.data_file = data_file
            file_type = "ICAD28_invivo"
            config.resolution.dx = 0.00098214 #0.0005*2
            config.resolution.dy =  0.00098214 # 0.0005*2
            config.resolution.dz =  0.001 
            config.resolution.dt = 0.0868 #0.02*2
        if data_file == "../data/invivo/ICAD48.h5":
            config.data_file = data_file
            file_type = "ICAD48_invivo"
            config.resolution.dx = 0.00098214 #0.0005*2
            config.resolution.dy =  0.00098214 # 0.0005*2
            config.resolution.dz =  0.001 
            config.resolution.dt = 0.0868 #0.02*2
        if data_file == "../data/invivo/ICAD98.h5":
            config.data_file = data_file
            file_type = "ICAD98_invivo"
            config.resolution.dx =  0.0010417 #0.0005*2
            config.resolution.dy =  0.0010417 # 0.0005*2
            config.resolution.dz =  0.001 
            config.resolution.dt = 0.042699 #0.02*2

        if data_file == "../data/invivo/ICAD17.h5":
            config.data_file = data_file
            file_type = "ICAD17_invivo"
            config.resolution.dx = 0.00098214 #0.0005*2
            config.resolution.dy =  0.00098214 # 0.0005*2
            config.resolution.dz =  0.001 
            config.resolution.dt = 0.0868 #0.02*2
        if data_file == "../data/invivo/ICAD21.h5":
            config.data_file = data_file
            file_type = "ICAD21_invivo"
            config.resolution.dx =  0.0011458 #0.0005*2
            config.resolution.dy =  0.0011458 # 0.0005*2
            config.resolution.dz =  0.0011 
            config.resolution.dt = 0.042699 #0.02*2
        if data_file == "../data/invivo/ICAD146.h5":
            config.data_file = data_file
            file_type = "ICAD146_invivo"
            config.resolution.dx = 0.00098214 #0.0005*2
            config.resolution.dy =  0.00098214 # 0.0005*2
            config.resolution.dz =  0.001 
            config.resolution.dt = 0.0434 #0.02*2

        #if data_file == "../data/stenosis_70/ICAD21_05mm3_20ms_sv26_tSNR10.h5":
        #    config.data_file = data_file
        #    file_type = "sv26_original"
        #    config.constants.venc = 2.6
        #    config.predictions.peak_flow_idx = 12
#
        #if data_file == "../data/stenosis_70/ICAD21_05mm3_20ms_sv26_tSNR10_newMask.h5":
        #    config.data_file = data_file
        #    file_type = "sv26_newMask"
        #    config.constants.venc = 2.6
        #    config.predictions.peak_flow_idx = 12
#
        #if data_file == "../data/stenosis_70/ICAD21_05mm3_20ms_LR_dv_hv26_tSNR8_newMask.h5":
        #    config.data_file = data_file
        #    file_type = "dv26_newMask"
        #    config.constants.venc = 2.6
        #    config.predictions.peak_flow_idx = 12
#
        #if data_file == "../data/stenosis_70/ICAD21_05mm3_20ms_LR_dv_hv26_tSNR8.h5":
        #    config.data_file = data_file
        #    file_type = "dv26_original"
        #    config.constants.venc = 2.6
        #    config.predictions.peak_flow_idx = 12


        # Sweep parameters:
        #omega_0 = sweep_config["network.omega_0"]
        #sigma_0 = sweep_config["network.sigma_0"]
        ##fourier_mapping_size = sweep_config["network.fourier_mapping_size"]
        ##fourier_scale = sweep_config["network.fourier_scale"]
        #t_len = sweep_config["template.t_len"]
        #hidden_features = sweep_config["network.hidden_features"]
        #if hidden_features == 128:
        #    config.training.data_points_per_batch = 10000
        #    config.training.coll_points_per_batch = 10000

        ##network_arch = sweep_config["network.arch"]
        ##hidden_features = sweep_config["network.hidden_features"]
        ##BFGS_lr = sweep_config["training.BFGS_lr"]
        ##iterations_before_BFGS = sweep_config["training.iterations_before_BFGS"]
        ##lr_decay_iter = sweep_config["training.lr_decay_iter"]
        ##network_sigma_0 = sweep_config["network.sigma_0"]
        #training_use_vector_potential = sweep_config["training.use_vector_potential"]
        #training_sample_collocation = sweep_config["training.sample_collocation"]
        #if training_sample_collocation:
        #    config.training.use_physics_loss = True
        #    config.training.physics_loss_on_data_points = True
        #    config.training.use_divergence = True
        #else:
        #    config.training.use_physics_loss = False
        #    config.training.physics_loss_on_data_points = False
        #    config.training.use_divergence = False
        ###training_lr_decay_iter = sweep_config["training.lr_decay_iter"]
        ###training_BFGS_lr = sweep_config["training.BFGS_lr"]
        #training_iterations_before_BFGS = sweep_config["training.iterations_before_BFGS"]
        #gradW = sweep_config["training.grad_weight_scheme"]
        ##training_use_cosine = sweep_config["training.use_cosine"]
        #training_sample_boundary = sweep_config["training.sample_boundary"]
        #if training_use_cosine:
        #    config.training.use_mse = False
        #    config.training.use_cosine = True
        #else:
        #    config.training.use_mse = True
        #    config.training.use_cosine = False
        #U_const = sweep_config["constants.U"]
        #L_const = sweep_config["constants.L"]
        #iterationes = sweep_config["training.iterations"]

        #train_loss = sweep_config["training.loss"]
        #loss_change_iter = None
        #if train_loss == 'mse':
        #    config.training.use_mse = True
        #    config.training.use_cosine = False
        #elif train_loss=='cos':
        #    config.training.use_mse = False
        #    config.training.use_cosine = True
        #elif train_loss == "mse_and_cos":
        #    config.training.use_mse = True
        #    config.training.use_cosine = False
        #    loss_change_iter = 1000

        #data_file = sweep_config["data_file"]
        #if data_file == "../data/stenosis_70/ICAD17_05mm3_20ms_LR_dv_hv41_tSNR8.h5":
        #    file_type = "ICAD17_hv41"
        #    #config.constants.venc = 2.6

        # Run names
        #run.name = f"SIREN_sweep_Omega{omega_0}"
        #run.name = f"GAUSS_sweep_Sigma{sigma_0}"
        #run.name = f"WIRE_COMPLEX_sweep_Sigma{sigma_0}_Omega{omega_0}"
        #run.name = f"FFN_bias_sweep_fourier_mapping_size{fourier_mapping_size}_fourier_scale{fourier_scale}"

        #run.name = f"FFN_TEMPORAL_ICAD21_sweep_t{t_len}"
        
        #run.name = f"HV01_WIRE_REAL_itBFGS{iterations_before_BFGS}_lrdecay{lr_decay_iter}"
        #run.name = f"HV01_WIRE_REALsweep_sigma{sigma_0}_VecPot{training_use_vector_potential}_Phys{training_sample_collocation}_tlen{t_len}_Bnd{training_sample_boundary}"
        #run.name = f"{network_arch}_hidden{hidden_features}_BFGSlr{BFGS_lr}_itBFGS{iterations_before_BFGS}_lrdecay{lr_decay_iter}_t{t_len}"
        #run.name = f"WIRE_sweep_Sigma{network_sigma_0}_VecPot{training_use_vector_potential}_Phys{training_use_physics_loss}_lrdecay{training_lr_decay_iter}_BFGSlr{training_BFGS_lr}_itBFGS{training_iterations_before_BFGS}_trueCos{training_use_cosine}_trainSampleBound{training_sample_boundary}"
        #run.name = f"HV01_WIRE_REAL_REY_U{U_const}_L{L_const}"
        #run.name = f"HV01_WIRE_REAL_LONG_ITERS_{iterationes}"
        #run.name = f"HV01_WIRE_REAL_128FEAT_tlen{t_len}_FEAT{hidden_features}"
        #run.name = f"HV01_WIRE_CMPLXsweep_VecPot{training_use_vector_potential}_Phys{training_sample_collocation}_tlen{t_len}_Bnd{training_sample_boundary}"
        #run.name = f"HV01_WIRE_REAL_momentum_data{file_type}_loss{train_loss}"
        #run.name = f"ICAD21_WIRE_momentum_data{file_type}_loss{train_loss}"
        #run.name = f"WIRE_CMPLX_data_ALL_{file_type}_U{U_const}"
        #run.name = f"WIRE_MOMENTUM_ALL_{file_type}"
        #run.name = f"WIRE_CMPLX_data_maskTest_{file_type}"
        #run.name = f"WIRE_MOMENTUM_HV01_LongRun"
        #run.name = f"WIRE_DIVERGENCE_HV01_VecPot{training_use_vector_potential}_Phys{training_sample_collocation}_itBFGS{training_iterations_before_BFGS}_gradWScheme{gradW}"
        #run.name = f"WIRE_SAPINN_TEST2"
        #run.name = f"251031_WIRE_MOMENTUM_ALL_SV_NewMask_{file_type}"
        run.name = f"251031_WIRE_MOMENTUM_INVIVO_{file_type}"

        run.log({"run_name": run.name})

        # Sweep overrides
        #config.network.fourier_mapping_size = fourier_mapping_size
        #config.network.fourier_scale = fourier_scale
        #config.template.t_len = t_len
        ##config.network.arch = network_arch
        #config.network.hidden_features = hidden_features
        ##config.training.BFGS_lr = BFGS_lr
        ##config.training.iterations_before_BFGS = iterations_before_BFGS
        ##config.training.lr_decay_iter = lr_decay_iter
        #config.network.sigma_0 = sigma_0
        #config.sample_collocation = training_sample_collocation
        #config.training.use_vector_potential = training_use_vector_potential
        ####config.training.lr_decay_iter = training_lr_decay_iter
        ####config.training.BFGS_lr = training_BFGS_lr
        #config.training.iterations_before_BFGS = training_iterations_before_BFGS
        ##config.sample_boundary = training_sample_boundary
        ##config.data_file = data_file
        ##config.constants.L = L_const
        ##config.constants.U = U_const
        ##config.training.iterations = iterationes
        #config.training.grad_weight_scheme = gradW

        timestamp = datetime.now().strftime('%Y%m%d-%H%M')
        config.log_dir = f"{config.networks_folder}/{run.name}_{timestamp}"

    else:
        # Initialize wandb for this run
        wandb.init(project="SRFlowNIR", name=run_name, config=config.to_dict())

    # Store source files
    copy_cource_code(config.log_dir, directory_to_backup= [".", "configs", "utils"])

    # Set random seed
    set_seed(config.random_seed)

    # Load data
    u, v, w, p, px, py, pz, mask, config = load_data(config)

    # Prepare data
    uvw_data, xyz_data, mask_flat, boundary_mask_flat, standardization_factors, U_max  = prepare_data(config, u, v, w, p, px, py, pz, mask)

    config.U_max = U_max

    # Load and prepare reference data
    if config.include_ref:
        u_ref, v_ref, w_ref, p_ref, px_ref, py_ref, pz_ref, mask_ref = load_ref_data(config)
        uvw_data_ref, xyz_data_ref, mask_flat_ref, boundary_mask_flat_ref = prepare_ref_data(config, u, u_ref, v_ref, w_ref, 
                                                                                             p_ref, px_ref, py_ref, pz_ref,
                                                                                             mask_ref, U_max)
    mask_flat = mask_flat.astype(np.uint8)
    if config.include_ref:
        mask_flat_ref = mask_flat_ref.astype(np.uint8)

    # Sample collocation points
    xyz_collocation = None
    if config.sample_collocation:
        xyz_collocation = sample_collocation_points(config, xyz_data, mask_flat)
    ### xyz_collocation = np.copy(xyz_train)

    # Sample boundary points
    xyz_boundary = None
    if config.sample_boundary:
        xyz_boundary = sample_boundary_points(config, xyz_data, boundary_mask_flat)
    
    # Expand mask
    if config.setup.expand_mask:
        mask_flat = mask_flat + boundary_mask_flat

    # Include fluid region data
    if config.setup.fluid_region:
        uvw_train, xyz_train = extract_fluid_region(uvw_data, xyz_data, mask_flat, print_fluid_points=True)
        if config.include_ref:
            uvw_ref, xyz_ref = extract_fluid_region(uvw_data_ref, xyz_data_ref, mask_flat_ref)
    else:
        uvw_train, xyz_train = uvw_data, xyz_data
        if config.include_ref:
            uvw_ref, xyz_ref = uvw_data_ref, xyz_data_ref

    # Initialize network
    DEVICE = torch.device('cuda')
    if config.network.arch == "SIREN":
        model = networks.SIREN(
            in_dim=config.network.in_dim,
            out_dim=config.network.out_dim,
            depth=config.network.depth,
            hidden_features=config.network.hidden_features,
            first_omega_0=config.network.omega_0,
            hidden_omega_0=config.network.omega_0
        ).to(DEVICE)
    elif config.network.arch == "FF_SIREN":
        model = networks.FF_SIREN(
            in_dim=config.network.in_dim,
            out_dim=config.network.out_dim,
            depth=config.network.depth,
            hidden_features=config.network.hidden_features,
            first_omega_0=config.network.omega_0,
            hidden_omega_0=config.network.omega_0,
            fourier_mapping_size=config.network.fourier_mapping_size,
            scale=config.network.fourier_scale
        ).to(DEVICE)
    elif config.network.arch == "FFN":
        model = networks.FFN(
            input_dim=config.network.in_dim,
            output_dim=config.network.out_dim,
            depth=config.network.depth,
            hidden_dim=config.network.hidden_features,
            fourier_mapping_size=config.network.fourier_mapping_size,
            scale=config.network.fourier_scale
        ).to(DEVICE)
    elif config.network.arch == "WIRE":
        model = networks.WIRE(
            in_dim=config.network.in_dim,
            out_dim=config.network.out_dim,
            depth=config.network.depth,
            hidden_features=config.network.hidden_features,
            first_omega_0=config.network.omega_0,
            hidden_omega_0=config.network.omega_0,
            scale=config.network.sigma_0,
            complex=config.network.complex
        ).to(DEVICE)
    else:
        raise ValueError("Unknown network.")

    ########
    c_weights = None
    if config.sample_collocation and config.training.self_adaptive:
        N_c = xyz_collocation.shape[0]
        c_weights = np.ones(N_c, dtype=np.float32) 


    xyz_train_gpu = torch.from_numpy(xyz_train).float().to(DEVICE)
    uvw_train_gpu = torch.from_numpy(uvw_train).float().to(DEVICE)
    mask_flat_gpu = torch.from_numpy(mask_flat).float().to(DEVICE).view(-1, 1)
    
    xyz_collocation_gpu = None
    if config.sample_collocation:
        xyz_collocation_gpu = torch.from_numpy(xyz_collocation).float().to(DEVICE)
    
    xyz_boundary_gpu = None
    if config.sample_boundary:
        xyz_boundary_gpu = torch.from_numpy(xyz_boundary).float().to(DEVICE)
    if config.include_ref:
        xyz_ref_gpu = torch.from_numpy(xyz_ref).float().to(DEVICE)
        uvw_ref_gpu = torch.from_numpy(uvw_ref).float().to(DEVICE)
        mask_flat_ref_gpu = torch.from_numpy(mask_flat_ref).float().to(DEVICE).view(-1, 1)


    # Initialize optimizers
    Adam_optimizer = torch.optim.Adam(params=model.parameters(), lr=config.training.lr)
    if config.training.use_LBFGS:
        BFGS_optimizer = torch.optim.LBFGS(params=model.parameters(), lr=config.training.BFGS_lr)

        def closure():

                # Zero out gradients
                BFGS_optimizer.zero_grad()

                # Recompute forward pass
                model.train()

                # Data loss
                data_loss, _, _, _, _ = compute_data_loss(config, model, xyz_data_batch, uvw_data_batch, mask_batch, standardization_factors)

                # PDE residuals (physics loss)
                physics_losses = compute_physics_loss(
                    config, it, model,
                    xyz_collocation_batch, xyz_data_batch,
                    standardization_factors,
                    c_weights=c_batch_weights 
                )
                physics_loss = physics_losses["physics_loss"] if config.training.use_physics_loss else 0.0
                bound_loss = compute_boundary_loss(
                    config, model, xyz_boundary_batch
                ) if config.sample_boundary else 0.0

                # Weighted total loss depending on LBFGS logic
                if config.training.grad_weight_scheme:
                    # Suppose data_weight, physics_weight, bound_weight come from loss_weights
                    if config.sample_boundary:
                        data_weight, physics_weight, bound_weight = loss_weights
                        total_loss = (
                            data_weight * data_loss 
                            + physics_weight * config.training.physics_weight * physics_loss
                            + bound_weight * config.training.boundary_weight * bound_loss
                        )
                    else:
                        data_weight, physics_weight = loss_weights
                        total_loss = (
                            data_weight * data_loss 
                            + physics_weight * config.training.physics_weight * physics_loss
                        )
                else:
                    total_loss = data_loss + config.training.physics_weight * physics_loss + config.training.boundary_weight * bound_loss

                # Backprop
                total_loss.backward()
                return total_loss

    # Initialize loss weights
    loss_weights = None

    # Initialize csv logger    
    writer = SummaryWriter(log_dir=f"{config.log_dir}/tensorboard")

    # Ability to PRELOAD NETWORK incl optimizer here HERE
    #network_path = "../models/251031_WIRE_MOMENTUM_ALL/WIRE_MOMENTUM_ALL_HV03_hv13_20251031-1914/checkpoints/251031_WIRE_MOMENTUM_ALL_it10000.pth"
    #checkpoint = torch.load(network_path, map_location=DEVICE)
    #model.load_state_dict(checkpoint['model_state_dict'])
    
    scheduler = None
    start_it = 0
    #w_before = sum(p.sum().item() for p in model.parameters())
    #print("Loading model weights...")
    #resume_path = "../models/251106_SAPINN/251031_WIRE_SAPINN_Aggressive_20251107-1333/checkpoints/251031_WIRE_SAPINN_Aggressive_it000040.pth"
    #start_it = 0
    #if resume_path:
    #    start_it, c_weights, loss_weights = load_ckpt(resume_path, model, DEVICE,
    #                            adam_opt=Adam_optimizer,
    #                            lbfgs_opt=BFGS_optimizer if config.training.use_LBFGS else None,
    #                            scheduler=scheduler)
    #    #torch.cuda.empty_cache()
    #    #model.train()
    #    Adam_optimizer.zero_grad(set_to_none=True)
    #    if BFGS_optimizer: BFGS_optimizer.zero_grad()
    #    w_after  = sum(p.sum().item() for p in model.parameters())
    #    print(f"Weights before loading: {w_before}, after loading: {w_after}")
    #    print(loss_weights)

    # Start training
    start_time = time.time()
    for it in range(start_it, config.training.iterations):

        # Time iteration
        it_start_time = time.time()

        # Train
        model.train()

        # Sample random points and set to device
        #(
        #    xyz_data_batch, 
        #    uvw_data_batch, 
        #    mask_batch,
        #    xyz_collocation_batch, 
        #    xyz_boundary_batch
        #) = sample_to_device(config, xyz_train, xyz_collocation, xyz_boundary, uvw_train, mask_flat, DEVICE)

        # Sample random points
        

        if config.training.self_adaptive and config.training.adaptive_sampling: ## 0. Adaptive sampling - CHECK
            (
                xyz_data_batch, 
                uvw_data_batch, 
                mask_batch,
                xyz_collocation_batch, 
                xyz_boundary_batch,
                coll_indices
            ) = sample_from_gpu(config, xyz_train_gpu, xyz_collocation_gpu, xyz_boundary_gpu, uvw_train_gpu, mask_flat_gpu, c_weights)
        else:
            (
                xyz_data_batch, 
                uvw_data_batch, 
                mask_batch,
                xyz_collocation_batch, 
                xyz_boundary_batch,
                coll_indices
            ) = sample_from_gpu(config, xyz_train_gpu, xyz_collocation_gpu, xyz_boundary_gpu, uvw_train_gpu, mask_flat_gpu)


        # Build batch weights tensor if available
        c_batch_weights = None
        if (coll_indices is not None) and (c_weights is not None):
            c_batch_weights = torch.as_tensor(
                c_weights[coll_indices.detach().cpu().numpy()],
                device=DEVICE, dtype=torch.float32
            )

            #c_batch_weights = c_batch_weights / (c_batch_weights.mean() + 1e-12)

        # Track gradients
        xyz_data_batch.requires_grad = True
        if config.training.use_physics_loss:
            xyz_collocation_batch.requires_grad = True

        # Update gradient-based loss weights
        if config.training.grad_weight_scheme:
            loss_weights = update_loss_weights(config, model, loss_weights, it, xyz_data_batch,
                                               uvw_data_batch, mask, xyz_collocation_batch,
                                               xyz_boundary_batch, standardization_factors)

        # Predict and calculate data loss
        data_loss, _, _, _, _ = compute_data_loss(config, model, xyz_data_batch, uvw_data_batch, mask_batch, standardization_factors)

        # Predict and calculate PDE residuals (physics loss)
        if config.training.self_adaptive:
            physics_losses = compute_physics_loss( ## 1. ADD c_weights to physics loss
                config,
                it,
                model,
                xyz_collocation_batch,
                xyz_data_batch,
                standardization_factors,
                c_weights=c_batch_weights
            )
        else:
            physics_losses = compute_physics_loss(
                config,
                it,
                model,
                xyz_collocation_batch,
                xyz_data_batch,
                standardization_factors
            )

        physics_loss = physics_losses["physics_loss"]
        momentum_loss = physics_losses["momentum_loss"]
        div_loss = physics_losses["div_loss"]
        physics_loss_data = physics_losses["physics_loss_data"]
        momentum_loss_data = physics_losses["momentum_loss_data"]
        div_loss_data = physics_losses["div_loss_data"]

        # Predict and calculate boundary points
        bound_loss = compute_boundary_loss(config, model, xyz_boundary_batch)
        
        # Total loss
        if config.training.grad_weight_scheme:
            if config.sample_boundary:
                data_weight, physics_weight, bound_weight = loss_weights[0], loss_weights[1], loss_weights[2]
                total_loss = data_weight*data_loss + physics_weight*config.training.physics_weight*physics_loss + bound_weight*config.training.boundary_weight*bound_loss
            else:
                data_weight, physics_weight = loss_weights[0], loss_weights[1]
                total_loss = data_weight*data_loss + physics_weight*config.training.physics_weight*physics_loss
                bound_weight = None
        else:
            total_loss = data_loss + config.training.physics_weight*physics_loss + config.training.boundary_weight*bound_loss
            data_weight, physics_weight, bound_weight = None, None, None

        # Optimizer Step
        if config.training.use_LBFGS:
            if it < config.training.iterations_before_BFGS:
                # Update Adam optimizer
                Adam_optimizer.zero_grad()
                total_loss.backward()
                Adam_optimizer.step()
            
                if (it + 1) % config.training.lr_decay_iter == 0:
                    for param_group in Adam_optimizer.param_groups:
                        param_group['lr'] *= config.training.lr_decay_factor
            else:
                # Update LBFGS optimizer
                BFGS_optimizer.step(closure)
        else:
            # Update Adam optimizer
            Adam_optimizer.zero_grad()
            total_loss.backward()
            Adam_optimizer.step()

            # Learning rate decay
            if (it + 1) % config.training.lr_decay_iter == 0:
                for param_group in Adam_optimizer.param_groups:
                    param_group['lr'] *= config.training.lr_decay_factor

        if config.include_ref_loss:
            # Sample random points and set to device
            #xyz_ref_batch, uvw_ref_batch, mask_ref_batch = sample_ref_to_device(config, xyz_ref, uvw_ref, mask_flat_ref, DEVICE)
            xyz_ref_batch, uvw_ref_batch, mask_ref_batch = sample_ref_from_gpu(config, xyz_ref_gpu, uvw_ref_gpu, mask_flat_ref_gpu)
            if config.training.use_vector_potential:
                xyz_ref_batch.requires_grad = True
            ref_loss, _, mse_px, mse_py, mse_pz = compute_data_loss(config, model, xyz_ref_batch, uvw_ref_batch, mask_ref_batch, standardization_factors, denormalize=True, reference=True)
        else:
            ref_loss, mse_px, mse_py, mse_pz = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

        # Logging
        metrics = {
            "Loss/Train": total_loss.item(),
            "Loss/Data": data_loss.item(),
            "Loss/Physics": physics_loss.item(),
            "Loss/Boundary": bound_loss.item(),
            "Loss/Momentum": momentum_loss.item(),
            "Loss/Divergence": div_loss.item(),
            "Loss/Physics_data": physics_loss_data.item(),
            "Loss/Momentum_data": momentum_loss_data.item(),
            "Loss/Divergence_data": div_loss_data.item(),
            "Loss/Pressure_x": mse_px.item() if mse_px is not None else 0.0,
            "Loss/Pressure_y": mse_py.item() if mse_px is not None else 0.0,
            "Loss/Pressure_z": mse_pz.item() if mse_px is not None else 0.0,
            "Loss/Ref": ref_loss.item(),
            "Loss/data_weight": data_weight if data_weight is not None else 0.0,
            "Loss/physics_weight": physics_weight if physics_weight is not None else 0.0,
            "Loss/boundary_weight": bound_weight if bound_weight is not None else 0.0,
        }
        for key, value in metrics.items():
            writer.add_scalar(key, value, it)
        if (it + 1) % config.training.log_iter == 0:
            print(f"[Iteration {it+1}] total_loss={total_loss.item():.4f}, data_loss={data_loss.item():.4f}, ref_loss={ref_loss.item():.4f}, physics_loss={physics_loss.item():.4E}, it_time={round((time.time()-it_start_time)/config.training.log_iter, 5)} s total_time={round((time.time()-start_time)/60, 1)} min")

            wandb.log({
                "Loss/Train": total_loss.item(),
                "Loss/Data": data_loss.item(),
                "Loss/Physics": physics_loss.item(),
                "Loss/Boundary": bound_loss.item(),
                "Loss/Momentum": momentum_loss.item(),
                "Loss/Divergence": div_loss.item(),
                "Loss/Physics_data": physics_loss_data.item(),
                "Loss/Momentum_data": momentum_loss_data.item(),
                "Loss/Divergence_data": div_loss_data.item(),
                "Loss/Pressure_x": mse_px.item() if mse_px is not None else 0.0,
                "Loss/Pressure_y": mse_py.item() if mse_px is not None else 0.0,
                "Loss/Pressure_z": mse_pz.item() if mse_px is not None else 0.0,
                "Loss/Ref": ref_loss.item(),
                "Loss/data_weight": data_weight if data_weight is not None else 0.0,
                "Loss/physics_weight": physics_weight if physics_weight is not None else 0.0,
                "Loss/boundary_weight": bound_weight if bound_weight is not None else 0.0,
            }, step=it+1)
            
        # Plot current model predictions
        if (it + 1) % config.plot.iter == 0:
            plot_predictions(config, model, DEVICE, it+1, u, mask, U_max)
       
        # Compare with reference data
        if config.include_ref:
            if (it + 1) % config.training.error_iter == 0 or it == 0:
                
                metrics_eval = evaluate_predictions(config, model, DEVICE, it+1, xyz_ref, u_ref, v_ref, w_ref, p_ref, px_ref, py_ref, pz_ref, mask_ref, mask_flat_ref, U_max, standardization_factors, save_pred=False)
        
                plot_predictions_vs_reference(config, model, DEVICE, it+1, xyz_ref, 
                                            u, v, w, p, u_ref, v_ref, w_ref, p_ref, px_ref, py_ref, pz_ref, mask_ref, 
                                            mask_flat_ref, U_max, standardization_factors)
                
                # Log metrics to wandb
                log_dict = {
                    "Relative Error [Fluid]": metrics_eval['Relative error [Fluid]'],
                    "VNRMSE [Fluid]": metrics_eval['VNRMSE [Fluid]'],
                    "Directional error [Fluid]": metrics_eval["Directional error [Fluid]"],
                    "Divergence prediction [Fluid]": metrics_eval["Divergence prediction [Fluid]"],
                    "Divergence reference [Fluid]": metrics_eval["Divergence reference [Fluid]"],
                    "W k [Core]": metrics_eval['W k [Core]'],
                    "W r^2 [Core]": metrics_eval['W r^2 [Core]'],
                }

                # Add pressure metrics only if reference gradients are used
                if config.training.reference_gradients:
                    log_dict.update({
                        "Pressure Gradient Relative Error [Fluid]": metrics_eval['Relative error Pressure Gradient (%) [Fluid]'],
                        "Pressure gradient PX k [Core]": metrics_eval['PX k [Core]'],
                        "Pressure gradient PX r^2 [Core]": metrics_eval['PX r^2 [Core]'],
                        "Pressure gradient PY k [Core]": metrics_eval['PY k [Core]'],
                        "Pressure gradient PY r^2 [Core]": metrics_eval['PY r^2 [Core]'],
                        "Pressure gradient PZ k [Core]": metrics_eval['PZ k [Core]'],
                        "Pressure gradient PZ r^2 [Core]": metrics_eval['PZ r^2 [Core]'],
                    })

                # Log to W&B
                wandb.log(log_dict, step=it + 1)
                    
        # Save model at checkpoint
        if (it + 1) % config.training.summary_iter == 0:
            #save_checkpoint(model, it+1, config)
            #save_ckpt(
            #    f"{config.log_dir}/checkpoints/{config.network_name}_it{it+1:06d}.pth",
            #    model, Adam_optimizer, BFGS_optimizer if config.training.use_LBFGS else None,
            #    scheduler=None, iteration=it+1
            #)
            save_ckpt(f"{config.log_dir}/checkpoints/{config.network_name}_it{it+1:06d}.pth",
                        model, Adam_optimizer, BFGS_optimizer if config.training.use_LBFGS else None,
                        scheduler=scheduler, loss_weights=loss_weights, c_weights=c_weights, iteration=it+1)

        # Save model at end of training
        if (it + 1) == config.training.iterations:
            #save_checkpoint(model, it+1, config, final=True)
            #save_ckpt(
            #    f"{config.log_dir}/checkpoints/{config.network_name}_it{it+1:06d}.pth",
            #    model, Adam_optimizer, BFGS_optimizer if config.training.use_LBFGS else None,
            #    scheduler=None, iteration=it+1
            #)
            save_ckpt(f"{config.log_dir}/checkpoints/{config.network_name}_it{it+1:06d}.pth",
                        model, Adam_optimizer, BFGS_optimizer if config.training.use_LBFGS else None,
                        scheduler=scheduler, loss_weights=loss_weights, c_weights=c_weights, iteration=it+1)
            
            final_log_time = time.time() - start_time
            if config.include_ref:
                metrics_eval = evaluate_predictions(config, model, DEVICE, it+1, xyz_ref, u_ref, v_ref, w_ref, p_ref, px_ref, py_ref, pz_ref, mask_ref, mask_flat_ref, U_max, standardization_factors, save_pred=True, save_vti=True)
                final_log_dict = {
                    "FINAL Relative Error [Fluid]": metrics_eval['Relative error [Fluid]'],
                    "FINAL VNRMSE [Fluid]": metrics_eval['VNRMSE [Fluid]'],
                    "FINAL training time [min]": round(final_log_time/60, 2),
                    "FINAL W k [Core]": metrics_eval['W k [Core]'],
                    "FINAL W r^2 [Core]": metrics_eval['W r^2 [Core]'],
                    "FINAL W 2 k [Core]": metrics_eval['W 2 k [Core]'],
                    "FINAL W 2 r^2 [Core]": metrics_eval['W 2 r^2 [Core]'],
                }
                if config.training.reference_gradients:
                    final_log_dict.update({
                        "FINAL Pressure Gradient Relative Error [Fluid]": metrics_eval['Relative error Pressure Gradient (%) [Fluid]'],
                        "FINAL Pressure gradient PZ k [Core]": metrics_eval['PZ k [Core]'],
                        "FINAL Pressure gradient PZ r^2 [Core]": metrics_eval['PZ r^2 [Core]'],
                        "FINAL Pressure gradient PZ 2 r^2 [Core]": metrics_eval['PZ 2 r^2 [Core]'],
                    })
            
            else:
                final_log_dict = {
                    "FINAL training time [min]": round(final_log_time/60, 2),
                }
                plot_predictions(config, model, DEVICE, it+1, u, mask, U_max, save_pred=True, save_vti=True)

            # Log metrics to wandb
            wandb.log(final_log_dict)

        # Self-adaptive collocation point weight update
        if config.sample_collocation and config.training.self_adaptive:
            step = it + 1

            do_refresh = (
                (step == config.training.K_initial) or
                (step > config.training.K_initial and ((step - config.training.K_initial) % config.training.K == 0))
            )

            if do_refresh:
                model.eval()  # eval mode for stable residuals

                N_c = xyz_collocation.shape[0]
                # choose a large subset for refresh (OOM-safe)
                subset_size = min(N_c, config.training.points_to_update)
                refresh_chunk = config.training.chunk_size

                # random subset of collocation indices (CPU numpy)
                subset_idx = np.random.choice(N_c, size=subset_size, replace=False)

                # compute per-point residuals on this subset, chunked
                resid_list = []
                with torch.enable_grad():
                    for s in range(0, subset_size, refresh_chunk):
                        e = min(s + refresh_chunk, subset_size)
                        idx_block = subset_idx[s:e]

                        X_block = torch.from_numpy(xyz_collocation[idx_block]).to(DEVICE).float()
                        X_block.requires_grad_(True)

                        uvw_block = model(X_block)
                        per_point, _, _ = navier_stokes_loss(
                            uvw_block, X_block, standardization_factors, config,
                            return_per_point=True, build_graph=True
                        )
                        # per_point: 1D tensor of per-point residual magnitudes
                        resid_list.append(per_point.detach().cpu().numpy())

                resid_subset = np.concatenate(resid_list)  # shape (subset_size,)

                # ---- map residuals -> attention weights (stable) ----
                tau = float(config.training.tau)
                eps = 1e-12

                # 1) Clean residuals
                res = np.asarray(resid_subset, dtype=np.float64)
                is_finite = np.isfinite(res)
                if not np.all(is_finite):
                    res = res[is_finite]
                    if res.size == 0:
                        # fallback: uniform weights for this refresh
                        w_new = np.ones_like(subset_idx, dtype=np.float64)
                        beta = float(config.training.beta)
                        c_weights[subset_idx] = (1.0 - beta) * c_weights[subset_idx] + beta * w_new
                        c_weights = np.maximum(c_weights, eps)
                        c_weights /= (c_weights.mean() + eps)
                        model.train()
                        continue

                # 2) Clip extreme tails to prevent exp overflow but keep ranking
                p99 = np.percentile(res, 99.5)
                res = np.minimum(res, p99)

                # 3) Stable softmax via centering
                r = res / max(tau, eps)
                r -= np.max(r)
                z = np.exp(r)
                den = z.mean()
                if not np.isfinite(den) or den <= 0:
                    w_new_core = np.ones_like(z, dtype=np.float64)
                else:
                    w_new_core = z / (den + eps)

                # 4) Clip & sanitize
                clip_vals = list(config.training.weight_clip)
                clip_min, clip_max = min(clip_vals), max(clip_vals)
                w_new_core = np.clip(w_new_core, clip_min, clip_max)
                w_new_core = np.nan_to_num(w_new_core, nan=1.0, posinf=clip_max, neginf=clip_min)

                # 5) Scatter back into full-sized w_new aligned to subset_idx
                w_new = np.ones_like(subset_idx, dtype=np.float64)
                # res was computed on resid_subset which corresponds to subset_idx; lengths match
                w_new = w_new_core  # same length as subset_idx

                # 6) EMA blend + renorm + floor
                beta = float(config.training.beta)
                c_weights[subset_idx] = (1.0 - beta) * c_weights[subset_idx] + beta * w_new

                c_weights = np.nan_to_num(c_weights, nan=1.0, posinf=clip_max, neginf=clip_min)
                c_weights = np.maximum(c_weights, eps)
                c_weights /= (c_weights.mean() + eps)

                # diagnostics (optional)
                a, b = c_weights.min(), c_weights.max()
                print(f"[Self-adaptive] Refreshed c_weights on {subset_size} / {N_c} "
                    f"min={a:.3g}, max={b:.3g})")
                
                # Log metrics to wandb
                log_dict = {
                    #"c_weights/mean":m,
                    "c_weights/min":a,
                    "c_weights/max":b,
                    "c_weights/p50": float(np.percentile(c_weights, 50)),
                    "c_weights/p90": float(np.percentile(c_weights, 90)),
                }

                # Log to W&B
                wandb.log(log_dict, step=it + 1)                
                
                del X_block, uvw_block, per_point
                torch.cuda.empty_cache()

                model.train()

    wandb.finish()

if __name__ == "__main__":
    
    config = get_config()
    
    if config.sweep:

        # Define sweep configuration
        sweep_config = get_sweep_config()
        sweep_id = wandb.sweep(sweep=sweep_config, project="SRFlowNIR")
        wandb.agent(sweep_id, function=lambda: train(config=config, use_sweep=True))#, count=20)

    else:
        run_name = f"{config.network_name}"
        train(config=config, run_name=run_name)