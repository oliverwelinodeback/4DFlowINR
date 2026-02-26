# Imports
import time
import torch
import wandb
from utils.prepare_data import prepare_data, load_data, extract_fluid_region, sample_collocation_points, sample_boundary_points, load_ref_data, prepare_ref_data
from utils.utils import save_checkpoint, sample_to_device, sample_ref_to_device, plot_predictions, evaluate_predictions, plot_predictions_vs_reference, set_seed
import networks
#from configs.Config_1x_HV01_highSNR_momentum_PG import get_config
#from configs.Config_ICAD_1t_2x_healthy_lowSNR import get_config
from configs.tunings_251106.Config_251031_sweep_WIRE_momentum_ALL_sv_newMasks_MetaLearning import get_config, get_sweep_config

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt

from utils.ntk import ntk_eigendecomposition, visualize_ntk_results, get_structured_batch

import os
import seaborn as sns


def get_ntk(config=None, run_name=None, use_sweep=True):

    print("Starting script")
    
    run = wandb.init(
            project="SRFlowNIR",
            name=run_name,
            config=config.to_dict()
        )

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
        

        #config.training.tau = sweep_config["training.tau"]
        #config.training.beta = sweep_config["training.beta"]
        #config.training.K = sweep_config["training.K"]
        #config.points_to_update = sweep_config["training.points_to_update"]
        #config.training.weight_clip = [sweep_config["training.weight_clip_max"], sweep_config["training.weight_clip_min"]]
        #config.training.iterations_before_BFGS = sweep_config["training.iterations_before_BFGS"]
        #config.training.BFGS_lr = sweep_config["training.BFGS_lr"]
        #config.training.BFGS_max_iter = sweep_config["training.BFGS_max_iter"]
        #config.training.BFGS_history_size = sweep_config["training.BFGS_history_size"]
        #config.training.BFGS_tolerance_grad = sweep_config["training.BFGS_tolerance_grad"]
        #config.training.BFGS_tolerance_change = sweep_config["training.BFGS_tolerance_change"]

        #data_file = sweep_config["data_file"]
        #if data_file == "../data/invivo/HV01.h5":
        #    config.data_file = data_file
        #    file_type = "HV01_invivo"
        #    config.resolution.dx = 0.0009821 #0.0005*2
        #    config.resolution.dy =  0.0009821 # 0.0005*2
        #    config.resolution.dz =  0.001 
        #    config.resolution.dt = 0.0826 #0.02*2
        #if data_file == "../data/invivo/HV03.h5":
        #    config.data_file = data_file
        #    file_type = "HV03_invivo"
        #    config.resolution.dx = 0.0009821 #0.0005*2
        #    config.resolution.dy =  0.0009821 # 0.0005*2
        #    config.resolution.dz =  0.001 
        #    config.resolution.dt = 0.0826 #0.02*2
        #if data_file == "../data/invivo/HV06.h5":
        #    config.data_file = data_file
        #    file_type = "HV06_invivo"
        #    config.resolution.dx = 0.0009821 #0.0005*2
        #    config.resolution.dy =  0.0009821 # 0.0005*2
        #    config.resolution.dz =  0.001 
        #    config.resolution.dt = 0.0826 #0.02*2

        #if data_file == "../data/invivo/ICAD28.h5":
        #    config.data_file = data_file
        #    file_type = "ICAD28_invivo"
        #    config.resolution.dx = 0.00098214 #0.0005*2
        #    config.resolution.dy =  0.00098214 # 0.0005*2
        #    config.resolution.dz =  0.001 
        #    config.resolution.dt = 0.0868 #0.02*2
        #if data_file == "../data/invivo/ICAD48.h5":
        #    config.data_file = data_file
        #    file_type = "ICAD48_invivo"
        #    config.resolution.dx = 0.00098214 #0.0005*2
        #    config.resolution.dy =  0.00098214 # 0.0005*2
        #    config.resolution.dz =  0.001 
        #    config.resolution.dt = 0.0868 #0.02*2
        #if data_file == "../data/invivo/ICAD98.h5":
        #    config.data_file = data_file
        #    file_type = "ICAD98_invivo"
        #    config.resolution.dx =  0.0010417 #0.0005*2
        #    config.resolution.dy =  0.0010417 # 0.0005*2
        #    config.resolution.dz =  0.001 
        #    config.resolution.dt = 0.042699 #0.02*2

        #if data_file == "../data/invivo/ICAD17.h5":
        #    config.data_file = data_file
        #    file_type = "ICAD17_invivo"
        #    config.resolution.dx = 0.00098214 #0.0005*2
        #    config.resolution.dy =  0.00098214 # 0.0005*2
        #    config.resolution.dz =  0.001 
        #    config.resolution.dt = 0.0868 #0.02*2
        #if data_file == "../data/invivo/ICAD21.h5":
        #    config.data_file = data_file
        #    file_type = "ICAD21_invivo"
        #    config.resolution.dx =  0.0011458 #0.0005*2
        #    config.resolution.dy =  0.0011458 # 0.0005*2
        #    config.resolution.dz =  0.0011 
        #    config.resolution.dt = 0.042699 #0.02*2
        #if data_file == "../data/invivo/ICAD146.h5":
        #    config.data_file = data_file
        #    file_type = "ICAD146_invivo"
        #    config.resolution.dx = 0.00098214 #0.0005*2
        #    config.resolution.dy =  0.00098214 # 0.0005*2
        #    config.resolution.dz =  0.001 
        #    config.resolution.dt = 0.0434 #0.02*2

        #if data_file == "../data/stenosis_70/ICAD21_05mm3_20ms_sv26_tSNR10.h5":
        #    config.data_file = data_file
        #    file_type = "sv26_original"
        #    config.constants.venc = 2.6
        #    config.predictions.peak_flow_idx = 12

        #if data_file == "../data/stenosis_70/ICAD21_05mm3_20ms_sv26_tSNR10_newMask.h5":
        #    config.data_file = data_file
        #    file_type = "sv26_newMask"
        #    config.constants.venc = 2.6
        #    config.predictions.peak_flow_idx = 12

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

        #config.network.omega_0 = omega_0
        #config.network.sigma_0 = sigma_0
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
        #tau = sweep_config["training.tau"]
        #beta = sweep_config["training.beta"]
        #k = sweep_config["training.K"]
        #points_to_update = sweep_config["training.points_to_update"]
        #iterations_before_BFGS = sweep_config["training.iterations_before_BFGS"]
        #BFGS_lr = sweep_config["training.BFGS_lr"]
        #BFGS_max_iter = sweep_config["training.BFGS_max_iter"]
        #BFGS_history_size = sweep_config["training.BFGS_history_size"]
        #BFGS_tolerance_grad = sweep_config["training.BFGS_tolerance_grad"]
        #BFGS_tolerance_change = sweep_config["training.BFGS_tolerance_change"]


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
        run.name = f"251223_WIRE_MOMENTUM_SV_NewMask_{file_type}"
        #run.name = f"251202_WIRE_MOMENTUM_SV_NewMask_{file_type}_BFGS_itBefore{iterations_before_BFGS}_lr{BFGS_lr}_maxIter{BFGS_max_iter}_historySize{BFGS_history_size}_tolGrad{BFGS_tolerance_grad}_tolChange{BFGS_tolerance_change}"
        
        #run.name = f"251031_WIRE_MOMENTUM_INVIVO_{file_type}"
        print("Run name: ", run.name)
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
    # copy_source_code(config.log_dir, directory_to_backup= [".", "configs"])

    # Set random seed
    set_seed(config["random_seed"])

    # Load data
    u, v, w, p, px, py, pz, mask, config = load_data(config)

    # Prepare data
    uvw_data, xyz_data, mask_flat, boundary_mask_flat, standardization_factors, U_max  = prepare_data(config, u, v, w, p, px, py, pz, mask)

    config["U_max"] = U_max

    # Load and prepare reference data
    if config["include_ref"]:
        u_ref, v_ref, w_ref, p_ref, px_ref, py_ref, pz_ref, mask_ref = load_ref_data(config)
        uvw_data_ref, xyz_data_ref, mask_flat_ref, boundary_mask_flat_ref = prepare_ref_data(config, u, u_ref, v_ref, w_ref, 
                                                                                             p_ref, px_ref, py_ref, pz_ref,
                                                                                             mask_ref, U_max)
    mask_flat = mask_flat.astype(np.uint8)
    mask_flat_ref = mask_flat_ref.astype(np.uint8)

    # Sample collocation points
    xyz_collocation = None
    if config["sample_collocation"]:
        xyz_collocation = sample_collocation_points(config, xyz_data, mask_flat)
    ### xyz_collocation = np.copy(xyz_train)

    # Sample boundary points
    xyz_boundary = None
    if config["sample_boundary"]:
        xyz_boundary = sample_boundary_points(config, xyz_data, boundary_mask_flat)
    
    # Expand mask
    if config["setup"]["expand_mask"]:
        mask_flat = mask_flat + boundary_mask_flat

    # Include fluid region data
    if config["setup"]["fluid_region"]:
        uvw_train, xyz_train = extract_fluid_region(uvw_data, xyz_data, mask_flat, print_fluid_points=True)
        if config["include_ref"]:
            uvw_ref, xyz_ref = extract_fluid_region(uvw_data_ref, xyz_data_ref, mask_flat_ref)
    else:
        uvw_train, xyz_train = uvw_data, xyz_data
        if config["include_ref"]:
            uvw_ref, xyz_ref = uvw_data_ref, xyz_data_ref

    # Initialize network
    DEVICE = torch.device('cuda')
    if config["network"]["arch"] == "SIREN":
        model = networks.SIREN(
            in_dim=config["network"]["in_dim"],
            out_dim=config["network"]["out_dim"],
            depth=config["network"]["depth"],
            hidden_features=config["network"]["hidden_features"],
            first_omega_0=config["network"]["omega_0"],
            hidden_omega_0=config["network"]["omega_0"]
        ).to(DEVICE)
    elif config["network"]["arch"] == "FF_SIREN":
        model = networks.FF_SIREN(
            in_dim=config["network"]["in_dim"],
            out_dim=config["network"]["out_dim"],
            depth=config["network"]["depth"],
            hidden_features=config["network"]["hidden_features"],
            first_omega_0=config["network"]["omega_0"],
            hidden_omega_0=config["network"]["omega_0"],
            fourier_mapping_size=config["network"]["fourier_mapping_size"],
            scale=config["network"]["fourier_scale"]
        ).to(DEVICE)
    elif config["network"]["arch"] == "FFN":
        model = networks.FFN(
            input_dim=config["network"]["in_dim"],
            output_dim=config["network"]["out_dim"],
            depth=config["network"]["depth"],
            hidden_dim=config["network"]["hidden_features"],
            fourier_mapping_size=config["network"]["fourier_mapping_size"],
            scale=config["network"]["fourier_scale"]
        ).to(DEVICE)
    elif config["network"]["arch"] == "WIRE":
        model = networks.WIRE(
            in_dim=config["network"]["in_dim"],
            out_dim=config["network"]["out_dim"],
            depth=config["network"]["depth"],
            hidden_features=config["network"]["hidden_features"],
            first_omega_0=config["network"]["omega_0"],
            hidden_omega_0=config["network"]["omega_0"],
            scale=config["network"]["sigma_0"],
            complex=config["network"]["complex"]
        ).to(DEVICE)
    else:
        raise ValueError("Unknown network.")
    
    
    print(f"Initialized {config['network']['arch']} model with {sum(p.numel() for p in model.parameters())} parameters.")

    if config.load_meta_init:
        meta_checkpoint_path = config.meta_init_path
        print(f"\n{'='*60}")
        print(f"Loading meta-learned initialization from:")
        print(f"  {meta_checkpoint_path}")
        print(f"{'='*60}\n")
        
        checkpoint = torch.load(meta_checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print("✓ Meta-learned weights loaded successfully!")
        print("  This should converge faster than random initialization.\n")

    num_ntk_points = 10000
    
    # Use Structured Sampling (Option B) for clean slices
    xyz_subset_torch = get_structured_batch(xyz_data, n_points=num_ntk_points, device=DEVICE)
    
    print(f"Shape of data for NTK computation: {xyz_subset_torch.shape}")

    model.eval()
    components_to_analyze = [
        ("sum", None),  
        ("u", 0),
        ("v", 1),
        ("w", 2)
    ]
    for comp_name, comp_idx in components_to_analyze:
        print(f"\n--- Analyzing Component: {comp_name.upper()} ---")
        
        with torch.no_grad():
            eigvals, eigvecs, _ = ntk_eigendecomposition(
                model,
                xyz_subset_torch, 
                k=100,
                batch_size=128,
                component_idx=comp_idx  # <--- Select component here
            )

        # 4. Visualize
        outdir = os.path.join(os.getcwd(), "figures", "WIRE_NTK_analysis", comp_name)
        keyword = f"{config['network']['arch']}_{comp_name}"
        
        print(f"Saving plots to: {outdir}")
        visualize_ntk_results(
            eigvals=eigvals,
            eigvecs=eigvecs,
            coords=xyz_subset_torch,
            outdir=outdir,
            keyword=keyword,
            plot_eigvec_indices=range(5),
            z_slice_relative_positions=[0.2, 0.5, 0.8] 
        )

    print("\nNTK analysis script completed successfully!")
    # wandb.finish()

if __name__ == "__main__":
    
    config = get_config()
    
    if config.sweep:

        # Define sweep configuration
        sweep_config = get_sweep_config()
        sweep_id = wandb.sweep(sweep=sweep_config, project="SRFlowNIR")
        wandb.agent(sweep_id, function=lambda: get_ntk(config=config, use_sweep=True))#, count=20)

    else:
        run_name = f"{config.network_name}"
        get_ntk(config=config, run_name=run_name)
