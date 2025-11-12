# Imports
import os
import time
import numpy as np
import torch
import pandas as pd
from scipy.ndimage import zoom
from utils.prepare_data import create_and_normalize_coords, upsample_1d, extract_fluid_region, compute_outer_boundary_mask
from utils.evaluation_utils import (
    create_boundary_and_core_masks, calculate_relative_error, calculate_absolute_error, 
    calculate_rmse, calculate_absolute_error_pressure, calculate_rmse_pressure, linreg,
    calculate_divergence, calculate_directional_error, calculate_vnrmse, 
    calculate_gradient_absolute_error, calculate_gradient_relative_error,
    calculate_gradient_directional_error, calculate_gradient_nrmse
    )
from utils.prepare_data import prepare_data, load_data, extract_fluid_region, load_ref_data, prepare_ref_data
from utils.utils import save_to_h5
from utils.loss_utils import vector_potential_fn
from utils.preprocessing_utils import compute_outer_boundary_mask
import networks
from configs.tunings_251106.Config_251031_sweep_WIRE_momentum_ALL_sv_newMasks import get_config
#from configs.Config_ICAD_1t_2x_healthy_lowSNR import get_config
from utils.loss_utils import vector_potential_fn
import h5py

if __name__ == "__main__":

    print("Starting script")

    config = get_config()

    # Path to stored weights
    # --------------------------------
    # network_path = "/proj/multipress/users/x_javbi/SRFLOW2/SRFlowNIR/models/250129_AoModel/FFN_1t_20250130-1713/checkpoints/FFN_1t_it1000.pth"
    # results_directory = "../results/250130_Tests/FFN_1t_1"

    # network_path = "/proj/multipress/users/x_javbi/SRFLOW2/SRFlowNIR/models/250131_00100833_HNCM_V150_sys/FFN_1t_20250131-1052/checkpoints/FFN_1t_it44000.pth"
    #network_path = "../models/250530_PINN_PG_test/Config_1x_HV01_highSNR_PG_correctNorm_2000_wScheme_phys1_20250530-1803/checkpoints/Config_1x_HV01_highSNR_PG_correctNorm_2000_wScheme_phys1_it50000.pth"
    #results_directory = "../results/250530_PINN_PG_test/Config_1x_HV01_highSNR_PG_correctNorm_2000_wScheme_phys1_20250530-1803/Config_1x_HV01_highSNR_PG_correctNorm_2000_wScheme_phys1_it50000"

    #network_path = "../models/250618_PG_seed_collpoints_boundary_test/Config_1x_HV01_highSNR_PG_1000_10_000_000collpoints_seed128_noBoundary_ActuallyNoDiv_correctedWeightScheme_20250619-1343/checkpoints/Config_1x_HV01_highSNR_PG_1000_10_000_000collpoints_seed128_noBoundary_ActuallyNoDiv_correctedWeightScheme_it100000.pth"
    #results_directory = "../results/250618_PG_seed_collpoints_boundary_test/Config_1x_HV01_highSNR_PG_1000_10_000_000collpoints_seed128_noBoundary_ActuallyNoDiv_correctedWeightScheme_20250619-1343/checkpoints/Config_1x_HV01_highSNR_PG_1000_10_000_000collpoints_seed128_noBoundary_ActuallyNoDiv_correctedWeightScheme_it100000"

    #network_path = "../models/250623_PG_coord_pressure_test/Config_1x_HV01_highSNR_AbsolutePressure_20250623-1052/checkpoints/Config_1x_HV01_highSNR_AbsolutePressure_it100000.pth"
    #results_directory = "../results/250623_PG_coord_pressure_test/Config_1x_HV01_highSNR_AbsolutePressure_20250623-1052/checkpoints/Config_1x_HV01_highSNR_AbsolutePressure_it100000"

    #network_path = "../models/250625_PG_ReTests/LowRe_2.65_U0.1_L0.0001_20250625-1725/checkpoints/LowRe_2.65_U0.1_L0.0001_it95000.pth"
    #results_directory = "../results/250625_PG_ReEvaluations/LowRe_2.65_U0.1_L0.0001_20250625-1723/checkpoints/LowRe_2.65_U0.1_L0.0001_it95000"

    #network_path = "../models/250627_PG_correctedDenorm/OriginalRe_1325_U1_L0.005_AdjustedOuterCollocation_NoMaskExp_AdjustedCollo_20250627-1654/checkpoints/OriginalRe_1325_U1_L0.005_AdjustedOuterCollocation_NoMaskExp_AdjustedCollo_it95000.pth"
    #results_directory = "../results/250627_PG_correctedDenorm/OriginalRe_1325_U1_L0.005_AdjustedOuterCollocation_NoMaskExp_AdjustedCollo_20250627-1654/"

    #network_path = "../models/250924/ICAD17_SIREN_momentum_20250925-1216/checkpoints/ICAD17_SIREN_momentum_it25000.pth"
    #results_directory = "../results/250924/ICAD17_SIREN_momentum_20250925-1216/"

    #network_path = "../models/251031_WIRE_MOMENTUM_HV01_LongRun/WIRE_MOMENTUM_HV01_LongRun_20251031-1451/checkpoints/251031_WIRE_MOMENTUM_HV01_LongRun_it130000.pth"
    #results_directory = "../results/251031_WIRE_MOMENTUM_ALL/WIRE_MOMENTUM_HV01_LongRun_20251031-1451/"
    #config.predictions.peak_flow_idx = 12
    #config.data_file = "../data/healthy/HV01_05mm3_20ms_LR_dv_hv17_tSNR8.h5"
    #config.data_file_ref = "../data/healthy/HV01_05mm3_20ms.h5"

    #network_path = "../models/251031_WIRE_MOMENTUM_ALL/WIRE_MOMENTUM_ALL_HV01_hv17_20251031-1422/checkpoints/251031_WIRE_MOMENTUM_ALL_it10000.pth"
    #results_directory = "../results/251031_WIRE_MOMENTUM_ALL/WIRE_MOMENTUM_ALL_HV01_hv17_20251031-1422/"
    #config.predictions.peak_flow_idx = 12
    #config.data_file = "../data/healthy/HV01_05mm3_20ms_LR_dv_hv17_tSNR8.h5"
    #config.data_file_ref = "../data/healthy/HV01_05mm3_20ms.h5"
    #network_path = "../models/251031_WIRE_MOMENTUM_ALL/WIRE_MOMENTUM_ALL_HV03_hv13_20251031-1914/checkpoints/251031_WIRE_MOMENTUM_ALL_it10000.pth"
    #results_directory = "../results/251031_WIRE_MOMENTUM_ALL/WIRE_MOMENTUM_ALL_HV03_hv13_20251031-1914/"
    #config.predictions.peak_flow_idx = 4
    #config.data_file = "../data/healthy/HV03_05mm3_20ms_LR_dv_hv13_tSNR8.h5"
    #config.data_file_ref = "../data/healthy/HV03_05mm3_20ms.h5"
    #network_path = "../models/251031_WIRE_MOMENTUM_ALL/WIRE_MOMENTUM_ALL_HV06_hv12_20251101-0006/checkpoints/251031_WIRE_MOMENTUM_ALL_it10000.pth"
    #results_directory = "../results/251031_WIRE_MOMENTUM_ALL/WIRE_MOMENTUM_ALL_HV06_hv12_20251101-0006/"
    #config.predictions.peak_flow_idx = 2
    #config.data_file = "../data/healthy/HV06_05mm3_20ms_LR_dv_hv12_tSNR8.h5"
    #config.data_file_ref = "../data/healthy/HV06_05mm3_20ms.h5"
    #network_path = "../models/251031_WIRE_MOMENTUM_ALL/WIRE_MOMENTUM_ALL_ICAD17_hv41_20251101-0008/checkpoints/251031_WIRE_MOMENTUM_ALL_it10000.pth"
    #results_directory = "../results/251031_WIRE_MOMENTUM_ALL/WIRE_MOMENTUM_ALL_ICAD17_hv41_20251101-0008/"
    #config.predictions.peak_flow_idx = 8
    #config.data_file = "../data/stenosis_70/ICAD17_05mm3_20ms_LR_dv_hv41_tSNR8.h5"
    #config.data_file_ref = "../data/stenosis_70/ICAD17_05mm3_20ms.h5"
    #network_path = "../models/251031_WIRE_MOMENTUM_ALL/WIRE_MOMENTUM_ALL_ICAD21_hv26_20251101-0504/checkpoints/251031_WIRE_MOMENTUM_ALL_it10000.pth"
    #results_directory = "../results/251031_WIRE_MOMENTUM_ALL/WIRE_MOMENTUM_ALL_ICAD21_hv26_20251101-0504/"
    #config.predictions.peak_flow_idx = 12
    #config.data_file = "../data/stenosis_70/ICAD21_05mm3_20ms_LR_dv_hv26_tSNR8.h5"
    #config.data_file_ref = "../data/stenosis_70/ICAD21_05mm3_20ms.h5"
    #network_path = "../models/251031_WIRE_MOMENTUM_ALL/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD28_sv13_20251107-1402_normal/checkpoints/251031_WIRE_MOMENTUM_ALL_it10000.pth"
    #results_directory = "../results/251031_WIRE_MOMENTUM_ALL/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD28_sv13_20251107-1402_normal/"
    #config.predictions.peak_flow_idx = 2
    #config.data_file = "../data/stenosis_50/ICAD28_05mm3_20ms_LR_dv_hv13_tSNR8.h5"
    #config.data_file_ref = "../data/stenosis_50/ICAD28_05mm3_20ms.h5"
    #network_path = "../models/251031_WIRE_MOMENTUM_ALL/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD48_sv13_20251107-1824_normal/checkpoints/251031_WIRE_MOMENTUM_ALL_it10000.pth"
    #results_directory = "../results/251031_WIRE_MOMENTUM_ALL/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD48_sv13_20251107-1824_normal/"
    #config.predictions.peak_flow_idx = 14
    #config.data_file = "../data/stenosis_50/ICAD48_05mm3_20ms_LR_dv_hv13_tSNR8.h5"
    #config.data_file_ref = "../data/stenosis_50/ICAD48_05mm3_20ms.h5"
    #network_path = "../models/251031_WIRE_MOMENTUM_ALL/WIRE_MOMENTUM_ALL_ICAD98_hv51_20251031-1918/checkpoints/251031_WIRE_MOMENTUM_ALL_it10000.pth"
    #results_directory = "../results/251031_WIRE_MOMENTUM_ALL/WIRE_MOMENTUM_ALL_ICAD98_hv51_20251031-1918/"
    #config.predictions.peak_flow_idx = 12
    #config.data_file = "../data/stenosis_50/ICAD98_05mm3_20ms_LR_dv_hv51_tSNR8.h5"
    #config.data_file_ref = "../data/stenosis_50/ICAD98_05mm3_20ms.h5"
    #network_path = "../models/251031_WIRE_MOMENTUM_ALL/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD98_sv51_20251107-2249_normal/checkpoints/251031_WIRE_MOMENTUM_ALL_it10000.pth"
    #results_directory = "../results/251031_WIRE_MOMENTUM_ALL/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD98_sv51_20251107-2249_normal/"
    #config.predictions.peak_flow_idx = 8
    #config.data_file = "../data/stenosis_70/ICAD146_05mm3_20ms_LR_dv_hv17_tSNR8.h5"
    #config.data_file_ref = "../data/stenosis_70/ICAD146_05mm3_20ms.h5"


    #network_path = "../models/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_HV01_sv17_20251107-1401_normal/checkpoints/251031_WIRE_MOMENTUM_ALL_SV_it040000.pth"
    #results_directory = "../results/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_HV01_sv17_20251107-1401_normal/"
    #config.predictions.peak_flow_idx = 12
    #config.data_file = "../data/healthy/HV01_05mm3_20ms_LR_sv17_tSNR10_newMask.h5"
    #config.data_file_ref = "../data/healthy/HV01_05mm3_20ms.h5"
    #network_path = "../models/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_HV03_sv13_20251107-1818_normal/checkpoints/251031_WIRE_MOMENTUM_ALL_SV_it040000.pth"
    #results_directory = "../results/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_HV03_sv13_20251107-1818_normal/"
    #config.predictions.peak_flow_idx = 4
    #config.data_file = "../data/healthy/HV03_05mm3_20ms_LR_sv13_tSNR10_newMask.h5"
    #config.data_file_ref = "../data/healthy/HV03_05mm3_20ms.h5"
    #network_path = "../models/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_HV06_sv12_20251107-2238_normal/checkpoints/251031_WIRE_MOMENTUM_ALL_SV_it040000.pth"
    #results_directory = "../results/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_HV06_sv12_20251107-2238_normal/"
    #config.predictions.peak_flow_idx = 2
    #config.data_file = "../data/healthy/HV06_05mm3_20ms_LR_sv12_tSNR10_newMask.h5"
    #config.data_file_ref = "../data/healthy/HV06_05mm3_20ms.h5"
    #network_path = "../models/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD17_sv41_20251107-1403_normal/checkpoints/251031_WIRE_MOMENTUM_ALL_SV_it040000.pth"
    #results_directory = "../results/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD17_sv41_20251107-1403_normal/"
    #config.predictions.peak_flow_idx = 8
    #config.data_file = "../data/stenosis_70/ICAD17_05mm3_20ms_LR_sv41_tSNR10_newMask.h5"
    #config.data_file_ref = "../data/stenosis_70/ICAD17_05mm3_20ms.h5"
    network_path = "../models/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD21_sv26_20251107-1832_normal/checkpoints/251031_WIRE_MOMENTUM_ALL_SV_it040000.pth"
    results_directory = "../results/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD21_sv26_20251107-1832_normal/"
    config.predictions.peak_flow_idx = 12
    config.data_file = "../data/stenosis_70/ICAD21_05mm3_20ms_LR_sv26_tSNR10_newMask.h5"
    config.data_file_ref = "../data/stenosis_70/ICAD21_05mm3_20ms.h5"
    #network_path = "../models/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD28_sv13_20251107-1402_normal/checkpoints/251031_WIRE_MOMENTUM_ALL_SV_it040000.pth"
    #results_directory = "../results/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD28_sv13_20251107-1402_normal/"
    #config.predictions.peak_flow_idx = 2
    #config.data_file = "../data/stenosis_50/ICAD28_05mm3_20ms_LR_sv13_tSNR10_newMask.h5"
    #config.data_file_ref = "../data/stenosis_50/ICAD28_05mm3_20ms.h5"
    #network_path = "../models/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD48_sv13_20251107-1825_normal/checkpoints/251031_WIRE_MOMENTUM_ALL_SV_it040000.pth"
    #results_directory = "../results/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD48_sv13_20251107-1825_normal/"
    #config.predictions.peak_flow_idx = 14
    #config.data_file = "../data/stenosis_50/ICAD48_05mm3_20ms_LR_sv13_tSNR10_newMask.h5"
    #config.data_file_ref = "../data/stenosis_50/ICAD48_05mm3_20ms.h5"
    #network_path = "../models/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD98_sv51_20251107-2251_normal/checkpoints/251031_WIRE_MOMENTUM_ALL_SV_it040000.pth"
    #results_directory = "../results/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD98_sv51_20251107-2251_normal/"
    #config.predictions.peak_flow_idx = 12
    #config.data_file = "../data/stenosis_50/ICAD98_05mm3_20ms_LR_sv51_tSNR10_newMask.h5"
    #config.data_file_ref = "../data/stenosis_50/ICAD98_05mm3_20ms.h5"
    #network_path = "../models/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD146_sv17_20251107-2256_normal/checkpoints/251031_WIRE_MOMENTUM_ALL_SV_it040000.pth"
    #results_directory = "../results/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD146_sv17_20251107-2256_normal/"
    #config.predictions.peak_flow_idx = 8
    #config.data_file = "../data/stenosis_70/ICAD146_05mm3_20ms_LR_sv17_tSNR10_newMask.h5"
    #config.data_file_ref = "../data/stenosis_70/ICAD146_05mm3_20ms.h5"

    #network_path = "../models/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD17_sv41_20251107-1403_normal/checkpoints/251031_WIRE_MOMENTUM_ALL_SV_it010000.pth"
    #results_directory = "../results/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD17_sv41_20251107-1403_normal/"
    #config.predictions.peak_flow_idx = 8
    #config.data_file = "../data/stenosis_70/ICAD17_05mm3_20ms_LR_sv41_tSNR10_newMask.h5"
    #config.data_file_ref = "../data/stenosis_70/ICAD17_05mm3_20ms.h5"
    #network_path = "../models/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD21_sv26_20251107-1832_normal/checkpoints/251031_WIRE_MOMENTUM_ALL_SV_it010000.pth"
    #results_directory = "../results/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD21_sv26_20251107-1832_normal/"
    #config.predictions.peak_flow_idx = 12
    #config.data_file = "../data/stenosis_70/ICAD21_05mm3_20ms_LR_sv26_tSNR10_newMask.h5"
    #config.data_file_ref = "../data/stenosis_70/ICAD21_05mm3_20ms.h5"
    #network_path = "../models/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD28_sv13_20251107-1402_normal/checkpoints/251031_WIRE_MOMENTUM_ALL_SV_it010000.pth"
    #results_directory = "../results/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD28_sv13_20251107-1402_normal/"
    #config.predictions.peak_flow_idx = 2
    #config.data_file = "../data/stenosis_50/ICAD28_05mm3_20ms_LR_sv13_tSNR10_newMask.h5"
    #config.data_file_ref = "../data/stenosis_50/ICAD28_05mm3_20ms.h5"
    #network_path = "../models/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD48_sv13_20251107-1825_normal/checkpoints/251031_WIRE_MOMENTUM_ALL_SV_it010000.pth"
    #results_directory = "../results/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD48_sv13_20251107-1825_normal/"
    #config.predictions.peak_flow_idx = 14
    #config.data_file = "../data/stenosis_50/ICAD48_05mm3_20ms_LR_sv13_tSNR10_newMask.h5"
    #config.data_file_ref = "../data/stenosis_50/ICAD48_05mm3_20ms.h5"
    #network_path = "../models/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD98_sv51_20251107-2251_normal/checkpoints/251031_WIRE_MOMENTUM_ALL_SV_it010000.pth"
    #results_directory = "../results/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD98_sv51_20251107-2251_normal/"
    #config.predictions.peak_flow_idx = 12
    #config.data_file = "../data/stenosis_50/ICAD98_05mm3_20ms_LR_sv51_tSNR10_newMask.h5"
    #config.data_file_ref = "../data/stenosis_50/ICAD98_05mm3_20ms.h5"
    #network_path = "../models/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD146_sv17_20251107-2256_normal/checkpoints/251031_WIRE_MOMENTUM_ALL_SV_it010000.pth"
    #results_directory = "../results/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD146_sv17_20251107-2256_normal/"
    #config.predictions.peak_flow_idx = 8
    #config.data_file = "../data/stenosis_70/ICAD146_05mm3_20ms_LR_sv17_tSNR10_newMask.h5"
    #config.data_file_ref = "../data/stenosis_70/ICAD146_05mm3_20ms.h5"



    #network_path = "../models/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_INVIVO_HV01_sv17_20251107-1732_SAPINN/checkpoints/251031_WIRE_MOMENTUM_ALL_SV_SA_it040000.pth"
    #results_directory = "../results/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_INVIVO_HV01_sv17_20251107-1732_SAPINN/"
    #config.predictions.peak_flow_idx = 12
    #config.data_file = "../data/healthy/HV01_05mm3_20ms_LR_sv17_tSNR10_newMask.h5"
    #config.data_file_ref = "../data/healthy/HV01_05mm3_20ms.h5"
    #network_path = "../models/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_INVIVO_ICAD17_sv41_20251107-1734_SAPINN/checkpoints/251031_WIRE_MOMENTUM_ALL_SV_SA_it040000.pth"
    #results_directory = "../results/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_INVIVO_ICAD17_sv41_20251107-1734_SAPINN/"
    #config.predictions.peak_flow_idx = 8
    #config.data_file = "../data/stenosis_70/ICAD17_05mm3_20ms_LR_sv41_tSNR10_newMask.h5"
    #config.data_file_ref = "../data/stenosis_70/ICAD17_05mm3_20ms.h5"


    # Higher Re
    config.constants.U = 4.0
    config.constants.L = 0.005
    #network_path = "../models/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_HV01_sv17_20251107-1411_HigherRE/checkpoints/251031_WIRE_MOMENTUM_ALL_SV_it040000.pth"
    #results_directory = "../results/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_HV01_sv17_20251107-1411_HigherRE/"
    #config.predictions.peak_flow_idx = 12
    #config.data_file = "../data/healthy/HV01_05mm3_20ms_LR_sv17_tSNR10_newMask.h5"
    #config.data_file_ref = "../data/healthy/HV01_05mm3_20ms.h5"
    #network_path = "../models/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_HV03_sv13_20251107-1830_HigherRE/checkpoints/251031_WIRE_MOMENTUM_ALL_SV_it040000.pth"
    #results_directory = "../results/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_HV03_sv13_20251107-1830_HigherRE/"
    #config.predictions.peak_flow_idx = 4
    #config.data_file = "../data/healthy/HV03_05mm3_20ms_LR_sv13_tSNR10_newMask.h5"
    #config.data_file_ref = "../data/healthy/HV03_05mm3_20ms.h5"
    #network_path = "../models/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_HV06_sv12_20251107-2259_HigherRE/checkpoints/251031_WIRE_MOMENTUM_ALL_SV_it040000.pth"
    #results_directory = "../results/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_HV06_sv12_20251107-2259_HigherRE/"
    #config.predictions.peak_flow_idx = 2
    #config.data_file = "../data/healthy/HV06_05mm3_20ms_LR_sv12_tSNR10_newMask.h5"
    #config.data_file_ref = "../data/healthy/HV06_05mm3_20ms.h5"
    network_path = "../models/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD17_sv41_20251107-1412_HigherRe/checkpoints/251031_WIRE_MOMENTUM_ALL_SV_it040000.pth"
    results_directory = "../results/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD17_sv41_20251107-1412_HigherRe/"
    config.predictions.peak_flow_idx = 8
    config.data_file = "../data/stenosis_70/ICAD17_05mm3_20ms_LR_sv41_tSNR10_newMask.h5"
    config.data_file_ref = "../data/stenosis_70/ICAD17_05mm3_20ms.h5"
    #network_path = "../models/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD21_sv26_20251107-1838_HigherRe/checkpoints/251031_WIRE_MOMENTUM_ALL_SV_it040000.pth"
    #results_directory = "../results/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD21_sv26_20251107-1838_HigherRe/"
    #config.predictions.peak_flow_idx = 12
    #config.data_file = "../data/stenosis_70/ICAD21_05mm3_20ms_LR_sv26_tSNR10_newMask.h5"
    #config.data_file_ref = "../data/stenosis_70/ICAD21_05mm3_20ms.h5"
    #network_path = "../models/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD146_sv17_20251107-2252_HigherRe/checkpoints/251031_WIRE_MOMENTUM_ALL_SV_it040000.pth"
    #results_directory = "../results/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD146_sv17_20251107-2252_HigherRe/"
    #config.predictions.peak_flow_idx = 8
    #config.data_file = "../data/stenosis_70/ICAD146_05mm3_20ms_LR_sv17_tSNR10_newMask.h5"
    #config.data_file_ref = "../data/stenosis_70/ICAD146_05mm3_20ms.h5"



    # Higher SB
    #network_path = "../models/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD17_sv41_20251107-1407_HigherSB/checkpoints/251031_WIRE_MOMENTUM_ALL_SV_it040000.pth"
    #results_directory = "../results/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD17_sv41_20251107-1407_HigherSB/"
    #config.predictions.peak_flow_idx = 8
    #config.data_file = "../data/stenosis_70/ICAD17_05mm3_20ms_LR_sv41_tSNR10_newMask.h5"
    #config.data_file_ref = "../data/stenosis_70/ICAD17_05mm3_20ms.h5"
    #network_path = "../models/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD21_sv26_20251107-1836_HigherSB/checkpoints/251031_WIRE_MOMENTUM_ALL_SV_it040000.pth"
    #results_directory = "../results/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD21_sv26_20251107-1836_HigherSB/"
    #config.predictions.peak_flow_idx = 12
    #config.data_file = "../data/stenosis_70/ICAD21_05mm3_20ms_LR_sv26_tSNR10_newMask.h5"
    #config.data_file_ref = "../data/stenosis_70/ICAD21_05mm3_20ms.h5"
    #network_path = "../models/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD146_sv17_20251107-2255_HigherSB/checkpoints/251031_WIRE_MOMENTUM_ALL_SV_it040000.pth"
    #results_directory = "../results/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD146_sv17_20251107-2255_HigherSB/"
    #config.predictions.peak_flow_idx = 8
    #config.data_file = "../data/stenosis_70/ICAD146_05mm3_20ms_LR_sv17_tSNR10_newMask.h5"
    #config.data_file_ref = "../data/stenosis_70/ICAD146_05mm3_20ms.h5"
    #network_path = "../models/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD28_sv13_20251107-1406_HigherSB/checkpoints/251031_WIRE_MOMENTUM_ALL_SV_it040000.pth"
    #results_directory = "../results/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD28_sv13_20251107-1406_HigherSB/"
    #config.predictions.peak_flow_idx = 2
    #config.data_file = "../data/stenosis_50/ICAD28_05mm3_20ms_LR_sv13_tSNR10_newMask.h5"
    #config.data_file_ref = "../data/stenosis_50/ICAD28_05mm3_20ms.h5"
    #network_path = "../models/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD48_sv13_20251107-1825_HigherSB/checkpoints/251031_WIRE_MOMENTUM_ALL_SV_it040000.pth"
    #results_directory = "../results/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD48_sv13_20251107-1825_HigherSB/"
    #config.predictions.peak_flow_idx = 14
    #config.data_file = "../data/stenosis_50/ICAD48_05mm3_20ms_LR_sv13_tSNR10_newMask.h5"
    #config.data_file_ref = "../data/stenosis_50/ICAD48_05mm3_20ms.h5"
    #network_path = "../models/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD98_sv51_20251107-2251_HigherSB/checkpoints/251031_WIRE_MOMENTUM_ALL_SV_it040000.pth"
    #results_directory = "../results/251107_WIRE_MOMENTUM_ALL_SV/251031_WIRE_MOMENTUM_ALL_SV_NewMask_ICAD98_sv51_20251107-2251_HigherSB/"
    #config.predictions.peak_flow_idx = 12
    #config.data_file = "../data/stenosis_50/ICAD98_05mm3_20ms_LR_sv51_tSNR10_newMask.h5"
    #config.data_file_ref = "../data/stenosis_50/ICAD98_05mm3_20ms.h5"
    # --------------------------------

    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    # Load data
    u, v, w, p, px, py, pz, mask, config = load_data(config)

    # Save noisy data truth to results directory
    #save_to_h5(f"{results_directory}/healthy-05mm3_LR_SNR5_x1.h5", "u", u*mask)
    #save_to_h5(f"{results_directory}/healthy-05mm3_LR_SNR5_x1.h5", "v", v*mask)
    #save_to_h5(f"{results_directory}/healthy-05mm3_LR_SNR5_x1.h5", "w", w*mask)
    #save_to_h5(f"{results_directory}/healthy-05mm3_LR_SNR5_x1.h5", "p", p*mask)
    ## save_to_h5(f"{results_directory}/healthy-05mm3_LR_dv_241211.h5", "u", u*mask)
    ## save_to_h5(f"{results_directory}/healthy-05mm3_LR_dv_241211.h5", "v", v*mask)
    ## save_to_h5(f"{results_directory}/healthy-05mm3_LR_dv_241211.h5", "w", w*mask)
    ## save_to_h5(f"{results_directory}/healthy-05mm3_LR_dv_241211.h5", "p", p*mask)

    # Save to vtk file
    #h5_to_paraview(u, v, w, p, (config.resolution.dx, config.resolution.dy, config.resolution.dz), f"{results_directory}/healthy-05mm3_LR_SNR5_x1.vti")

    # Prepare data
    uvw_data, xyz_data, mask_flat, boundary_mask_flat, standardization_factors, U_max  = prepare_data(config, u, v, w, p, px, py, pz, mask)

    # Load and prepare reference data
    if config.include_ref:
        u_ref, v_ref, w_ref, p_ref, px_ref, py_ref, pz_ref, mask_ref = load_ref_data(config)
        uvw_data_ref, xyz_data_ref, mask_flat_ref, boundary_mask_flat_ref = prepare_ref_data(config, u, u_ref, v_ref, w_ref, p_ref, px_ref, py_ref, pz_ref, mask_ref, U_max)

        # Save noisy data truth to results directory
        ## save_to_h5(f"{results_directory}/healthy-05mm3.h5", "u", u_ref)
        ## save_to_h5(f"{results_directory}/healthy-05mm3.h5", "v", v_ref)
        ## save_to_h5(f"{results_directory}/healthy-05mm3.h5", "w", w_ref)
        ## save_to_h5(f"{results_directory}/healthy-05mm3.h5", "p", p_ref)

    # Expand mask
    ## if config.setup.expand_mask:
    ##     mask_flat = mask_flat + boundary_mask_flat
    ##     if config.include_ref:
    ##         mask_flat_ref = mask_flat_ref + boundary_mask_flat_ref

    # Include fluid region data
    if config.setup.fluid_region:
        uvw_train, xyz_train = extract_fluid_region(uvw_data, xyz_data, mask_flat)
        if config.include_ref:
            xyz_ref = xyz_data_ref[mask_flat_ref==1]
    else:
        uvw_train, xyz_train = uvw_data, xyz_data
        if config.include_ref:
            xyz_ref = xyz_data_ref

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
        )
    else:
        raise ValueError("Unknown network.")

    #print(torch.cuda.get_device_name(0))
    #print(torch.cuda.mem_get_info())

    model.to(DEVICE)
    # # Print model weights before loading
    # print("Model weights before loading:")
    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.data}")

    # Load trained model
    checkpoint = torch.load(network_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    if config.network.arch == "FFN":
        model.fourier_encoder.B = checkpoint['fourier_B']

    # # Print model weights after loading
    # print("\nModel weights after loading:")
    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.data}")

    # Predict and compare with reference data
    if config.predictions.predict_reference_data:

        # Create directory
        ref_directory = f'{results_directory}/ref_data'
        if not os.path.exists(ref_directory):
            os.makedirs(ref_directory)

        
        ##############
        # Predict reference coordinates
        ### model.eval()
        ### xyz_ref = torch.from_numpy(xyz_ref).float().to(DEVICE) ## cahnge to xyz_ref
        ### xyz_ref.requires_grad = config.training.use_vector_potential
        ### if config.training.use_vector_potential:
        ###     with torch.set_grad_enabled(True):
        ###         uvw_pred = model(xyz_ref)
        ###         uvw_pred = vector_potential_fn(uvw_pred, xyz_ref)
        ###         uvw_pred = uvw_pred.detach().cpu().numpy()
        ### else:
        ###     with torch.no_grad():
        ###         uvw_pred = model(xyz_ref)
        ###         uvw_pred = uvw_pred.cpu().numpy()
        ### #
        ###################
        
        # Predict reference coordinates (chunked; optional fluid-only)
        model.eval()

        # --- pick coordinates to evaluate ---
        scatter_back = False
        if config.predictions.fluid_region:
            # If xyz_ref is already fluid-only (common in your setup), just use it.
            # Only re-mask if lengths match (i.e., xyz_ref is still full grid).
            if ("mask_flat_ref" in locals() or "mask_flat_ref" in globals()) and \
            (xyz_ref.shape[0] == mask_flat_ref.shape[0]):
                fluid_indices = (mask_flat_ref == 1)
                coords_np = xyz_ref[fluid_indices]
                scatter_back = True   # we kept length; may want full-length output later
            else:
                coords_np = xyz_ref   # already masked upstream
                scatter_back = False
        else:
            coords_np = xyz_ref       # evaluate all points
            scatter_back = False

        num_pts = coords_np.shape[0]
        B = 20_000 # tune if needed

        pred_chunks = []
        start = 0
        while start < num_pts:
            end = min(start + B, num_pts)
            Xb = torch.from_numpy(coords_np[start:end]).float().to(DEVICE)

            if config.training.use_vector_potential:
                # Need grads for curl(A), but we do NOT keep graphs across batches
                Xb.requires_grad_(True)
                with torch.enable_grad():
                    Ub = model(Xb)
                    Ub = vector_potential_fn(Ub, Xb)     # returns velocities (and possibly pressure)
                    Ub = Ub.detach().cpu().numpy()
            else:
                # Pure inference: no graph, lowest memory
                with torch.no_grad():
                    Ub = model(Xb).cpu().numpy()

            pred_chunks.append(Ub)

            # free GPU early
            del Xb
            if 'Ub' in locals():
                del Ub
            torch.cuda.empty_cache()
            start = end

        uvw_pred = np.concatenate(pred_chunks, axis=0)  # shape: (N_evaluated, C)

        # --- if we only evaluated fluid, scatter back to full-length vector ---
        if scatter_back:
            uvw_pred_full = np.zeros((len(mask_flat_ref), uvw_pred.shape[1]), dtype=uvw_pred.dtype)
            uvw_pred_full[fluid_indices] = uvw_pred
            uvw_pred = uvw_pred_full
        
        '''
        # -------- Load trained model (inference) ----------
        checkpoint = torch.load(network_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)

        # Only set B if you actually stored it in the checkpoint
        if config.network.arch == "FFN" and ("fourier_B" in checkpoint):
            model.fourier_encoder.B = checkpoint["fourier_B"]

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        # -------- Chunked prediction over reference coords ----------
        # Decide what to evaluate (fluid-only if available & still full-length)
        scatter_back = False
        if config.predictions.fluid_region:
            if ("mask_flat_ref" in locals() or "mask_flat_ref" in globals()) and \
            (xyz_ref.shape[0] == mask_flat_ref.shape[0]):
                fluid_indices = (mask_flat_ref == 1)
                coords_np = xyz_ref[fluid_indices]        # fluid subset
                scatter_back = True
            else:
                coords_np = xyz_ref                        # already fluid-only upstream
                scatter_back = False
        else:
            coords_np = xyz_ref                            # full set

        # Ensure numpy array
        if isinstance(coords_np, torch.Tensor):
            coords_np = coords_np.detach().cpu().numpy()

        # Match model's device & dtype to avoid upcasting
        first_param = next(model.parameters())
        MODEL_DEVICE = first_param.device
        MODEL_DTYPE  = first_param.dtype

        def to_model_tensor(x_np):
            return torch.from_numpy(x_np).to(MODEL_DEVICE, dtype=MODEL_DTYPE, non_blocking=True)

        num_pts = int(coords_np.shape[0])
        # Batch size: reduce if memory is tight; smaller if using vector potential
        B = 500 if not config.training.use_vector_potential else 256

        pred_chunks = []
        start = 0

        if config.training.use_vector_potential:
            # Need grads per batch for curl(A), but don't keep graphs across batches
            while start < num_pts:
                end = min(start + B, num_pts)
                Xb = to_model_tensor(coords_np[start:end]).requires_grad_(True)
                with torch.enable_grad():
                    Ub = model(Xb)
                    Ub = vector_potential_fn(Ub, Xb)
                    Ub = Ub.detach().cpu().float().numpy()   # cast back to fp32 for safety
                pred_chunks.append(Ub)
                del Xb, Ub
                torch.cuda.empty_cache()
                start = end
        else:
            with torch.inference_mode():
                while start < num_pts:
                    end = min(start + B, num_pts)
                    Xb = to_model_tensor(coords_np[start:end])
                    Ub = model(Xb).detach().cpu().float().numpy()
                    pred_chunks.append(Ub)
                    del Xb, Ub
                    torch.cuda.empty_cache()
                    start = end

        uvw_pred = np.concatenate(pred_chunks, axis=0)
        del pred_chunks
        torch.cuda.empty_cache()

        # If we predicted only fluid, scatter back to full-length once (avoid doing it again later)
        if scatter_back:
            uvw_full = np.zeros((len(mask_flat_ref), uvw_pred.shape[1]), dtype=uvw_pred.dtype)
            uvw_full[fluid_indices] = uvw_pred
            uvw_pred = uvw_full
            del uvw_full
            torch.cuda.empty_cache()

        # --- Now split channels as you already do ---
        # u_pred = uvw_pred[:, 0] (reshape to t/x/y/z as needed)
        # ...

        # ---------- END CHUNKED PREDICTION ----------
        #################
        '''
        
        
        if config.predictions.fluid_region:
            fluid_indices = mask_flat_ref==1
            uvw_pred_full = np.zeros(((len(mask_flat_ref), len(uvw_pred[0]))))
            uvw_pred_full[fluid_indices] = uvw_pred
            uvw_pred = uvw_pred_full

        # Denormalize predictions
        if config.predictions.denormalize:
            if config.vel_normalization == "characteristic":
                uvw_pred[:, 0] *= config.constants.U  # u
                uvw_pred[:, 1] *= config.constants.U  # v
                uvw_pred[:, 2] *= config.constants.U  # w
                if (config.setup.include_pressure and config.training.reference_gradients):
                    _, _, _, std_x, _, std_y, _, std_z = standardization_factors
                    print(std_x, std_y, std_z)

                    uvw_pred[:, 3] *= config.constants.rho * (config.constants.U ** 2) / config.constants.L / std_x  # px
                    uvw_pred[:, 4] *= config.constants.rho * (config.constants.U ** 2) / config.constants.L / std_y # py
                    uvw_pred[:, 5] *= config.constants.rho * (config.constants.U ** 2) / config.constants.L / std_z # pz
                elif config.setup.include_pressure and not config.training.reference_gradients:
                    uvw_pred[:, 3] *= config.constants.rho * (config.constants.U ** 2)  # p

            elif config.vel_normalization == "max_velocity":
                uvw_pred[:, 0] *= U_max  # u
                uvw_pred[:, 1] *= U_max  # v
                uvw_pred[:, 2] *= U_max  # w
                if (config.setup.include_pressure and config.training.reference_gradients):
                    uvw_pred[:, 3] *= config.constants.rho * (config.constants.U ** 2) / config.constants.L  # px
                    uvw_pred[:, 4] *= config.constants.rho * (config.constants.U ** 2) / config.constants.L # py
                    uvw_pred[:, 5] *= config.constants.rho * (config.constants.U ** 2) / config.constants.L # pz
                elif config.setup.include_pressure and not config.training.reference_gradients:
                    uvw_pred[:, 3] *= config.constants.rho * (config.constants.U ** 2)  # p

        # Define dimensions based on include_time
        if config.setup.include_time:
            T, X, Y, Z = u_ref.shape
        else:
            X, Y, Z = u_ref.shape
            T = 1  # Single time step

            u_ref = np.expand_dims(u_ref, axis=0)
            v_ref = np.expand_dims(v_ref, axis=0)
            w_ref = np.expand_dims(w_ref, axis=0)
            p_ref = np.expand_dims(p_ref, axis=0) if config.setup.include_pressure else None

        D_pred = uvw_pred.shape[1]

        uvw_pred = uvw_pred.reshape(T, X, Y, Z, D_pred)

        u_pred = uvw_pred[:, :, :, :, 0]
        v_pred = uvw_pred[:, :, :, :, 1]
        w_pred = uvw_pred[:, :, :, :, 2]
        #p_pred = uvw_pred[config.plot.t_step*config.ref_temporal_factor, :, :, :, 3] if config.setup.include_pressure else None
        p_pred_x = uvw_pred[:, :, :, :, 3] if config.training.reference_gradients else None
        p_pred_y = uvw_pred[:, :, :, :, 4] if config.training.reference_gradients else None
        p_pred_z = uvw_pred[:, :, :, :, 5] if config.training.reference_gradients else None

        p_pred = uvw_pred[:, :, :, :, 3] if (config.setup.include_pressure and not config.training.reference_gradients) else None

        print(f"Predicted data shape: u: {u_pred.shape}, v: {v_pred.shape}, w: {w_pred.shape}")
        print("Max predicted velocities: ", np.max(u_pred), np.max(v_pred), np.max(w_pred))

        print(f"Reference data shape: u: {u_ref.shape}, v: {v_ref.shape}, w: {w_ref.shape}")
        print("Max reference velocities: ", np.max(u_ref), np.max(v_ref), np.max(w_ref))
        print("Ref mask shape: ", mask_ref.shape)

        # Save ref predictions to results directory
        #print("Saving velocity to h5...")
        #save_to_h5(f"{ref_directory}/healthy-05mm3_SR.h5", "u", u_pred, expand_dim=False)
        #save_to_h5(f"{ref_directory}/healthy-05mm3_SR.h5", "v", v_pred, expand_dim=False)
        #save_to_h5(f"{ref_directory}/healthy-05mm3_SR.h5", "w", w_pred, expand_dim=False)
        #if (config.setup.include_pressure and not config.training.reference_gradients):
        #    save_to_h5(f"{ref_directory}/healthy-05mm3_SR.h5", "p", p_pred, expand_dim=False)
        #elif config.training.reference_gradients:
        #    print("Saving pressure gradients to h5...")
        #    save_to_h5(f"{ref_directory}/healthy-05mm3_SR.h5", "p_x", p_pred_x, expand_dim=False)
        #    save_to_h5(f"{ref_directory}/healthy-05mm3_SR.h5", "p_y", p_pred_y, expand_dim=False)
        #    save_to_h5(f"{ref_directory}/healthy-05mm3_SR.h5", "p_z", p_pred_z, expand_dim=False)

        # -----------------
        # Mean normalize p_ref
        ## print(uvw_data_ref)
        ## print(uvw_data_ref.shape)
        ## print(mask_ref.shape)
        ## #mask_tiled = np.tile(mask_ref.ravel(), T)
        ## mask_tiled = np.stack([mask_ref]*T, axis=0)
        ## print('hejeeeeee')
        ## print(T)
        ## print(mask_tiled.shape)
        ## mean_p = np.mean(p_ref[mask_tiled==1])
        ## print('mean p: ', mean_p)
        ## mean_p_pred = np.mean(p_pred[mask_tiled==1])
        ## print('mean p pred: ', mean_p_pred)
        ## mean_diff = mean_p_pred - mean_p
        ## print('mean diff: ', mean_diff)
        ## p_normalized = p_pred
        ## p_normalized[mask_tiled==1] -= mean_diff
        ## save_to_h5(f"{ref_directory}/healthy-05mm3_SR.h5", "p_normalized", p_normalized)
        # -----------------

        # Get metrics
        peak_flow_idx = config.predictions.peak_flow_idx
        T = len(u_pred)
        nf_mask = 1.0 - mask_ref
        boundary_mask, core_mask = create_boundary_and_core_masks(mask_ref, 0.1, 'voxels')

        X,Y,Z = mask_ref.shape
        cov_a = np.sum(mask_ref)/(X*Y*Z)
        cov_b = np.sum(boundary_mask)/(X*Y*Z)
        cov_c = np.sum(core_mask)/(X*Y*Z)
        ratio_b = np.sum(boundary_mask)/np.sum(mask_ref)
        ratio_c = np.sum(core_mask)/np.sum(mask_ref)

        print(' ')
        print(f'Coverage: {100*cov_a:.3f} %')
        print(f'Boundary --- cov: {100*cov_b:.3f} %, ratio: {100*ratio_b:.3f} %')
        print(f'Core --- cov: {100*cov_c:.3f} %, ratio: {100*ratio_c:.3f} %')

        rel_err = np.zeros((T,3))
        abs_err = np.zeros((T,5))
        rmse = np.zeros((T,5))

        vnrmse = np.zeros((T,4))
        d_error = np.zeros((T,4))
        div_err = np.zeros((T,4))

        Ks = np.zeros((T,3,3))
        Ms = np.zeros((T,3,3))
        Rs = np.zeros((T,3,3))

        Ks_pgrad = np.zeros((T,3,3))
        Ms_pgrad = np.zeros((T,3,3))
        Rs_pgrad = np.zeros((T,3,3))

        grad_abs_err = np.zeros((T,3))
        grad_rel_err = np.zeros((T,3))
        grad_dir_err = np.zeros((T,3))
        grad_nrmse =   np.zeros((T,3))

        for t in range(T):

            rel_err[t,0] = (calculate_relative_error(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], mask_ref))
            rel_err[t,1] = (calculate_relative_error(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], boundary_mask))
            rel_err[t,2] = (calculate_relative_error(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], core_mask))

            abs_err[t,0] = (calculate_absolute_error(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], mask_ref))
            abs_err[t,1] = (calculate_absolute_error(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], boundary_mask))
            abs_err[t,2] = (calculate_absolute_error(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], core_mask))
            abs_err[t,3] = (calculate_absolute_error(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], nf_mask))

            rmse[t,0] = (calculate_rmse(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], mask_ref))
            rmse[t,1] = (calculate_rmse(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], boundary_mask))
            rmse[t,2] = (calculate_rmse(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], core_mask))
            rmse[t,3] = (calculate_rmse(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], nf_mask))

            vnrmse[t,0] = (calculate_vnrmse(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], mask_ref))
            vnrmse[t,1] = (calculate_vnrmse(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], boundary_mask))
            vnrmse[t,2] = (calculate_vnrmse(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], core_mask))
            vnrmse[t,3] = (calculate_vnrmse(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], nf_mask))

            d_error[t,0] = (calculate_directional_error(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], mask_ref))
            d_error[t,1] = (calculate_directional_error(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], boundary_mask))
            d_error[t,2] = (calculate_directional_error(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], core_mask))
            d_error[t,3] = (calculate_directional_error(u_pred[t], v_pred[t], w_pred[t], u_ref[t], v_ref[t], w_ref[t], nf_mask))

            div_err[t,0] = (calculate_divergence([u_pred[t], v_pred[t], w_pred[t]], [config.resolution.dx, config.resolution.dy, config.resolution.dz], mask_ref))
            div_err[t,1] = (calculate_divergence([u_pred[t], v_pred[t], w_pred[t]], [config.resolution.dx, config.resolution.dy, config.resolution.dz], boundary_mask))
            div_err[t,2] = (calculate_divergence([u_pred[t], v_pred[t], w_pred[t]], [config.resolution.dx, config.resolution.dy, config.resolution.dz], core_mask))
            div_err[t,3] = (calculate_divergence([u_pred[t], v_pred[t], w_pred[t]], [config.resolution.dx, config.resolution.dy, config.resolution.dz], nf_mask))

            Ks[t][0][0], Ms[t][0][0], Rs[t][0][0] = linreg(u_pred[t], u_ref[t], mask_ref)
            Ks[t][1][0], Ms[t][1][0], Rs[t][1][0] = linreg(v_pred[t], v_ref[t], mask_ref)
            Ks[t][2][0], Ms[t][2][0], Rs[t][2][0] = linreg(w_pred[t], w_ref[t], mask_ref)

            Ks[t][0][1], Ms[t][0][1], Rs[t][0][1] = linreg(u_pred[t], u_ref[t], boundary_mask)
            Ks[t][1][1], Ms[t][1][1], Rs[t][1][1] = linreg(v_pred[t], v_ref[t], boundary_mask)
            Ks[t][2][1], Ms[t][2][1], Rs[t][2][1] = linreg(w_pred[t], w_ref[t], boundary_mask)

            Ks[t][0][2], Ms[t][0][2], Rs[t][0][2] = linreg(u_pred[t], u_ref[t], core_mask)
            Ks[t][1][2], Ms[t][1][2], Rs[t][1][2] = linreg(v_pred[t], v_ref[t], core_mask)
            Ks[t][2][2], Ms[t][2][2], Rs[t][2][2] = linreg(w_pred[t], w_ref[t], core_mask)

            # px
            Ks_pgrad[t][0][0], Ms_pgrad[t][0][0], Rs_pgrad[t][0][0] = linreg(p_pred_x[t], px_ref[t], mask_ref)
            Ks_pgrad[t][0][1], Ms_pgrad[t][0][1], Rs_pgrad[t][0][1] = linreg(p_pred_x[t], px_ref[t], boundary_mask)
            Ks_pgrad[t][0][2], Ms_pgrad[t][0][2], Rs_pgrad[t][0][2] = linreg(p_pred_x[t], px_ref[t], core_mask)

            # py
            Ks_pgrad[t][1][0], Ms_pgrad[t][1][0], Rs_pgrad[t][1][0] = linreg(p_pred_y[t], py_ref[t], mask_ref)
            Ks_pgrad[t][1][1], Ms_pgrad[t][1][1], Rs_pgrad[t][1][1] = linreg(p_pred_y[t], py_ref[t], boundary_mask)
            Ks_pgrad[t][1][2], Ms_pgrad[t][1][2], Rs_pgrad[t][1][2] = linreg(p_pred_y[t], py_ref[t], core_mask)

            # pz
            Ks_pgrad[t][2][0], Ms_pgrad[t][2][0], Rs_pgrad[t][2][0] = linreg(p_pred_z[t], pz_ref[t], mask_ref)
            Ks_pgrad[t][2][1], Ms_pgrad[t][2][1], Rs_pgrad[t][2][1] = linreg(p_pred_z[t], pz_ref[t], boundary_mask)
            Ks_pgrad[t][2][2], Ms_pgrad[t][2][2], Rs_pgrad[t][2][2] = linreg(p_pred_z[t], pz_ref[t], core_mask)

            # Pressure gradient errors
            grad_abs_err[t, 0] = calculate_gradient_absolute_error(p_pred_x[t], p_pred_y[t], p_pred_z[t], px_ref[t], py_ref[t], pz_ref[t], mask_ref)
            grad_abs_err[t, 1] = calculate_gradient_absolute_error(p_pred_x[t], p_pred_y[t], p_pred_z[t], px_ref[t], py_ref[t], pz_ref[t], boundary_mask)
            grad_abs_err[t, 2] = calculate_gradient_absolute_error(p_pred_x[t], p_pred_y[t], p_pred_z[t], px_ref[t], py_ref[t], pz_ref[t], core_mask)

            grad_rel_err[t, 0] = calculate_gradient_relative_error(p_pred_x[t], p_pred_y[t], p_pred_z[t], px_ref[t], py_ref[t], pz_ref[t], mask_ref)
            grad_rel_err[t, 1] = calculate_gradient_relative_error(p_pred_x[t], p_pred_y[t], p_pred_z[t], px_ref[t], py_ref[t], pz_ref[t], boundary_mask)
            grad_rel_err[t, 2] = calculate_gradient_relative_error(p_pred_x[t], p_pred_y[t], p_pred_z[t], px_ref[t], py_ref[t], pz_ref[t], core_mask)

            grad_dir_err[t, 0] = calculate_gradient_directional_error(p_pred_x[t], p_pred_y[t], p_pred_z[t], px_ref[t], py_ref[t], pz_ref[t], mask_ref)
            grad_dir_err[t, 1] = calculate_gradient_directional_error(p_pred_x[t], p_pred_y[t], p_pred_z[t], px_ref[t], py_ref[t], pz_ref[t], boundary_mask)
            grad_dir_err[t, 2] = calculate_gradient_directional_error(p_pred_x[t], p_pred_y[t], p_pred_z[t], px_ref[t], py_ref[t], pz_ref[t], core_mask)

            grad_nrmse[t, 0] = calculate_gradient_nrmse(p_pred_x[t], p_pred_y[t], p_pred_z[t], px_ref[t], py_ref[t], pz_ref[t], mask_ref)
            grad_nrmse[t, 1] = calculate_gradient_nrmse(p_pred_x[t], p_pred_y[t], p_pred_z[t], px_ref[t], py_ref[t], pz_ref[t], boundary_mask)
            grad_nrmse[t, 2] = calculate_gradient_nrmse(p_pred_x[t], p_pred_y[t], p_pred_z[t], px_ref[t], py_ref[t], pz_ref[t], core_mask)
        
        print('Total avg')
        ## TODO - other losses (DE, nRMSE)
        rel_err_tot = np.mean(rel_err, axis=0)
        print(f'Relative error [Fluid] {rel_err_tot[0]:.1f}')
        print(f'Relative error [Bound] {rel_err_tot[1]:.1f}')
        print(f'Relative error [Core] {rel_err_tot[2]:.1f}')

        abs_err_tot = np.mean(abs_err, axis=0)
        print(f'Absolute error [Fluid] {abs_err_tot[0]:.4f}')
        print(f'Absolute error [Bound] {abs_err_tot[1]:.4f}')
        print(f'Absolute error [Core] {abs_err_tot[2]:.4f}')
        print(f'Absolute error [Non-F] {abs_err_tot[3]:.4f}')
        print(f'Absolute error Pressure [Fluid] {abs_err_tot[4]:.4f}')

        rmse_tot = np.mean(rmse, axis=0)
        print(f'R.M.S.   error [Fluid] {rmse_tot[0]:.4f}')
        print(f'R.M.S.   error [Bound] {rmse_tot[1]:.4f}')
        print(f'R.M.S.   error [Core] {rmse_tot[2]:.4f}')
        print(f'R.M.S.   error [Non-F] {rmse_tot[3]:.4f}')
        print(f'R.M.S.   error Pressure [Fluid] {rmse_tot[4]:.4f}')

        Ks_tot = np.mean(Ks, axis=0)
        print(f'U   K   [Fluid] {Ks_tot[0][0]:.3f}')
        print(f'U   K   [Bound] {Ks_tot[0][1]:.3f}')
        print(f'U   K   [Core] {Ks_tot[0][2]:.3f}')

        print(f'V   K   [Fluid] {Ks_tot[1][0]:.3f}')
        print(f'V   K   [Bound] {Ks_tot[1][1]:.3f}')
        print(f'V   K   [Core] {Ks_tot[1][2]:.3f}')

        print(f'W   K   [Fluid] {Ks_tot[2][0]:.3f}')
        print(f'W   K   [Bound] {Ks_tot[2][1]:.3f}')
        print(f'W   K   [Core] {Ks_tot[2][2]:.3f}')

        Ks_pgrad_tot = np.mean(Ks_pgrad, axis=0)
        #print(f'PX  K   [Fluid] {Ks_pgrad_tot[0][0]:.3f}')
        #print(f'PX  K   [Bound] {Ks_pgrad_tot[0][1]:.3f}')
        print(f'PX  K   [Core] {Ks_pgrad_tot[0][2]:.3f}')

        #print(f'PY  K   [Fluid] {Ks_pgrad_tot[1][0]:.3f}')
        #print(f'PY  K   [Bound] {Ks_pgrad_tot[1][1]:.3f}')
        print(f'PY  K   [Core] {Ks_pgrad_tot[1][2]:.3f}')

        #print(f'PZ  K   [Fluid] {Ks_pgrad_tot[2][0]:.3f}')
        #print(f'PZ  K   [Bound] {Ks_pgrad_tot[2][1]:.3f}')
        print(f'PZ  K   [Core] {Ks_pgrad_tot[2][2]:.3f}')

        Rs_tot = np.mean(Rs, axis=0)
        print(f'U R2     [Fluid] {Rs_tot[0][0]:.3f}')
        print(f'U R2     [Bound] {Rs_tot[0][1]:.3f}')
        print(f'U R2     [Core] {Rs_tot[0][2]:.3f}')

        print(f'V R2     [Fluid] {Rs_tot[1][0]:.3f}')
        print(f'V R2     [Bound] {Rs_tot[1][1]:.3f}')
        print(f'V R2     [Core] {Rs_tot[1][2]:.3f}')

        print(f'W R2     [Fluid] {Rs_tot[2][0]:.3f}')
        print(f'W R2     [Bound] {Rs_tot[2][1]:.3f}')
        print(f'W R2     [Core] {Rs_tot[2][2]:.3f}')

        Rs_pgrad_tot = np.mean(Rs_pgrad, axis=0)
        #print(f'PX R2    [Fluid] {Rs_pgrad_tot[0][0]:.3f}')
        #print(f'PX R2    [Bound] {Rs_pgrad_tot[0][1]:.3f}')
        print(f'PX R2    [Core] {Rs_pgrad_tot[0][2]:.3f}') 

        #print(f'PY R2    [Fluid] {Rs_pgrad_tot[1][0]:.3f}')
        #print(f'PY R2    [Bound] {Rs_pgrad_tot[1][1]:.3f}')
        print(f'PY R2    [Core] {Rs_pgrad_tot[1][2]:.3f}')

        #print(f'PZ R2    [Fluid] {Rs_pgrad_tot[2][0]:.3f}')
        #print(f'PZ R2    [Bound] {Rs_pgrad_tot[2][1]:.3f}')
        print(f'PZ R2    [Core] {Rs_pgrad_tot[2][2]:.3f}')

        print('-  '*9)

        print(' ')
        print(peak_flow_idx, 'Peak')
        print(f'U [Fluid] k: {Ks[peak_flow_idx][0][0]:.3f} \t m: {Ms[peak_flow_idx][0][0]:.4f} \t r^2: {Rs[peak_flow_idx][0][0]:.3f}')
        print(f'  [Bound] k: {Ks[peak_flow_idx][0][1]:.3f} \t m: {Ms[peak_flow_idx][0][1]:.4f} \t r^2: {Rs[peak_flow_idx][0][1]:.3f}')
        print(f'  [Core] k: {Ks[peak_flow_idx][0][2]:.3f} \t m: {Ms[peak_flow_idx][0][2]:.4f} \t r^2: {Rs[peak_flow_idx][0][2]:.3f}')

        print(' ')
        print(f'V [Fluid] k: {Ks[peak_flow_idx][1][0]:.3f} \t m: {Ms[peak_flow_idx][1][0]:.4f} \t r^2: {Rs[peak_flow_idx][1][0]:.3f}')
        print(f'  [Bound] k: {Ks[peak_flow_idx][1][1]:.3f} \t m: {Ms[peak_flow_idx][1][1]:.4f} \t r^2: {Rs[peak_flow_idx][1][1]:.3f}')
        print(f'  [Core] k: {Ks[peak_flow_idx][1][2]:.3f} \t m: {Ms[peak_flow_idx][1][2]:.4f} \t r^2: {Rs[peak_flow_idx][1][2]:.3f}')

        print(' ')
        print(f'W [Fluid] k: {Ks[peak_flow_idx][2][0]:.3f} \t m: {Ms[peak_flow_idx][2][0]:.4f} \t r^2: {Rs[peak_flow_idx][2][0]:.3f}')
        print(f'  [Bound] k: {Ks[peak_flow_idx][2][1]:.3f} \t m: {Ms[peak_flow_idx][2][1]:.4f} \t r^2: {Rs[peak_flow_idx][2][1]:.3f}')
        print(f'  [Core] k: {Ks[peak_flow_idx][2][2]:.3f} \t m: {Ms[peak_flow_idx][2][2]:.4f} \t r^2: {Rs[peak_flow_idx][2][2]:.3f}')

        print(' ')
        #print(f'PX [Fluid] k: {Ks_pgrad[peak_flow_idx][0][0]:.3f} \t m: {Ms_pgrad[peak_flow_idx][0][0]:.4f} \t r^2: {Rs_pgrad[peak_flow_idx][0][0]:.3f}')
        #print(f'   [Bound] k: {Ks_pgrad[peak_flow_idx][0][1]:.3f} \t m: {Ms_pgrad[peak_flow_idx][0][1]:.4f} \t r^2: {Rs_pgrad[peak_flow_idx][0][1]:.3f}')
        print(f'  PX [Core] k: {Ks_pgrad[peak_flow_idx][0][2]:.3f} \t m: {Ms_pgrad[peak_flow_idx][0][2]:.4f} \t r^2: {Rs_pgrad[peak_flow_idx][0][2]:.3f}')

        print(' ')
        #print(f'PY [Fluid] k: {Ks_pgrad[peak_flow_idx][1][0]:.3f} \t m: {Ms_pgrad[peak_flow_idx][1][0]:.4f} \t r^2: {Rs_pgrad[peak_flow_idx][1][0]:.3f}')
        #print(f'   [Bound] k: {Ks_pgrad[peak_flow_idx][1][1]:.3f} \t m: {Ms_pgrad[peak_flow_idx][1][1]:.4f} \t r^2: {Rs_pgrad[peak_flow_idx][1][1]:.3f}')
        print(f'  PY [Core] k: {Ks_pgrad[peak_flow_idx][1][2]:.3f} \t m: {Ms_pgrad[peak_flow_idx][1][2]:.4f} \t r^2: {Rs_pgrad[peak_flow_idx][1][2]:.3f}')

        print(' ')
        #print(f'PZ [Fluid] k: {Ks_pgrad[peak_flow_idx][2][0]:.3f} \t m: {Ms_pgrad[peak_flow_idx][2][0]:.4f} \t r^2: {Rs_pgrad[peak_flow_idx][2][0]:.3f}')
        #print(f'   [Bound] k: {Ks_pgrad[peak_flow_idx][2][1]:.3f} \t m: {Ms_pgrad[peak_flow_idx][2][1]:.4f} \t r^2: {Rs_pgrad[peak_flow_idx][2][1]:.3f}')
        print(f'  PZ [Core] k: {Ks_pgrad[peak_flow_idx][2][2]:.3f} \t m: {Ms_pgrad[peak_flow_idx][2][2]:.4f} \t r^2: {Rs_pgrad[peak_flow_idx][2][2]:.3f}')

        print(' ')

        grad_abs_err_tot = np.mean(grad_abs_err, axis=0)
        print(f'Absolute error Pressure Gradient [Fluid] {grad_abs_err_tot[0]:.4f}')
        print(f'Absolute error Pressure Gradient [Bound] {grad_abs_err_tot[1]:.4f}')
        print(f'Absolute error Pressure Gradient [Core] {grad_abs_err_tot[2]:.4f}')

        grad_rel_err_tot = np.mean(grad_rel_err, axis=0)
        print(f'Relative error Pressure Gradient [Fluid] {grad_rel_err_tot[0]*100:.4f} %')
        print(f'Relative error Pressure Gradient [Bound] {grad_rel_err_tot[1]*100:.4f} %')
        print(f'Relative error Pressure Gradient [Core] {grad_rel_err_tot[2]*100:.4f} %')

        grad_nrmse_tot = np.mean(grad_nrmse, axis=0)
        print(f'Pressure Gradient NRMSE [Fluid] {grad_nrmse_tot[0]*100:.2f} %')
        print(f'Pressure Gradient NRMSE [Bound] {grad_nrmse_tot[1]*100:.2f} %')
        print(f'Pressure Gradient NRMSE [Core] {grad_nrmse_tot[2]*100:.2f} %')

        grad_dir_err_tot = np.mean(grad_dir_err, axis=0)
        print(f'Pressure Gradient Directional Error [Fluid] {grad_dir_err_tot[0]:.2f} deg')
        print(f'Pressure Gradient Directional Error [Bound] {grad_dir_err_tot[1]:.2f} deg')
        print(f'Pressure Gradient Directional Error [Core] {grad_dir_err_tot[2]:.2f} deg')

        # Save metrics to csv
        metrics = {
            'Relative error [Fluid]': rel_err_tot[0],
            'Relative error [Bound]': rel_err_tot[1],
            'Relative error [Core]': rel_err_tot[2],

            'Absolute error [Fluid]': abs_err_tot[0],
            'Absolute error [Bound]': abs_err_tot[1],
            'Absolute error [Core]': abs_err_tot[2],
            'Absolute error [Non-F]': abs_err_tot[3],
            'Absolute error Pressure [Fluid]': abs_err_tot[4],

            'R.M.S. error [Fluid]': rmse_tot[0],
            'R.M.S. error [Bound]': rmse_tot[1],
            'R.M.S. error [Core]': rmse_tot[2],
            'R.M.S. error [Non-F]': rmse_tot[3],
            'R.M.S. error Pressure [Fluid]': rmse_tot[4],

            'Absolute error Pressure Gradient [Fluid]': grad_abs_err_tot[0],
            'Absolute error Pressure Gradient [Bound]': grad_abs_err_tot[1],
            'Absolute error Pressure Gradient [Core]': grad_abs_err_tot[2],

            'Relative error Pressure Gradient [%] [Fluid]': grad_rel_err_tot[0]*100,
            'Relative error Pressure Gradient [%] [Bound]': grad_rel_err_tot[1]*100,
            'Relative error Pressure Gradient [%] [Core]': grad_rel_err_tot[2]*100,

            'Pressure Gradient NRMSE [%] [Fluid]': grad_nrmse_tot[0]*100,
            'Pressure Gradient NRMSE [%] [Bound]': grad_nrmse_tot[1]*100,
            'Pressure Gradient NRMSE [%] [Core]': grad_nrmse_tot[2]*100,

            'Pressure Gradient Directional Error (deg) [Fluid]': grad_dir_err_tot[0],
            'Pressure Gradient Directional Error (deg) [Bound]': grad_dir_err_tot[1],
            'Pressure Gradient Directional Error (deg) [Core]': grad_dir_err_tot[2],

            'U R2     [Fluid]': Rs_tot[0][0],
            'U R2     [Bound]': Rs_tot[0][1],
            'U R2     [Core]': Rs_tot[0][2],
            'V R2     [Fluid]': Rs_tot[1][0],
            'V R2     [Bound]': Rs_tot[1][1],
            'V R2     [Core]': Rs_tot[1][2],
            'W R2     [Fluid]': Rs_tot[2][0],
            'W R2     [Bound]': Rs_tot[2][1],
            'W R2     [Core]': Rs_tot[2][2],

            'U K     [Fluid]': Ks_tot[0][0],
            'U K     [Bound]': Ks_tot[0][1],
            'U K     [Core]': Ks_tot[0][2],
            'V K     [Fluid]': Ks_tot[1][0],
            'V K     [Bound]': Ks_tot[1][1],
            'V K     [Core]': Ks_tot[1][2],
            'W K     [Fluid]': Ks_tot[2][0],
            'W K     [Bound]': Ks_tot[2][1],
            'W K     [Core]': Ks_tot[2][2],

            'PX K    [Fluid]': Ks_pgrad_tot[0][0],
            'PX K    [Bound]': Ks_pgrad_tot[0][1],
            'PX K    [Core]': Ks_pgrad_tot[0][2],
            'PY K    [Fluid]': Ks_pgrad_tot[1][0],
            'PY K    [Bound]': Ks_pgrad_tot[1][1],
            'PY K    [Core]': Ks_pgrad_tot[1][2],
            'PZ K    [Fluid]': Ks_pgrad_tot[2][0],
            'PZ K    [Bound]': Ks_pgrad_tot[2][1],
            'PZ K    [Core]': Ks_pgrad_tot[2][2],

            'PX R2    [Fluid]': Rs_pgrad_tot[0][0],
            'PX R2    [Bound]': Rs_pgrad_tot[0][1],
            'PX R2    [Core]': Rs_pgrad_tot[0][2],
            'PY R2    [Fluid]': Rs_pgrad_tot[1][0],
            'PY R2    [Bound]': Rs_pgrad_tot[1][1],
            'PY R2    [Core]': Rs_pgrad_tot[1][2],
            'PZ R2    [Fluid]': Rs_pgrad_tot[2][0],
            'PZ R2    [Bound]': Rs_pgrad_tot[2][1],
            'PZ R2    [Core]': Rs_pgrad_tot[2][2],

            'PEAK FLOW INDEX:': peak_flow_idx,
            'Relative error [Fluid] Peak': rel_err[peak_flow_idx][0],
            'Relative error [Bound] Peak': rel_err[peak_flow_idx][1],
            'Relative error [Core] Peak': rel_err[peak_flow_idx][2],
            'Absolute error [Fluid] Peak': abs_err[peak_flow_idx][0],
            'Absolute error [Bound] Peak': abs_err[peak_flow_idx][1],
            'Absolute error [Core] Peak': abs_err[peak_flow_idx][2],
            'Absolute error [Non-F] Peak': abs_err[peak_flow_idx][3],
            'R.M.S. error [Fluid] Peak': rmse[peak_flow_idx][0],
            'R.M.S. error [Bound] Peak': rmse[peak_flow_idx][1],
            'R.M.S. error [Core] Peak': rmse[peak_flow_idx][2],
            'R.M.S. error [Non-F] Peak': rmse[peak_flow_idx][3],

            'VNRMSE [Fluid]': vnrmse[peak_flow_idx,0],
            'VNRMSE [Bound]': vnrmse[peak_flow_idx,1],
            'VNRMSE [Core]':  vnrmse[peak_flow_idx,2],
            'VNRMSE [Non-F]': vnrmse[peak_flow_idx,3],

            'Directional error [Fluid]': d_error[peak_flow_idx,0],
            'Directional error [Bound]': d_error[peak_flow_idx,1],
            'Directional error [Core]':  d_error[peak_flow_idx,2],
            'Directional error [Non-F]': d_error[peak_flow_idx,3],

            'Divergence prediction [Fluid]': div_err[peak_flow_idx,0],
            'Divergence prediction [Bound]': div_err[peak_flow_idx,1],
            'Divergence prediction [Core]':  div_err[peak_flow_idx,2],
            'Divergence prediction [Non-F]': div_err[peak_flow_idx,3],

            'Absolute error Pressure Gradient [Fluid] Peak': grad_abs_err[peak_flow_idx][0],
            'Absolute error Pressure Gradient [Bound] Peak': grad_abs_err[peak_flow_idx][1],
            'Absolute error Pressure Gradient [Core] Peak': grad_abs_err[peak_flow_idx][2],

            'Relative error Pressure Gradient [%] [Fluid] Peak': grad_rel_err[peak_flow_idx][0]*100,
            'Relative error Pressure Gradient [%] [Bound] Peak': grad_rel_err[peak_flow_idx][1]*100,
            'Relative error Pressure Gradient [%] [Core] Peak': grad_rel_err[peak_flow_idx][2]*100,

            'Pressure Gradient NRMSE [%] [Fluid] Peak': grad_nrmse[peak_flow_idx][0]*100,
            'Pressure Gradient NRMSE [%] [Bound] Peak': grad_nrmse[peak_flow_idx][1]*100,
            'Pressure Gradient NRMSE [%] [Core] Peak': grad_nrmse[peak_flow_idx][2]*100,

            'Pressure Gradient Directional Error (deg) [Fluid] Peak': grad_dir_err[peak_flow_idx][0],
            'Pressure Gradient Directional Error (deg) [Bound] Peak': grad_dir_err[peak_flow_idx][1],
            'Pressure Gradient Directional Error (deg) [Core] Peak': grad_dir_err[peak_flow_idx][2],

            'U [Fluid] k': Ks[peak_flow_idx][0][0],
            'U [Bound] k': Ks[peak_flow_idx][0][1],
            'U [Core] k': Ks[peak_flow_idx][0][2],
            'U [Fluid] m': Ms[peak_flow_idx][0][0],
            'U [Bound] m': Ms[peak_flow_idx][0][1],
            'U [Core] m': Ms[peak_flow_idx][0][2],
            'U [Fluid] r^2': Rs[peak_flow_idx][0][0],
            'U [Bound] r^2': Rs[peak_flow_idx][0][1],
            'U [Core] r^2': Rs[peak_flow_idx][0][2],

            'V [Fluid] k': Ks[peak_flow_idx][1][0],
            'V [Bound] k': Ks[peak_flow_idx][1][1],
            'V [Core] k': Ks[peak_flow_idx][1][2],
            'V [Fluid] m': Ms[peak_flow_idx][1][0],
            'V [Bound] m': Ms[peak_flow_idx][1][1],
            'V [Core] m': Ms[peak_flow_idx][1][2],
            'V [Fluid] r^2': Rs[peak_flow_idx][1][0],
            'V [Bound] r^2': Rs[peak_flow_idx][1][1],
            'V [Core] r^2': Rs[peak_flow_idx][1][2],


            'W [Fluid] k': Ks[peak_flow_idx][2][0],
            'W [Bound] k': Ks[peak_flow_idx][2][1],
            'W [Core] k': Ks[peak_flow_idx][2][2],
            'W [Fluid] m': Ms[peak_flow_idx][2][0],
            'W [Bound] m': Ms[peak_flow_idx][2][1],
            'W [Core] m': Ms[peak_flow_idx][2][2],
            'W [Fluid] r^2': Rs[peak_flow_idx][2][0],
            'W [Bound] r^2': Rs[peak_flow_idx][2][1],
            'W [Core] r^2': Rs[peak_flow_idx][2][2],

            'PX [Fluid] k': Ks_pgrad[peak_flow_idx][0][0],
            'PX [Bound] k': Ks_pgrad[peak_flow_idx][0][1],
            'PX [Core] k': Ks_pgrad[peak_flow_idx][0][2],
            'PX [Fluid] m': Ms_pgrad[peak_flow_idx][0][0],
            'PX [Bound] m': Ms_pgrad[peak_flow_idx][0][1],
            'PX [Core] m': Ms_pgrad[peak_flow_idx][0][2],
            'PX [Fluid] r^2': Rs_pgrad[peak_flow_idx][0][0],
            'PX [Bound] r^2': Rs_pgrad[peak_flow_idx][0][1],
            'PX [Core] r^2': Rs_pgrad[peak_flow_idx][0][2],

            'PY [Fluid] k': Ks_pgrad[peak_flow_idx][1][0],
            'PY [Bound] k': Ks_pgrad[peak_flow_idx][1][1],
            'PY [Core] k': Ks_pgrad[peak_flow_idx][1][2],
            'PY [Fluid] m': Ms_pgrad[peak_flow_idx][1][0],
            'PY [Bound] m': Ms_pgrad[peak_flow_idx][1][1],
            'PY [Core] m': Ms_pgrad[peak_flow_idx][1][2],
            'PY [Fluid] r^2': Rs_pgrad[peak_flow_idx][1][0],
            'PY [Bound] r^2': Rs_pgrad[peak_flow_idx][1][1],
            'PY [Core] r^2': Rs_pgrad[peak_flow_idx][1][2],

            'PZ [Fluid] k': Ks_pgrad[peak_flow_idx][2][0],
            'PZ [Bound] k': Ks_pgrad[peak_flow_idx][2][1],
            'PZ [Core] k': Ks_pgrad[peak_flow_idx][2][2],
            'PZ [Fluid] m': Ms_pgrad[peak_flow_idx][2][0],
            'PZ [Bound] m': Ms_pgrad[peak_flow_idx][2][1],
            'PZ [Core] m': Ms_pgrad[peak_flow_idx][2][2],
            'PZ [Fluid] r^2': Rs_pgrad[peak_flow_idx][2][0],
            'PZ [Bound] r^2': Rs_pgrad[peak_flow_idx][2][1],

        }

        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        metrics_filename = f"{ref_directory}/metrics.csv"
        metrics_df.to_csv(metrics_filename, index=False)

        print(stopp)

    # Predict super-resolved data
    if config.predictions.predict_SR_data:

        # Create directory
        SR_directory = f'{results_directory}/SR_data'
        if not os.path.exists(SR_directory):
            os.makedirs(SR_directory)

        # Extract boundaries
        ## if config.plot.expand_mask:
        ##     boundary_mask = compute_outer_boundary_mask(mask)
        ##     mask = mask + boundary_mask

        if config.setup.include_time:
            t_len, x_len, y_len, z_len = u.shape
        else:
            x_len, y_len, z_len = u.shape
            t_len = 1

        t_normalized, x_normalized, y_normalized, z_normalized, standardization_factors = create_and_normalize_coords(config, t_len, x_len, y_len, z_len)

        # Upsample each coordinate
        t_ups = upsample_1d(t_normalized, config.plot.temporal_factor,'extend') if config.setup.include_time else []
        x_ups = upsample_1d(x_normalized, config.plot.spatial_factor, mode='centered')
        y_ups = upsample_1d(y_normalized, config.plot.spatial_factor, mode='centered')
        z_ups = upsample_1d(z_normalized, config.plot.spatial_factor, mode='centered')
        
        if config.setup.include_time:
            grids = np.meshgrid(t_ups, x_ups, y_ups, z_ups, indexing='ij')
        else:
            grids = np.meshgrid(x_ups, y_ups, z_ups, indexing='ij')
        
        flat_coords = [grid.ravel() for grid in grids]
        xyz_ups_full = np.stack(flat_coords, axis=-1) 

        if config.predictions.fluid_region:
            # Upsample mask
            mask_ups = zoom(mask, zoom=config.plot.spatial_factor, order=0, grid_mode=True, mode='nearest')
            mask_ups_flat = np.tile(mask_ups.ravel(), len(t_ups)) if config.setup.include_time else mask_ups.ravel()
            fluid_indices = mask_ups_flat == 1

            xyz_ups = xyz_ups_full[fluid_indices]
        else:
            xyz_ups = xyz_ups_full  
        
        model.eval()

        # ---- Tunable chunk size (points per forward) ----
        # Start conservative; increase if you have headroom.
        CHUNK = 500_000

        # If you want mixed precision for extra headroom (safe for many MLPs):
        USE_AUTOMIXED = True and (DEVICE.type == "cuda") and (not config.training.use_vector_potential)

        N = xyz_ups.shape[0]
        n_out = 6 if (config.setup.include_pressure and config.training.reference_gradients) else 4  # u,v,w,(p|px,py,pz)

        # We’ll collect outputs as numpy chunks to avoid big GPU tensors
        out_chunks = []

        if not config.training.use_vector_potential:
            # No gradients needed: fastest, lowest memory
            ctx = torch.inference_mode()
        else:
            # Need grads only within each chunk for vector potential
            ctx = torch.enable_grad()

        with ctx:
            for s in range(0, N, CHUNK):
                e = min(s + CHUNK, N)
                Xb = torch.from_numpy(xyz_ups[s:e]).to(DEVICE).float()
                if config.training.use_vector_potential:
                    Xb.requires_grad_(True)

                if USE_AUTOMIXED:
                    # Mixed precision for inference path (not for vector potential)
                    from torch.cuda.amp import autocast
                    with autocast(dtype=torch.float16):
                        Yb = model(Xb)
                else:
                    Yb = model(Xb)

                if config.training.use_vector_potential:
                    # Compute curl on this chunk only (keeps graph tiny)
                    Yb = vector_potential_fn(Yb, Xb)

                out_chunks.append(Yb.detach().cpu().numpy())

                # Free asap
                del Xb, Yb
                torch.cuda.empty_cache()

        uvw_pred_ups = np.concatenate(out_chunks, axis=0)
        del out_chunks


        if config.plot.fluid_region:
            uvw_pred_full = np.zeros(((len(xyz_ups_full), len(uvw_pred_ups[0])))) + config.plot.non_fluid_value
            uvw_pred_full[fluid_indices] = uvw_pred_ups
            uvw_pred_ups = uvw_pred_full

        if config.setup.include_time:
            uvw_pred = uvw_pred_ups.reshape(len(t_ups), len(x_ups), len(y_ups), len(z_ups), len(uvw_pred_ups[0]))
        else:
            uvw_pred = uvw_pred_ups.reshape(1, len(x_ups), len(y_ups), len(z_ups), len(uvw_pred_ups[0]))

        u_pred = uvw_pred[:, :, :, :, 0]
        v_pred = uvw_pred[:, :, :, :, 1]
        w_pred = uvw_pred[:, :, :, :, 2]
        p_pred = uvw_pred[:, :, :, :, 3] if config.setup.include_pressure else None
        px_pred = uvw_pred[:, :, :, :, 3] if (config.setup.include_pressure and config.training.reference_gradients) else None
        py_pred = uvw_pred[:, :, :, :, 4] if (config.setup.include_pressure and config.training.reference_gradients) else None
        pz_pred = uvw_pred[:, :, :, :, 5] if (config.setup.include_pressure and config.training.reference_gradients) else None

        # Denormalize predictions
        if config.plot.denormalize:
            if config.vel_normalization == "characteristic":
                u_pred = u_pred*config.constants.U
                v_pred = v_pred*config.constants.U
                w_pred = w_pred*config.constants.U
                if (config.setup.include_pressure and config.training.reference_gradients):
                    _, _, _, std_x, _, std_y, _, std_z = standardization_factors
                    print(std_x, std_y, std_z)

                    px_pred *= config.constants.rho * (config.constants.U ** 2) / config.constants.L / std_x  # px
                    py_pred *= config.constants.rho * (config.constants.U ** 2) / config.constants.L / std_y # py
                    pz_pred *= config.constants.rho * (config.constants.U ** 2) / config.constants.L / std_z # pz
                elif config.setup.include_pressure and not config.training.reference_gradients:
                    p_pred *= config.constants.rho * (config.constants.U ** 2)  # p
                    
            elif config.vel_normalization == "max_velocity":
                u_pred = u_pred*U_max
                v_pred = v_pred*U_max
                w_pred = w_pred*U_max
                p_pred = p_pred*(config.constants.rho*(config.constants.U**2)) if config.setup.include_pressure else None


        # Save SR predictions
        save_to_h5(f"{SR_directory}/healthy-05mm3_SR.h5", "u", u_pred, expand_dim=False)
        save_to_h5(f"{SR_directory}/healthy-05mm3_SR.h5", "v", v_pred, expand_dim=False)
        save_to_h5(f"{SR_directory}/healthy-05mm3_SR.h5", "w", w_pred, expand_dim=False)
        save_to_h5(f"{SR_directory}/healthy-05mm3_SR.h5", "mask_ups", mask_ups, expand_dim=False)
        if (config.setup.include_pressure and not config.training.reference_gradients):
            save_to_h5(f"{SR_directory}/healthy-05mm3_SR.h5", "p", p_pred, expand_dim=False)
        elif config.training.reference_gradients:
            print("Saving pressure gradients to h5...")
            save_to_h5(f"{SR_directory}/healthy-05mm3_SR.h5", "p_x", px_pred, expand_dim=False)
            save_to_h5(f"{SR_directory}/healthy-05mm3_SR.h5", "p_y", py_pred, expand_dim=False)
            save_to_h5(f"{SR_directory}/healthy-05mm3_SR.h5", "p_z", pz_pred, expand_dim=False)



    if config.predictions.compare_noisy_vs_ref:

        assert config.ref_spatial_factor == 1 ## TODO - interpolation option if not the same resolution

        # Create directory
        ref_directory = f'{results_directory}/ref_data'
        if not os.path.exists(ref_directory):
            os.makedirs(ref_directory)

        if not config.setup.include_time:

            if len(u_ref.shape) == 3: 
                u_ref = np.expand_dims(u_ref, axis=0)
                v_ref = np.expand_dims(v_ref, axis=0)
                w_ref = np.expand_dims(w_ref, axis=0)
                p_ref = np.expand_dims(p_ref, axis=0) if config.setup.include_pressure else None

            if len(u.shape) == 3: 
                u = np.expand_dims(u, axis=0)
                v = np.expand_dims(v, axis=0)
                w = np.expand_dims(w, axis=0)
                p = np.expand_dims(p, axis=0) if config.setup.include_pressure else None

        # Get metrics
        peak_flow_idx = config.predictions.peak_flow_idx
        T = len(u)
        nf_mask = 1.0 - mask_ref
        boundary_mask, core_mask = create_boundary_and_core_masks(mask_ref, 0.1, 'voxels')

        X,Y,Z = mask_ref.shape
        cov_a = np.sum(mask_ref)/(X*Y*Z)
        cov_b = np.sum(boundary_mask)/(X*Y*Z)
        cov_c = np.sum(core_mask)/(X*Y*Z)
        ratio_b = np.sum(boundary_mask)/np.sum(mask_ref)
        ratio_c = np.sum(core_mask)/np.sum(mask_ref)

        print(' ')
        print(f'Coverage: {100*cov_a:.3f} %')
        print(f'Boundary --- cov: {100*cov_b:.3f} %, ratio: {100*ratio_b:.3f} %')
        print(f'Core --- cov: {100*cov_c:.3f} %, ratio: {100*ratio_c:.3f} %')

        rel_err = np.zeros((T,3))
        abs_err = np.zeros((T,5))
        rmse = np.zeros((T,5))

        vnrmse = np.zeros((T,4))
        d_error = np.zeros((T,4))
        div_err = np.zeros((T,4))

        Ks = np.zeros((T,3,3))
        Ms = np.zeros((T,3,3))
        Rs = np.zeros((T,3,3))

        for t in range(T):
            rel_err[t,0] = (calculate_relative_error(u[t], v[t], w[t], u_ref[t], v_ref[t], w_ref[t], mask_ref))
            rel_err[t,1] = (calculate_relative_error(u[t], v[t], w[t], u_ref[t], v_ref[t], w_ref[t], boundary_mask))
            rel_err[t,2] = (calculate_relative_error(u[t], v[t], w[t], u_ref[t], v_ref[t], w_ref[t], core_mask))

            abs_err[t,0] = (calculate_absolute_error(u[t], v[t], w[t], u_ref[t], v_ref[t], w_ref[t], mask_ref))
            abs_err[t,1] = (calculate_absolute_error(u[t], v[t], w[t], u_ref[t], v_ref[t], w_ref[t], boundary_mask))
            abs_err[t,2] = (calculate_absolute_error(u[t], v[t], w[t], u_ref[t], v_ref[t], w_ref[t], core_mask))
            abs_err[t,3] = (calculate_absolute_error(u[t], v[t], w[t], u_ref[t], v_ref[t], w_ref[t], nf_mask))
            # abs_err[t,4] = (calculate_absolute_error_pressure(p[t], p_ref[t], mask_ref))

            rmse[t,0] = (calculate_rmse(u[t], v[t], w[t], u_ref[t], v_ref[t], w_ref[t], mask_ref))
            rmse[t,1] = (calculate_rmse(u[t], v[t], w[t], u_ref[t], v_ref[t], w_ref[t], boundary_mask))
            rmse[t,2] = (calculate_rmse(u[t], v[t], w[t], u_ref[t], v_ref[t], w_ref[t], core_mask))
            rmse[t,3] = (calculate_rmse(u[t], v[t], w[t], u_ref[t], v_ref[t], w_ref[t], nf_mask))
            # rmse[t,4] = (calculate_rmse_pressure(p[t], p_ref[t], mask_ref))

            vnrmse[t,0] = (calculate_vnrmse(u[t], v[t], w[t], u_ref[t], v_ref[t], w_ref[t], mask_ref))
            vnrmse[t,1] = (calculate_vnrmse(u[t], v[t], w[t], u_ref[t], v_ref[t], w_ref[t], boundary_mask))
            vnrmse[t,2] = (calculate_vnrmse(u[t], v[t], w[t], u_ref[t], v_ref[t], w_ref[t], core_mask))
            vnrmse[t,3] = (calculate_vnrmse(u[t], v[t], w[t], u_ref[t], v_ref[t], w_ref[t], nf_mask))

            d_error[t,0] = (calculate_directional_error(u[t], v[t], w[t], u_ref[t], v_ref[t], w_ref[t], mask_ref))
            d_error[t,1] = (calculate_directional_error(u[t], v[t], w[t], u_ref[t], v_ref[t], w_ref[t], boundary_mask))
            d_error[t,2] = (calculate_directional_error(u[t], v[t], w[t], u_ref[t], v_ref[t], w_ref[t], core_mask))
            d_error[t,3] = (calculate_directional_error(u[t], v[t], w[t], u_ref[t], v_ref[t], w_ref[t], nf_mask))

            div_err[t,0] = (calculate_divergence([u[t], v[t], w[t]], [config.resolution.dx, config.resolution.dy, config.resolution.dz], mask_ref))
            div_err[t,1] = (calculate_divergence([u[t], v[t], w[t]], [config.resolution.dx, config.resolution.dy, config.resolution.dz], boundary_mask))
            div_err[t,2] = (calculate_divergence([u[t], v[t], w[t]], [config.resolution.dx, config.resolution.dy, config.resolution.dz], core_mask))
            div_err[t,3] = (calculate_divergence([u[t], v[t], w[t]], [config.resolution.dx, config.resolution.dy, config.resolution.dz], nf_mask))

            Ks[t][0][0], Ms[t][0][0], Rs[t][0][0] = linreg(u[t], u_ref[t], mask_ref)
            Ks[t][1][0], Ms[t][1][0], Rs[t][1][0] = linreg(v[t], v_ref[t], mask_ref)
            Ks[t][2][0], Ms[t][2][0], Rs[t][2][0] = linreg(w[t], w_ref[t], mask_ref)

            Ks[t][0][1], Ms[t][0][1], Rs[t][0][1] = linreg(u[t], u_ref[t], boundary_mask)
            Ks[t][1][1], Ms[t][1][1], Rs[t][1][1] = linreg(v[t], v_ref[t], boundary_mask)
            Ks[t][2][1], Ms[t][2][1], Rs[t][2][1] = linreg(w[t], w_ref[t], boundary_mask)

            Ks[t][0][2], Ms[t][0][2], Rs[t][0][2] = linreg(u[t], u_ref[t], core_mask)
            Ks[t][1][2], Ms[t][1][2], Rs[t][1][2] = linreg(v[t], v_ref[t], core_mask)
            Ks[t][2][2], Ms[t][2][2], Rs[t][2][2] = linreg(w[t], w_ref[t], core_mask)
        
        print('Total avg')
        rel_err_tot = np.mean(rel_err, axis=0)
        print(f'Relative error [Fluid] {rel_err_tot[0]:.1f}')
        print(f'Relative error [Bound] {rel_err_tot[1]:.1f}')
        print(f'Relative error [Core] {rel_err_tot[2]:.1f}')

        abs_err_tot = np.mean(abs_err, axis=0)
        print(f'Absolute error [Fluid] {abs_err_tot[0]:.4f}')
        print(f'Absolute error [Bound] {abs_err_tot[1]:.4f}')
        print(f'Absolute error [Core] {abs_err_tot[2]:.4f}')
        print(f'Absolute error [Non-F] {abs_err_tot[3]:.4f}')
        # print(f'Absolute error Pressure [Fluid] {abs_err_tot[4]:.4f}')

        rmse_tot = np.mean(rmse, axis=0)
        print(f'R.M.S.   error [Fluid] {rmse_tot[0]:.4f}')
        print(f'R.M.S.   error [Bound] {rmse_tot[1]:.4f}')
        print(f'R.M.S.   error [Core] {rmse_tot[2]:.4f}')
        print(f'R.M.S.   error [Non-F] {rmse_tot[3]:.4f}')
        # print(f'R.M.S.   error Pressure [Fluid] {rmse_tot[4]:.4f}')

        print(' ')
        print(f'U [Fluid] k: {Ks[peak_flow_idx][0][0]:.4f} \t m: {Ms[peak_flow_idx][0][0]:.4f} \t r^2: {Rs[peak_flow_idx][0][0]:.4f}')
        print(f'  [Bound] k: {Ks[peak_flow_idx][0][1]:.4f} \t m: {Ms[peak_flow_idx][0][1]:.4f} \t r^2: {Rs[peak_flow_idx][0][1]:.4f}')
        print(f'  [Core] k: {Ks[peak_flow_idx][0][2]:.4f} \t m: {Ms[peak_flow_idx][0][2]:.4f} \t r^2: {Rs[peak_flow_idx][0][2]:.4f}')

        print(' ')
        print(f'V [Fluid] k: {Ks[peak_flow_idx][1][0]:.4f} \t m: {Ms[peak_flow_idx][1][0]:.4f} \t r^2: {Rs[peak_flow_idx][1][0]:.4f}')
        print(f'  [Bound] k: {Ks[peak_flow_idx][1][1]:.4f} \t m: {Ms[peak_flow_idx][1][1]:.4f} \t r^2: {Rs[peak_flow_idx][1][1]:.4f}')
        print(f'  [Core] k: {Ks[peak_flow_idx][1][2]:.4f} \t m: {Ms[peak_flow_idx][1][2]:.4f} \t r^2: {Rs[peak_flow_idx][1][2]:.4f}')

        print(' ')
        print(f'W [Fluid] k: {Ks[peak_flow_idx][2][0]:.4f} \t m: {Ms[peak_flow_idx][2][0]:.4f} \t r^2: {Rs[peak_flow_idx][2][0]:.4f}')
        print(f'  [Bound] k: {Ks[peak_flow_idx][2][1]:.4f} \t m: {Ms[peak_flow_idx][2][1]:.4f} \t r^2: {Rs[peak_flow_idx][2][1]:.4f}')
        print(f'  [Core] k: {Ks[peak_flow_idx][2][2]:.4f} \t m: {Ms[peak_flow_idx][2][2]:.4f} \t r^2: {Rs[peak_flow_idx][2][2]:.4f}')


        # Save metrics to csv
        metrics = {
            'Relative error [Fluid]': rel_err_tot[0],
            'Relative error [Bound]': rel_err_tot[1],
            'Relative error [Core]': rel_err_tot[2],

            'Absolute error [Fluid]': abs_err_tot[0],
            'Absolute error [Bound]': abs_err_tot[1],
            'Absolute error [Core]': abs_err_tot[2],
            'Absolute error [Non-F]': abs_err_tot[3],
            # 'Absolute error Pressure [Fluid]': abs_err_tot[4],

            'R.M.S. error [Fluid]': rmse_tot[0],
            'R.M.S. error [Bound]': rmse_tot[1],
            'R.M.S. error [Core]': rmse_tot[2],
            'R.M.S. error [Non-F]': rmse_tot[3],
            # 'R.M.S. error Pressure [Fluid]': rmse_tot[4],

            'VNRMSE [Fluid]': vnrmse[0,0],
            'VNRMSE [Bound]': vnrmse[0,1],
            'VNRMSE [Core]': vnrmse[0,2],
            'VNRMSE [Non-F]': vnrmse[0,3],

            'Directional error [Fluid]': d_error[0,0],
            'Directional error [Bound]': d_error[0,1],
            'Directional error [Core]': d_error[0,2],
            'Directional error [Non-F]': d_error[0,3],

            'Divergence prediction [Fluid]': div_err[0,0],
            'Divergence prediction [Bound]': div_err[0,1],
            'Divergence prediction [Core]': div_err[0,2],
            'Divergence prediction [Non-F]': div_err[0,3],

            'PEAK FLOW INDEX:': peak_flow_idx,
            'Relative error [Fluid] Peak': rel_err[peak_flow_idx][0],
            'Relative error [Bound] Peak': rel_err[peak_flow_idx][1],
            'Relative error [Core] Peak': rel_err[peak_flow_idx][2],
            'Absolute error [Fluid] Peak': abs_err[peak_flow_idx][0],
            'Absolute error [Bound] Peak': abs_err[peak_flow_idx][1],
            'Absolute error [Core] Peak': abs_err[peak_flow_idx][2],
            'Absolute error [Non-F] Peak': abs_err[peak_flow_idx][3],
            'R.M.S. error [Fluid] Peak': rmse[peak_flow_idx][0],
            'R.M.S. error [Bound] Peak': rmse[peak_flow_idx][1],
            'R.M.S. error [Core] Peak': rmse[peak_flow_idx][2],
            'R.M.S. error [Non-F] Peak': rmse[peak_flow_idx][3],

            'U [Fluid] k': Ks[peak_flow_idx][0][0],
            'U [Bound] k': Ks[peak_flow_idx][0][1],
            'U [Core] k': Ks[peak_flow_idx][0][2],
            'U [Fluid] m': Ms[peak_flow_idx][0][0],
            'U [Bound] m': Ms[peak_flow_idx][0][1],
            'U [Core] m': Ms[peak_flow_idx][0][2],
            'U [Fluid] r^2': Rs[peak_flow_idx][0][0],
            'U [Bound] r^2': Rs[peak_flow_idx][0][1],
            'U [Core] r^2': Rs[peak_flow_idx][0][2],

            'V [Fluid] k': Ks[peak_flow_idx][1][0],
            'V [Bound] k': Ks[peak_flow_idx][1][1],
            'V [Core] k': Ks[peak_flow_idx][1][2],
            'V [Fluid] m': Ms[peak_flow_idx][1][0],
            'V [Bound] m': Ms[peak_flow_idx][1][1],
            'V [Core] m': Ms[peak_flow_idx][1][2],
            'V [Fluid] r^2': Rs[peak_flow_idx][1][0],
            'V [Bound] r^2': Rs[peak_flow_idx][1][1],
            'V [Core] r^2': Rs[peak_flow_idx][1][2],

            'W [Fluid] k': Ks[peak_flow_idx][2][0],
            'W [Bound] k': Ks[peak_flow_idx][2][1],
            'W [Core] k': Ks[peak_flow_idx][2][2],
            'W [Fluid] m': Ms[peak_flow_idx][2][0],
            'W [Bound] m': Ms[peak_flow_idx][2][1],
            'W [Core] m': Ms[peak_flow_idx][2][2],
            'W [Fluid] r^2': Rs[peak_flow_idx][2][0],
            'W [Bound] r^2': Rs[peak_flow_idx][2][1],
            'W [Core] r^2': Rs[peak_flow_idx][2][2],
        }

        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        metrics_filename = f"{ref_directory}/metrics_noisyvsref.csv"
        metrics_df.to_csv(metrics_filename, index=False)