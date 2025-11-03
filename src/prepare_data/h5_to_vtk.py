import numpy as np
import h5py
from pyevtk.hl import imageToVTK

def h5_to_vtk(h5_filename, output_basename="velocity_field", index=25, gradients=True):
    """
    Convert an HDF5 file containing 3D velocity components u, v, w and
    a mask to a VTK image data (.vti) file readable by ParaView.

    Parameters:
    -----------
    h5_filename : str
        Path to the input .h5 file.
    output_basename : str
        Base name (without extension) for the output VTK file.
    """
    
    # Decide on the image origin and spacing
    origin = (0.0, 0.0, 0.0)
    spacing = (1.0, 1.0, 1.0)
    
    if index == "all":
        # Read the HDF5 file
        with h5py.File(h5_filename, 'r') as f:
            T = len(np.squeeze(np.asarray(f["u"])))
            print(T)
        
        for i in range(T):
            with h5py.File(h5_filename, 'r') as f:
                # Assuming datasets named 'u', 'v', 'w', 'mask'
                u = f["u"][i] 
                v = f["v"][i]
                w = f["w"][i]
                mask = f["mask"][:] if "mask" in f else np.ones_like(u)
                binary_mask = (mask != 0).astype(np.uint8) if mask is not None else np.ones_like(u)
                if len(binary_mask.shape) == 4:
                    binary_mask = binary_mask[0]
                if gradients:
                    px = f["px"][i]*1000 
                    py = f["py"][i]*1000
                    pz = f["pz"][i]*1000

            print(u.shape)
            #print(px.shape)

            imageToVTK(
                f"{output_basename}_t{i:02d}",
                origin=origin,
                spacing=spacing,
                pointData={},  
                cellData={
                    "velocity": (u, v, w),
                    "pressure": (px, py, pz) if gradients else (np.zeros_like(u), np.zeros_like(v), np.zeros_like(w)),
                    "mask": binary_mask
                }
            )
    else:
        # Read the HDF5 file
        with h5py.File(h5_filename, 'r') as f:
            # Assuming datasets named 'u', 'v', 'w', 'mask'
            u = f["u"][0][index] 
            v = f["v"][0][index]
            w = f["w"][0][index]
            mask_name = f["mask"][:] if "mask" in f else np.ones_like(u)
            if len(mask_name.shape) == 4:
                mask_name = mask_name[0]
            binary_mask = (mask_name != 0).astype(np.uint8) if mask_name is not None else np.ones_like(u)

            if gradients:
                px = f["p_x"][0][index]#*1000
                py = f["p_y"][0][index]#*1000
                pz = f["p_z"][0][index]#*1000

        # Read the HDF5 file
        ## with h5py.File(h5_filename, 'r') as f:
        ##     # Assuming datasets named 'u', 'v', 'w', 'mask'
        ##     u = np.asarray(f["u"])#[:] 
        ##     v = np.asarray(f["v"])#[:]
        ##     w = np.asarray(f["w"])#[:]
        ##     mask = f["mask"][:]


        # imageToVTK(
        #     output_basename,
        #     origin=origin,
        #     spacing=spacing,
        #     pointData={
        #         "velocity": (u, v, w),
        #         #"mask": mask  
        #     }
        # )
        imageToVTK(
            output_basename,
            origin=origin,
            spacing=spacing,
            pointData={},  
            cellData={
                "velocity": (u, v, w),
                "pressure": (px, py, pz) if gradients else (np.zeros_like(u), np.zeros_like(v), np.zeros_like(w)),
                "mask": binary_mask
            }
        )

if __name__ == "__main__":
    
    # Example usage
    #h5_filename = "../../data/healthy/HV01_newT_05mm3_20ms_TEST4.h5"
    #output_basename = "vti_files/HV01_newT_05mm3_20ms_TEST4"

    #h5_filename = "../../data/healthy/HV01_05mm3_20ms_REARRANGEDCORRECT.h5"
    #output_basename = "vti_files/HV01_05mm3_20ms_REARRANGEDCORRECT"

    #h5_filename = "../../data/healthy/HV01_05mm3_20ms.h5"
    #output_basename = "vti_files/HV01_05mm3_20ms"

    #h5_filename = "../../data/healthy/HV01_05mm3_20ms_LR_dv_tSNR8.h5"
    #output_basename = "vti_files/HV01_05mm3_20ms_LR_dv_tSNR8"

    #h5_filename = "../../results/250924_antialias/HV01_SIREN_momentum_antialiased_08_20250926-1005/ref_data/healthy-05mm3_SR.h5"
    #output_basename = "vti_files/healthy-05mm3_SR"

    #h5_filename = "../../data/stenosis_70/ICAD21_05mm3_20ms.h5"
    #output_basename = "vti_files/ICAD21/HR/ICAD21_05mm3_20ms"

    #h5_filename = "../../models/251020_WIRE_CMPLX_abstract1_ICAD/ICAD21_WIRE_CMPLX_datahv17_losscos_20251022-0930/SR_final.h5"
    #output_basename = "vti_files/ICAD21/SR/ICAD21_05mm3_20ms_SR_dv_hv26_tSNR8"

    #h5_filename = "../../models/251020_WIRE_CMPLX_abstract1/HV01_WIRE_REAL_momentum_datahv17_losscos_20251020-1753/SR_final.h5"
    #output_basename = "vti_files/HV01/SR_2/HV01_05mm3_20ms_SR_dv_hv17_tSNR8"

    #h5_filename = "../../models/251020_WIRE_CMPLX_abstract1_ICAD48/ICAD48_WIRE_CMPLX_dataICAD48_hv13_20251028-0911/SR_final.h5"
    #output_basename = "vti_files/ICAD48/SR/ICAD48_05mm3_20ms_SR_dv_hv13_tSNR8"

    #h5_filename = "../../data/stenosis_70/ICAD17_05mm3_20ms_LR_dv_hv41_tSNR8.h5"
    #output_basename = "vti_files/ICAD17/LR/ICAD17_05mm3_20ms_LR_dv_hv41_tSNR8"

    #h5_filename = "../../models/251020_WIRE_CMPLX_abstract1_ICAD17/ICAD17_WIRE_CMPLX_dataICAD17_hv41_U1_20251028-0946/SR_final.h5"
    #output_basename = "vti_files/ICAD17/SR/ICAD17_05mm3_20ms_SR_dv_hv41_tSNR8"

    #h5_filename = "../../data/stenosis_50/ICAD48_05mm3_20ms_LR_dv_hv13_tSNR8.h5"
    #output_basename = "vti_files/ICAD48/LR/ICAD48_05mm3_20ms_LR_dv_hv13_tSNR8"

    #h5_filename = "../../data/healthy/HV01_05mm3_20ms.h5"
    #output_basename = "vti_files/HR_peak/HV01_05mm3_20ms_t12"
    #index = 12
    #h5_filename = "../../data/healthy/HV03_05mm3_20ms.h5"
    #output_basename = "vti_files/HR_peak/HV03_05mm3_20ms_t4"
    #index = 4
    #h5_filename = "../../data/healthy/HV06_05mm3_20ms.h5"
    #output_basename = "vti_files/HR_peak/HV06_05mm3_20ms_t2"
    #index = 2    
    #h5_filename = "../../data/stenosis_50/ICAD28_05mm3_20ms.h5"
    #output_basename = "vti_files/HR_peak/ICAD28_05mm3_20ms_t2"
    #index = 2
    #h5_filename = "../../data/stenosis_50/ICAD48_05mm3_20ms.h5"
    #output_basename = "vti_files/HR_peak/ICAD48_05mm3_20ms_t14"
    #index = 14
    #h5_filename = "../../data/stenosis_50/ICAD98_05mm3_20ms.h5"
    #output_basename = "vti_files/HR_peak/ICAD98_05mm3_20ms_t12"
    #index = 12    
    #h5_filename = "../../data/stenosis_70/ICAD17_05mm3_20ms.h5"
    #output_basename = "vti_files/HR_peak/ICAD17_05mm3_20ms_t8"
    #index = 8
    #h5_filename = "../../data/stenosis_70/ICAD21_05mm3_20ms.h5"
    #output_basename = "vti_files/HR_peak/ICAD21_05mm3_20ms_t12"
    #index = 12
    #h5_filename = "../../data/stenosis_70/ICAD146_05mm3_20ms.h5"
    #output_basename = "vti_files/HR_peak/ICAD146_05mm3_20ms_t8"
    #index = 8    

    #h5_filename = "../../results/251031_WIRE_MOMENTUM_ALL/WIRE_MOMENTUM_ALL_HV01_hv17_20251031-1422/ref_data/healthy-05mm3_SR.h5"
    #output_basename = "vti_files/SR_mom_10000it_peak/HV01_05mm3_20ms_t12"
    #index = 12
    #h5_filename = "../../results/251031_WIRE_MOMENTUM_ALL/WIRE_MOMENTUM_ALL_HV03_hv13_20251031-1914/ref_data/healthy-05mm3_SR.h5"
    #output_basename = "vti_files/SR_mom_10000it_peak/HV03_05mm3_20ms_t4"
    #index = 4
    #h5_filename = "../../results/251031_WIRE_MOMENTUM_ALL/WIRE_MOMENTUM_ALL_HV06_hv12_20251101-0006/ref_data/healthy-05mm3_SR.h5"
    #output_basename = "vti_files/SR_mom_10000it_peak/HV06_05mm3_20ms_t2" 
    #index = 2    
    #h5_filename = "../../results/251031_WIRE_MOMENTUM_ALL/WIRE_MOMENTUM_ALL_ICAD28_hv13_20251101-0411/ref_data/healthy-05mm3_SR.h5"
    #output_basename = "vti_files/SR_mom_10000it_peak/ICAD28_05mm3_20ms_t2"
    #index = 2
    #h5_filename = "../../results/251031_WIRE_MOMENTUM_ALL/WIRE_MOMENTUM_ALL_ICAD48_hv13_20251031-1423/ref_data/healthy-05mm3_SR.h5"
    #output_basename = "vti_files/SR_mom_10000it_peak/ICAD48_05mm3_20ms_t14"
    #index = 14
    #h5_filename = "../../results/251031_WIRE_MOMENTUM_ALL/WIRE_MOMENTUM_ALL_ICAD98_hv51_20251031-1918/ref_data/healthy-05mm3_SR.h5"
    #output_basename = "vti_files/SR_mom_10000it_peak/ICAD98_05mm3_20ms_t12" 
    #index = 12    
    #h5_filename = "../../results/251031_WIRE_MOMENTUM_ALL/WIRE_MOMENTUM_ALL_ICAD17_hv41_20251101-0008/ref_data/healthy-05mm3_SR.h5"
    #output_basename = "vti_files/SR_mom_10000it_peak/ICAD17_05mm3_20ms_t8"
    #index = 8
    #h5_filename = "../../results/251031_WIRE_MOMENTUM_ALL/WIRE_MOMENTUM_ALL_ICAD21_hv26_20251101-0504/ref_data/healthy-05mm3_SR.h5"
    #output_basename = "vti_files/SR_mom_10000it_peak/ICAD21_05mm3_20ms_t12"
    #index = 12
    h5_filename = "../../results/251031_WIRE_MOMENTUM_ALL/WIRE_MOMENTUM_ALL_ICAD146_hv17_20251101-0710/ref_data/healthy-05mm3_SR.h5"
    output_basename = "vti_files/SR_mom_10000it_peak/ICAD146_05mm3_20ms_t8" ########
    index = 8    

    #index = 'all'
    h5_to_vtk(h5_filename, output_basename, index=index, gradients=True)

    #x_oliwe/SRFlowNIR/models/251020_WIRE_CMPLX_abstract1_ICAD/ICAD21_WIRE_CMPLX_datahv17_losscos_20251022-0930/SR_final.h5