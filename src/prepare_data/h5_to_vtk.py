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
                    px = f["px"][i]/1000 
                    py = f["py"][i]/1000
                    pz = f["pz"][i]/1000

            print(u.shape)
            print(px.shape)

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
            u = f["u"][index] 
            v = f["v"][index]
            w = f["w"][index]
            mask = f["mask"][:] if "mask" in f else np.ones_like(u)

            if gradients:
                px = f["px"][index] 
                py = f["py"][index]
                pz = f["pz"][index]


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
                #mask: _to_f32(mask)
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

    h5_filename = "../../data/healthy/HV01_05mm3_20ms_LR_dv_tSNR8.h5"
    output_basename = "vti_files/HV01_05mm3_20ms_LR_dv_tSNR8"

    #h5_filename = "../../results/250924_antialias/HV01_SIREN_momentum_antialiased_08_20250926-1005/ref_data/healthy-05mm3_SR.h5"
    #output_basename = "vti_files/healthy-05mm3_SR"

    h5_to_vtk(h5_filename, output_basename, index='all', gradients=True)