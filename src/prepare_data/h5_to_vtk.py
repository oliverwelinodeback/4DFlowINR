import numpy as np
import h5py
from pyevtk.hl import imageToVTK

def h5_to_vtk(h5_filename, output_basename="velocity_field", index=25):
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

    # Read the HDF5 file
    with h5py.File(h5_filename, 'r') as f:
        # Assuming datasets named 'u', 'v', 'w', 'mask'
        u = f["u"][index] 
        v = f["v"][index]
        w = f["w"][index]
        mask = f["mask"][:]


    # Read the HDF5 file
    ## with h5py.File(h5_filename, 'r') as f:
    ##     # Assuming datasets named 'u', 'v', 'w', 'mask'
    ##     u = np.asarray(f["u"])#[:] 
    ##     v = np.asarray(f["v"])#[:]
    ##     w = np.asarray(f["w"])#[:]
    ##     mask = f["mask"][:]


    # Decide on the image origin and spacing
    origin = (0.0, 0.0, 0.0)
    spacing = (1.0, 1.0, 1.0)

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
            #mask: mask
        }
    )

if __name__ == "__main__":
    
    # Example usage
    #h5_filename = "../../data/healthy/HV01_newT_05mm3_20ms_TEST4.h5"
    #output_basename = "vti_files/HV01_newT_05mm3_20ms_TEST4"

    h5_filename = "../../data/healthy/HV01_05mm3_20ms_REARRANGEDCORRECT.h5"
    output_basename = "vti_files/HV01_05mm3_20ms_REARRANGEDCORRECT"

    h5_to_vtk(h5_filename, output_basename, index=1)