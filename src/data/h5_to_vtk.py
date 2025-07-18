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
        #mask = f["mask"][:]

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
            mask_name: binary_mask
        }
    )

if __name__ == "__main__":
    
    # Example usage
    h5_filename = "../../results/250124_Testing/SIREN_1t_VP_20250124-1119/healthy-05mm3.h5"
    output_basename = "vti_files/healthy-05mm3"
    h5_to_vtk(h5_filename, output_basename, index=0)