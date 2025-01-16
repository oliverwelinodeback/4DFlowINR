# Imports
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
from modulus.sym.utils.io.vtk import grid_to_vtk
from stl import mesh
import os
import h5py
from modulus.sym.geometry.tessellation import Tessellation
# from modulus.geometry.primitives_3d import Plane
from modulus.sym.utils.io.vtk import var_to_polyvtk

def compute_outer_boundary_mask(mask):
    x, y, z = mask.shape
    boundary_mask = np.zeros_like(mask, dtype=bool)
  
    # Check for boundary in each direction
    for i in range(1, x-1):
        for j in range(1, y-1):
            for k in range(1, z-1):
                if mask[i, j, k] == 1:
                    if (mask[i+1, j, k] == 0 or mask[i-1, j, k] == 0 or
                        mask[i, j+1, k] == 0 or mask[i, j-1, k] == 0 or
                        mask[i, j, k+1] == 0 or mask[i, j, k-1] == 0):
                        boundary_mask[i, j, k] = 1
    return boundary_mask

def create_stl_from_mask(mask, output_dir, name):
    print('Create stl file from mask..')
    x, y, z = mask.shape
    t = 1

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for time_step in range(t):
        # Get the mask for the current time step
        current_mask = mask
        
        # Create an empty list to hold vertices and faces
        vertices = []
        faces = []

        # Iterate through the mask to find surface voxels
        for i in range(1, x-1):
            for j in range(1, y-1):
                for k in range(1, z-1):
                    if current_mask[i, j, k]:
                        # Add vertices for the voxel surface
                        voxel_vertices = [
                            [i, j, k],
                            [i+1, j, k],
                            [i+1, j+1, k],
                            [i, j+1, k],
                            [i, j, k+1],
                            [i+1, j, k+1],
                            [i+1, j+1, k+1],
                            [i, j+1, k+1],
                        ]
                        vertices.extend(voxel_vertices)
                        v_index = len(vertices) - 8

                        # Create faces for the voxel surface
                        voxel_faces = [
                            [v_index, v_index+1, v_index+2],
                            [v_index, v_index+2, v_index+3],
                            [v_index+4, v_index+5, v_index+6],
                            [v_index+4, v_index+6, v_index+7],
                            [v_index, v_index+1, v_index+5],
                            [v_index, v_index+5, v_index+4],
                            [v_index+1, v_index+2, v_index+6],
                            [v_index+1, v_index+6, v_index+5],
                            [v_index+2, v_index+3, v_index+7],
                            [v_index+2, v_index+7, v_index+6],
                            [v_index+3, v_index+0, v_index+4],
                            [v_index+3, v_index+4, v_index+7],
                        ]
                        faces.extend(voxel_faces)

        # Convert vertices and faces to NumPy arrays
        vertices = np.array(vertices)
        faces = np.array(faces)

        # Create the mesh
        fluid_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                fluid_mesh.vectors[i][j] = vertices[f[j],:]

        # Save the mesh to an STL file for the current time step
        output_file = os.path.join(output_dir, f"{name}.stl")
        fluid_mesh.save(output_file)

if __name__ == "__main__":

    # Specify data directory
    data_dir = '../../../data'

    # Specify data file

    data_file = '{}/data_05mm_incl_pressure/healthy-05mm3.h5'.format(data_dir)
    name = 'healthy-05mm3_cropped'

    # Load data 
    with h5py.File(data_file, mode = 'r' ) as hf:
        
        mask = np.asarray(hf.get('mask'))
        print(f"Mask shape: {mask.shape}")
        if len(mask.shape) == 4: 
            mask = mask[0]

        mask = mask[120:201, 0:57, 19:69] ## Check coordinates

    # Save ground truth to results directory
    if not os.path.exists("geometries"):
        os.makedirs("geometries")
    
    boundary_mask = compute_outer_boundary_mask(mask)
    create_stl_from_mask(boundary_mask, 'geometries', name)

    # Sample points on boundary
    ### nr_points = 100000
    ### # make tesselated geometry from stl file
    ### geo = Tessellation.from_stl(f"geometries/{name}.stl")
    ### # sample geometry for plotting in Paraview
    ### s = geo.sample_boundary(nr_points=nr_points)
    ### var_to_polyvtk(s, f"geometries/{name}")    