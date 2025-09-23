import h5py
import numpy as np
from skimage import measure
from stl import mesh

# Load the HDF5 file & extract mask
data_dir = '../../data/data_05mm_incl_pressure'
hdf5_file_path = '{}/healthy-05mm3.h5'.format(data_dir)

with h5py.File(hdf5_file_path, 'r') as f:
    binary_mask = f['mask'][:]
    if len(binary_mask.shape) == 4:
        binary_mask = binary_mask[0]
    binary_mask = binary_mask[120:201, 0:57, 19:69]
    #binary_mask = binary_mask[80:201, 0:100, 19:100]

    print(binary_mask.shape)

# Generate surface mesh using marching cubes
verts, faces, normals, _ = measure.marching_cubes(binary_mask, level=0.5)

# Create mesh object
fluid_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

for i, face in enumerate(faces):
    for j in range(3):
        fluid_mesh.vectors[i][j] = verts[face[j], :]

# mesh to STL
output_stl_file = 'geometries/healthy-05mm3_geom2.stl'
fluid_mesh.save(output_stl_file)

print(f'Successfully saved the STL file to {output_stl_file}')