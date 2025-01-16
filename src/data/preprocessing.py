import numpy as np
import scipy.ndimage as ndimage

def calculate_padding(len_x, divider=8):
    """
        Calculate the necessary padding to make sure the size is divisible by the "divider"
        An additional (divider) size is always added as the padding
    """
    additional_x = divider - (len_x % divider) + divider
    left_x = additional_x // 2
    right_x = additional_x - left_x

    return (left_x, right_x)

def add_padding(img):
    """
        Add padding to a 3D image for all 3 dimensions
    """
    addition = 4
    
    # print("before",img.shape)

    (len_x, len_y, len_z) = img.shape

    padding_x = calculate_padding(len_x)
    padding_y = calculate_padding(len_y)
    padding_z = calculate_padding(len_z)

    # img = np.pad(img, ((addition+5, addition+4),(addition+2,addition+ 2),(addition+5,addition+ 4)), 'constant', constant_values=(np.nan))
    img = np.pad(img, (padding_x, padding_y, padding_z), 'constant', constant_values=(np.nan))
    # print("after", img.shape)

    return img

def supersample3D(img, gauss_sigma = 2):
    """
        Antialiasing using Super sampling
        Zoom up 2x, apply gaussian filter, zoom down 2x to original size
    """
    # zoom-up gauss zoom-down 3D
    new_img = ndimage.zoom(img, 2, order=1)
    new_img = ndimage.filters.gaussian_filter(new_img, sigma=gauss_sigma)
    # TODO: check if there is a bug here or not, the zoom down has shifting issue
    new_img = ndimage.zoom(new_img, 1/2, order=1)
    return new_img

def fsaa_full(velocity, mask):
    """
        Anti alias the whole image for both velocity and mask
    """
    new_velocity = supersample3D(velocity)
    new_mask = supersample3D(mask)

    return new_velocity, new_mask

def fsaa_edge(vx, vy, vz, mask, edge_only=True):
    """
        Anti alias the velocity image only on the edges.
        First we do the FSAA full on both images, and check the difference in the mask image.
        Velocity values on the masked XOR regions will be replaced with the velocity FSAA values on the original velocity image

        Mask must have 0 values on NO SIGNAL regions
    """
    # Make sure there is no NaN values
    vx = np.nan_to_num(vx)
    vy = np.nan_to_num(vy)
    vz = np.nan_to_num(vz)

    # Do supersampling
    new_mask = supersample3D(mask) # we keep this mask
    new_vx = supersample3D(vx)
    new_vy = supersample3D(vy)
    new_vz = supersample3D(vz)

    if edge_only:
        # 1. do an xor on mask vs new mask
        mask_xor = ((mask >0) != (new_mask > 0)) * 1.

        # 2. replace the phase values only in the XOR mask
        vx_edges = (new_vx * mask_xor)
        vy_edges = (new_vy * mask_xor)
        vz_edges = (new_vz * mask_xor)

        # 3. add up the newly interpolated values (edges) with the original velocity image
        new_vx = vx_edges + vx
        new_vy = vy_edges + vy
        new_vz = vz_edges + vz

    # round the values only for 5 decimals
    new_vx = np.round(new_vx, decimals=5)
    new_vy = np.round(new_vy, decimals=5)
    new_vz = np.round(new_vz, decimals=5)

    new_mask = np.round(new_mask, decimals=2)

    return new_vx, new_vy, new_vz, new_mask