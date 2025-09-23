import time
import math
import numpy as np
from scipy.special import i0
from scipy.signal import tukey

def rescale_magnitude_on_ratio(new_mag, old_mag):
    old_mag_flat = np.reshape(old_mag, [-1])
    new_mag_flat = np.reshape(new_mag, [-1])

    rescale_ratio = new_mag_flat.shape[0] / old_mag_flat.shape[0]

    return new_mag * rescale_ratio

def add_complex_signal_noise(imgfft, targetSNR):
    """
        Add gaussian noise to real and imaginary signal
        The sigma is assumed to be the same (Gudbjartsson et al. 1995)

        SNRdb = 10 log SNR
        SNRdb / 10 = log SNR
        SNR = 10 ^ (SNRdb/10)
        
        Pn = Pn / SNR
        Pn = variance
        
        Relation of std and Pn is taken from Matlab Communication Toolbox, awgn.m

        For complex signals, we can use the equation above.
        If we do it in real and imaginary signal, half the variance is in real and other half in in imaginary.
        
        https://www.researchgate.net/post/How_can_I_add_complex_white_Gaussian_to_the_signal_with_given_signal_to_noise_ratio
        "The square of the signal magnitude is proportional to power or energy of the signal.
        SNR is the ratio of this power to the variance of the noise (assuming zero-mean additive WGN).
        Half the variance is in the I channel, and half is in the Q channel.  "

    """    
    add_complex_noise =True
    # adding noise on the real and complex image
    # print("--------------Adding Gauss noise to COMPLEX signal----------------")

    # Deconstruct the complex numbers into real and imaginary
    mag_signal = np.abs(imgfft)
    
    signal_power = np.mean((mag_signal) ** 2)

    noise_power = signal_power / targetSNR

    #print(f"signal power: {signal_power}")
    #print(f"noise power: {noise_power}")

    if add_complex_noise:
        sigma  = np.sqrt(noise_power)
        # print("Adding sigma in complex signal, sigma=", sigma)

        # add the noise to the complex signal directly
        gauss_noise = np.random.normal(0, sigma, imgfft.shape)
        imgfft += gauss_noise
    else:
        # Add the noise to real and imaginary separately
        sigma  = np.sqrt(noise_power/2)
        
        real_signal = np.real(imgfft)
        imj_signal = np.imag(imgfft)
        
        real_noise = np.random.normal(0, sigma, real_signal.shape)
        imj_noise  = np.random.normal(0, sigma, imj_signal.shape)
        
        # add the noise to both components
        real_signal = real_signal + real_noise
        imj_signal  = imj_signal + imj_noise
        
        # reconstruct the image back to complex numbers
        imgfft = real_signal + 1j * imj_signal

    return imgfft

def kaiser_bessel_window(size, beta):
    """
    Generate a 1D Kaiser-Bessel window.
    """
    n = np.arange(size)
    window = i0(beta * np.sqrt(1 - ((2 * n / (size - 1)) - 1) ** 2)) / i0(beta)
    return window

def rectangular_crop3d(f, crop_ratio):
    half_x = f.shape[0] // 2
    half_y = f.shape[1] // 2
    half_z = f.shape[2] // 2
    
    # print('half', half_x, half_y, half_z)

    x_crop = int(half_x * crop_ratio)
    y_crop = int(half_y * crop_ratio)
    z_crop = int(half_z * crop_ratio)

    # shift it to make it easier to crop, otherwise we need to concat half left and half right
    new_kspace = np.fft.fftshift(f)
    new_kspace = new_kspace[half_x-x_crop:half_x+x_crop, half_y-y_crop:half_y+y_crop, half_z-z_crop : half_z+z_crop]
    # shift it back to original freq domain
    new_kspace = np.fft.fftshift(new_kspace)
     
    return new_kspace

def rectangular_crop3d_arctan(f, crop_ratio, steepness=50, cutoff=0.95):
    """
    Apply a 3D custom arctan-based radial filter after cropping in k-space.
    """
    # Define the crop center and size
    half_x, half_y, half_z = f.shape[0] // 2, f.shape[1] // 2, f.shape[2] // 2
    x_crop, y_crop, z_crop = int(half_x * crop_ratio), int(half_y * crop_ratio), int(half_z * crop_ratio)

    # Shift to center the k-space data for cropping
    new_kspace = np.fft.fftshift(f)
    cropped_kspace = new_kspace[half_x - x_crop:half_x + x_crop, 
                                 half_y - y_crop:half_y + y_crop, 
                                 half_z - z_crop:half_z + z_crop]

    # Generate the arctan-based filter
    x_vals = np.linspace(-1, 1, 2 * x_crop)
    y_vals = np.linspace(-1, 1, 2 * y_crop)
    z_vals = np.linspace(-1, 1, 2 * z_crop)
    
    kx, ky, kz = np.meshgrid(x_vals, y_vals, z_vals, indexing="ij")
    radius = np.sqrt(kx**2 + ky**2 + kz**2)
    
    # Create the arctan filter
    arctan_filter = 0.5 + np.arctan(steepness * (cutoff - radius)) / np.pi

    # Apply the arctan filter to the cropped k-space data
    cropped_kspace *= arctan_filter

    # Shift it back to the original frequency domain layout
    cropped_kspace = np.fft.fftshift(cropped_kspace)
    
    return cropped_kspace

def rectangular_crop3d_hamming(f, crop_ratio):
    # Define the crop center and size
    half_x, half_y, half_z = f.shape[0] // 2, f.shape[1] // 2, f.shape[2] // 2
    x_crop, y_crop, z_crop = int(half_x * crop_ratio), int(half_y * crop_ratio), int(half_z * crop_ratio)

    # Shift to center the k-space data for cropping
    new_kspace = np.fft.fftshift(f)
    cropped_kspace = new_kspace[half_x - x_crop:half_x + x_crop, half_y - y_crop:half_y + y_crop, half_z - z_crop:half_z + z_crop]

    # Create a 3D Hamming window with the shape of the cropped region
    hamming_x = np.hamming(2 * x_crop)
    hamming_y = np.hamming(2 * y_crop)
    hamming_z = np.hamming(2 * z_crop)
    hamming_3d = np.outer(hamming_x, hamming_y).reshape(2 * x_crop, 2 * y_crop, 1) * hamming_z.reshape(1, 1, 2 * z_crop)

    # Apply the Hamming window to the cropped k-space data
    cropped_kspace *= hamming_3d

    # Shift it back to the original frequency domain layout
    cropped_kspace = np.fft.fftshift(cropped_kspace)
    
    return cropped_kspace

def rectangular_crop3d_tukey(f, crop_ratio, alpha=0.5):
    """
    Apply a 3D Tukey filter after cropping in k-space.
    
    Args:
        f (np.ndarray): Input 3D k-space data.
        crop_ratio (float): Fraction of k-space to retain (0 to 1).
        alpha (float): Fraction of the Tukey window that is tapered (0 <= alpha <= 1).
    
    Returns:
        np.ndarray: Filtered k-space data.
    """
    # Define the crop center and size
    half_x, half_y, half_z = f.shape[0] // 2, f.shape[1] // 2, f.shape[2] // 2
    x_crop, y_crop, z_crop = int(half_x * crop_ratio), int(half_y * crop_ratio), int(half_z * crop_ratio)

    # Shift to center the k-space data for cropping
    new_kspace = np.fft.fftshift(f)
    cropped_kspace = new_kspace[half_x - x_crop:half_x + x_crop, 
                                 half_y - y_crop:half_y + y_crop, 
                                 half_z - z_crop:half_z + z_crop]

    # Create a 3D Tukey window with the shape of the cropped region
    tukey_x = tukey(2 * x_crop, alpha)
    tukey_y = tukey(2 * y_crop, alpha)
    tukey_z = tukey(2 * z_crop, alpha)
    tukey_3d = np.outer(tukey_x, tukey_y).reshape(2 * x_crop, 2 * y_crop, 1) * tukey_z.reshape(1, 1, 2 * z_crop)

    # Apply the Tukey window to the cropped k-space data
    cropped_kspace *= tukey_3d

    # Shift it back to the original frequency domain layout
    cropped_kspace = np.fft.fftshift(cropped_kspace)
    
    return cropped_kspace

def rectangular_crop3d_kaiser(f, crop_ratio, beta):
    """
    Apply a 3D Kaiser-Bessel filter after cropping in k-space.
    """
    # Define the crop center and size
    half_x, half_y, half_z = f.shape[0] // 2, f.shape[1] // 2, f.shape[2] // 2
    x_crop, y_crop, z_crop = int(half_x * crop_ratio), int(half_y * crop_ratio), int(half_z * crop_ratio)

    # Shift to center the k-space data for cropping
    new_kspace = np.fft.fftshift(f)
    cropped_kspace = new_kspace[half_x - x_crop:half_x + x_crop, 
                                 half_y - y_crop:half_y + y_crop, 
                                 half_z - z_crop:half_z + z_crop]

    # Create a 3D Kaiser-Bessel window with the shape of the cropped region
    kaiser_x = kaiser_bessel_window(2 * x_crop, beta)
    kaiser_y = kaiser_bessel_window(2 * y_crop, beta)
    kaiser_z = kaiser_bessel_window(2 * z_crop, beta)
    kaiser_3d = np.outer(kaiser_x, kaiser_y).reshape(2 * x_crop, 2 * y_crop, 1) * kaiser_z.reshape(1, 1, 2 * z_crop)

    # Apply the Kaiser-Bessel window to the cropped k-space data
    cropped_kspace *= kaiser_3d

    # Shift it back to the original frequency domain layout
    cropped_kspace = np.fft.fftshift(cropped_kspace)
    
    return cropped_kspace

def rectangular_crop(f, crop_ratio):
    half_x = f.shape[0] // 2
    half_y = f.shape[1] // 2
    
    x_crop = int(half_x * crop_ratio)
    y_crop = int(half_y * crop_ratio)

    # shift it to make it easier to crop, otherwise we need to concat half left and half right
    new_kspace = np.fft.fftshift(f)
    new_kspace = new_kspace[half_x-x_crop:half_x+x_crop, half_y-y_crop:half_y+y_crop]
    # shift it back to original freq domain
    new_kspace = np.fft.fftshift(new_kspace)

    return new_kspace

def downsample_complex_img(complex_img, crop_ratio, targetSNR, k_space_filter):
    #mag_signal_imspace = np.abs(complex_img)
    #signal_power = np.mean((mag_signal_imspace) ** 2)
    #noise_power = signal_power / targetSNR
    #print(f"signal power imspace: {signal_power}")
    #print(f"noise power imspace: {noise_power}")

    imgfft = np.fft.fftn(complex_img)

    if crop_ratio == 1:
        print("No downsample")
    else:
        if imgfft.ndim == 3:
            # print("Downsample 3D")
            if k_space_filter:
                #imgfft = rectangular_crop3d_hamming(imgfft, crop_ratio)
                imgfft = rectangular_crop3d_kaiser(imgfft, crop_ratio, beta=2)
                #imgfft = rectangular_crop3d_arctan(imgfft, crop_ratio, steepness=50, cutoff=0.99)
                #imgfft = rectangular_crop3d_tukey(imgfft, crop_ratio, alpha=0.2)

            else:
                imgfft = rectangular_crop3d(imgfft, crop_ratio)

        else:
            # print("Downsample 2D")
            imgfft = rectangular_crop(imgfft, crop_ratio)

    shifted_mag  = 20*np.log(np.fft.fftshift(np.abs(imgfft)))

    if targetSNR is not None:
        # print("adding noise SNR", targetSNR)
        # add noise on freq domain
        imgfft = add_complex_signal_noise(imgfft, targetSNR)
    else:
        print("No noise added.")

    # inverse fft to image domain
    new_complex_img = np.fft.ifftn(imgfft)

    return new_complex_img, shifted_mag

def get_absmax(u):
    # get which one is bigger, max or min
    vmax = max(u.min(), u.max(), key=abs)
    # make sure we get the 'bigger' max
    vmax = abs(vmax)
    
    return vmax

def downsample_phase_img(velocity_img, mag_image, venc, crop_ratio, targetSNR, k_space_filter=False):

    #print(f"TSNR: {targetSNR}")
    
    # convert to phase
    phase_image = velocity_img / venc * np.pi

    complex_img = np.multiply(mag_image, np.exp(1j*phase_image))
    
    # -----------------------------------------------------------
    new_complex_img, shifted_freqmag = downsample_complex_img(complex_img, crop_ratio, targetSNR, k_space_filter)
    # -----------------------------------------------------------

    # Get the MAGnitude and rescale
    new_mag = np.abs(new_complex_img)
    new_mag = rescale_magnitude_on_ratio(new_mag, mag_image)

    # Get the PHASE
    new_phase = np.angle(new_complex_img)
    
    # Get the velocity image
    new_velocity_img = new_phase / np.pi * venc

    return new_velocity_img, new_mag