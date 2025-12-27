# from https://github.com/utsav-akhaury/SUNet/blob/main/Deconvolution/deconv_sunet.py
from skimage.transform import resize
import numpy as np
# Utility functions

# Resize image while conserving flux
def resize_conserve_flux(img, size):
        orig_size = img.shape[0]
        img = resize(img, size, anti_aliasing=True)
        return img / (size[0]/orig_size)**2

# Convert impulse response to transfer function
def ir2tf(imp_resp, shape):
    
    dim = 2
    # Zero padding and fill
    irpadded = np.zeros(shape)
    irpadded[tuple([slice(0, s) for s in imp_resp.shape])] = imp_resp
    # Roll for zero convention of the fft to avoid the phase problem. Work with odd and even size.
    for axis, axis_size in enumerate(imp_resp.shape):

        irpadded = np.roll(irpadded,
                        shift=-int(np.floor(axis_size / 2)),
                        axis=axis)

    return np.fft.rfftn(irpadded, axes=range(-dim, 0))

# Laplacian regularization
def laplacian_func(shape):
    
    impr = np.zeros([3,3])
    for dim in range(2):
        idx = tuple([slice(1, 2)] * dim +
                    [slice(None)] +
                    [slice(1, 2)] * (1 - dim))
        impr[idx] = np.array([-1.0,
                            0.0,
                            -1.0]).reshape([-1 if i == dim else 1
                                            for i in range(2)])
    impr[(slice(1, 2), ) * 2] = 4.0
    return ir2tf(impr, shape), impr

# Wiener filter
def wiener(image, psf, balance, laplacian=True):

    r"""Applies Wiener filter to image.
    This function takes an image in the direct space and its corresponding PSF in the
    Fourier space and performs a deconvolution using the Wiener Filter.

    Parameters
    ----------
    image   : 2D TensorFlow tensor
        Image in the direct space.
    psf     : 2D TensorFlow tensor
        PSF in the Fourier space (or K space).
    balance : scalar
        Weight applied to regularization.
    laplacian : boolean
        If true the Laplacian regularization is used else the identity regularization 
        is used.

    Returns
    -------
    tuple
        The first element is the filtered image in the Fourier space.
        The second element is the PSF in the Fourier space (also know as the Transfer
        Function).
    """

    trans_func = psf

    # Compute the regularization
    if laplacian:
        reg = laplacian_func(image.shape)[0]
        if psf.shape != reg.shape:
            trans_func = np.fft.rfft2(np.fft.ifftshift(psf).astype('float32'))  
        else:
            trans_func = psf
    
    arg1 = np.conj(trans_func).astype('complex64')
    arg2 = np.absolute(trans_func).astype('complex64') ** 2
    arg3 = balance
    if laplacian:
        arg3 *= np.absolute(laplacian_func(image.shape)[0]).astype('complex64')**2
    wiener_filter = arg1 / (arg2 + arg3)
    
    # Apply wiener in Fourier (or K) space
    wiener_applied = wiener_filter * np.fft.rfft2(image.astype('float32'))
    wiener_applied = np.fft.irfft2(wiener_applied)
    
    return wiener_applied, trans_func

def deconv_Wiener(noisy,
                  psf,
                  sampling_factor=1,
                  balance = 20.0
                 ):
    
    r"""The Wiener filter is applied to the image in the Fourier space. 
        
    Parameters
    ----------
    noisy   : 4D numpy array
        Noisy image in the direct space. Convention: (samples, channels, height, width).
    psf     : 3D numpy array
        PSF in the direct space. Convention: (channels, height, width).
    sampling_factor : scalar
        Factor by which the PSF is oversampled with respect to the noisy image.
    balance : scalar
        Weight applied to regularization.

    Returns
    -------
    tikho_deconv : 4D numpy array
        The intermediate deconvolved image in the direct space. 
        Output will be of size (samples, channels, height*sampling_factor, width*sampling_factor).
    """

    # Convert PSF to Fourier space
    rfft_psf = np.zeros((psf.shape[0], psf.shape[1], psf.shape[2]//2+1), dtype='complex64')
    for ch in range(psf.shape[0]):
        rfft_psf[ch] = np.fft.rfft2(np.fft.ifftshift(psf[ch]))

    # Tikhonov regularization weight
    ####balance = 20.0 #9e-3

    tikho_deconv = np.zeros((noisy.shape[0], noisy.shape[1], 
                             noisy.shape[2]*sampling_factor, noisy.shape[3]*sampling_factor))
    
    # Perform tikhonov deconvolution
    for i in range(noisy.shape[0]):
        for ch in range(noisy.shape[1]):
            tikho_deconv[i,ch], _ = wiener(resize_conserve_flux(noisy[i,ch],
                                                                (tikho_deconv.shape[2],
                                                                 tikho_deconv.shape[3])), 
                                           rfft_psf[ch], 
                                           balance)
    return tikho_deconv
