import numpy as np
import json
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.fft import fft2, ifft2, fftshift, ifftshift

import galsim
##############
# Galaxy & Image utils
##############

def rescale_image_range(im, max_I=1.0, min_I=-1.0):
    """ rescale pixel values to match [min_I,max_I] range """
    temp = (im - im.min()) / ((im.max() - im.min()))
    return temp * (max_I - min_I) + min_I


def down_sample(input, rate=4):
    """Downsample the input image with a factor of 4 using an average filter.

    Args:
        input (`torch.Tensor`): The input image with shape `[H, W]`.
        rate (`int`, optional): Downsampling rate. Defaults to `4`.

    Returns:
        `torch.Tensor`: The downsampled image.
    """
    weight = torch.ones([1, 1, rate, rate]) / (rate**2)  # Average filter.
    input = input.unsqueeze(0).unsqueeze(0)
    output = F.conv2d(input=input, weight=weight, stride=rate).squeeze(0).squeeze(0)

    return output


def get_Obs_PSF(
    lam_over_diam,
    opt_defocus,
    opt_c1,
    opt_c2,
    opt_a1,
    opt_a2,
    opt_obscuration,
    atmos_fwhm,
    atmos_e,
    atmos_beta,
    spher,
    trefoil1,
    trefoil2,
    g1_err=0,
    g2_err=0,
    fov_pixels=48,
    pixel_scale=0.03,
    upsample=4,
    skip_atmos=False
):
    """Simulate a PSF from a ground-based observation. The PSF consists of an optical component and an atmospheric component.

    Args:
        lam_over_diam (float): Wavelength over diameter of the telescope.
        opt_defocus (float): Defocus in units of incident light wavelength.
        opt_c1 (float): Coma along y in units of incident light wavelength.
        opt_c2 (float): Coma along x in units of incident light wavelength.
        opt_a1 (float): Astigmatism (like e2) in units of incident light wavelength.
        opt_a2 (float): Astigmatism (like e1) in units of incident light wavelength.
        opt_obscuration (float): Linear dimension of central obscuration as fraction of pupil linear dimension, [0., 1.).
        atmos_fwhm (float): The full width at half maximum of the Kolmogorov function for atmospheric PSF.
        atmos_e (float): Ellipticity of the shear to apply to the atmospheric component.
        atmos_beta (float): Position angle (in radians) of the shear to apply to the atmospheric component, twice the phase of a complex valued shear.
        spher (float): Spherical aberration in units of incident light wavelength.
        trefoil1 (float): Trefoil along y axis in units of incident light wavelength.
        trefoil2 (float): Trefoil along x axis in units of incident light wavelength.
        g1_err (float, optional): The first component of extra shear applied to the overall PSF to simulated a erroneously estimated PSF. Defaults to `0`.
        g2_err (float, optional): The second component of extra shear applied to the overall PSF to simulated a erroneously estimated PSF. Defaults to `0`.
        fov_pixels (int, optional): Width of the simulated images in pixels. Defaults to `48`.
        pixel_scale (float, optional): Pixel scale of the simulated image determining the resolution. Defaults to `0.2`.
        upsample (int, optional): Upsampling factor for the PSF image. Defaults to `4`.

    Returns:
        `torch.Tensor`: Simulated PSF image with shape `(fov_pixels*upsample, fov_pixels*upsample)`.
    """

    # Atmospheric PSF
    atmos = galsim.Kolmogorov(fwhm=atmos_fwhm, flux=1)
    atmos = atmos.shear(e=atmos_e, beta=atmos_beta * galsim.radians)

    # Optical PSF
    optics = galsim.OpticalPSF(
        lam_over_diam,
        defocus=opt_defocus,
        coma1=opt_c1,
        coma2=opt_c2,
        astig1=opt_a1,
        astig2=opt_a2,
        spher=spher,
        trefoil1=trefoil1,
        trefoil2=trefoil2,
        obscuration=opt_obscuration,
        flux=1,
    )
    # Convolve the two components.
    if skip_atmos:
        psf=optics
    else:
        psf = galsim.Convolve([atmos, optics])


    # Shear the overall PSF to simulate a erroneously estimated PSF when necessary.
    psf = psf.shear(g1=g1_err, g2=g2_err)

    # Draw PSF images.
    psf_image = galsim.ImageF(fov_pixels * upsample, fov_pixels * upsample)
    psf.drawImage(psf_image, scale=pixel_scale / upsample, method="auto")
    psf_image = torch.from_numpy(psf_image.array)

    return psf_image


def get_Galaxy_img(
    gal_orig,
    psf_hst,
    gal_g,
    gal_beta,
    gal_mu,
    dx,
    dy,
    fov_pixels=48,
    pixel_scale=0.03,
    upsample=4,
    theta=0.0,
):
    """Simulate a background galaxy with data from COSMOS Catalog.

    Args:
        gal_orig: original COSMOS galaxy (type galsim.real.RealGalaxy)
        psf_hst: associated HST PSF to reconvolved the gal_orig (type  galsim.interpolatedimage.InterpolatedImage)
        gal_g (float): The shear to apply.
        gal_beta (float): Position angle (in radians) of the shear to apply, twice the phase of a complex valued shear.
        gal_mu (float): The lensing magnification to apply.
        fov_pixels (int, optional): Width of the simulated images in pixels. Defaults to `48`.
        pixel_scale (float, optional): Pixel scale of the simulated image determining the resolution. Defaults to `0.2`.
        upsample (int, optional): Upsampling factor for galaxy image. Defaults to `4`.
        theta (float): Rotation angle of the galaxy (in radians, positive means anticlockwise). Defaults 0].

    Returns:
        `torch.Tensor`: Simulated galaxy image of shape `(fov_pixels*upsample, fov_pixels*upsample)`.
    """
    # Add random rotation, shear, and magnification.
    gal = gal_orig.rotate(theta * galsim.radians)  # Rotate by a random angle
    gal = gal_orig.shear(g=gal_g, beta=gal_beta * galsim.radians)  # Apply the desired shear
    gal = gal.magnify(gal_mu) # Also apply a magnification mu = ( (1-kappa)^2 - |gamma|^2 )^-1, this conserves surface brightness, so it scales both the area and flux.

    # Draw galaxy image.
    gal_image = galsim.ImageF(fov_pixels * upsample, fov_pixels * upsample)
    gal = galsim.Convolve([psf_hst, gal])  # Concolve wth original PSF of HST.
    gal.drawImage(
        gal_image, scale=pixel_scale / upsample, offset=(dx, dy), method="auto"
    )

    gal_image_arr = rescale_image_range(gal_image.array)    #  rescale in the range [-1,1] (default)
    gal_image = torch.from_numpy(gal_image_arr)  # Convert to PyTorch.Tensor.

    return gal_image

####
# generators
###

def gener_PSF_obs(args):
    # Atmospheric PSF
     # Atmospheric seeing (arcsec), the FWHM of the Kolmogorov function.
    atmos_fwhm = args.atmos_fwhm * args.rng()
    # Ellipticity of atmospheric PSF (magnitude of the shear in the “distortion” definition)
    atmos_e = args.atmos_e * (1.0+ 2.0 * args.rng())
    atmos_beta = 2.0 * np.pi * args.rng()  # Shear position angle (radians), N(0,2*pi).

    # Optical PSF
    opt_defocus = args.rng_defocus()  # Defocus (wavelength)
    opt_a1 = args.rng_gaussian()  # Astigmatism (like e2) (wavelength)
    opt_a2 = args.rng_gaussian()  # Astigmatism (like e1) (wavelength)
    opt_c1 = args.rng_gaussian()  # Coma along y axis (wavelength)
    opt_c2 = args.rng_gaussian()  # Coma along x axis (wavelength)
    spher = args.rng_gaussian()  # Spherical aberration (wavelength)
    trefoil1 = args.rng_gaussian()  # Trefoil along y axis (wavelength)
    trefoil2 = args.rng_gaussian()  # Trefoil along x axis (wavelength)
    # Linear dimension of central obscuration as fraction of pupil linear dimension
    opt_obscuration = args.opt_obs_min + args.opt_obs_width * args.rng()
    lam_over_diam = args.lam_ov_d_min + args.lam_ov_d_width * args.rng() # Wavelength over diameter (arcsec)

    psf_obs = get_Obs_PSF(
        lam_over_diam,
        opt_defocus,
        opt_c1,
        opt_c2,
        opt_a1,
        opt_a2,
        opt_obscuration,
        atmos_fwhm,
        atmos_e,
        atmos_beta,
        spher,
        trefoil1,
        trefoil2,
        0,
        0,
        args.fov_pixels,
        args.pixel_scale,
        args.upsample,
    )
    return psf_obs,\
            {"atm":{"fwhm":atmos_fwhm,"e":atmos_e,"beta":atmos_beta},
             "opt":{"lod":lam_over_diam,
        "defocus":opt_defocus,
        "coma1":opt_c1,
        "coma2":opt_c2,
        "astig1":opt_a1,
        "astig2":opt_a2,
        "obs":opt_obscuration,
        "spher":spher,
        "tref1":trefoil1,
        "tref2":trefoil2}}

def generator(args, gal_orig, psf_hst, debug=False):
    """
    args: settings
    gal_orig: original Real Galaxy extracted from catalog
    psf_hst: HST PSF associated to gal_orig
    """
    psf_obs,_ = gener_PSF_obs(args)
    
    # Galaxy parameters .
    gal_g = args.rng_gal_shear() # Shear of the galaxy (magnitude of the shear in the "reduced shear" definition), U(0.01, 0.05).
    gal_beta = 2.0 * np.pi * args.rng()  # Shear position angle (radians), N(0,2*pi).
    gal_mu = 1.0 + args.rng() * 0.1  # Magnification, U(1.,1.1).
    theta = np.pi/2. * np.floor(4*args.rng()) #  Rotation angle (radians), 0,pi/2,pi,3/2pi
    dx = 2 * args.rng() - 1  # Offset along x axis, U(-1,1).
    dy = 2 * args.rng() - 1  # Offset along y axis, U(-1,1).
        
    gt = get_Galaxy_img(
        gal_orig,
        psf_hst,
        gal_g=gal_g,
        gal_beta=gal_beta,
        theta=theta,
        gal_mu=gal_mu,
        dx=dx,
        dy=dy,
        fov_pixels=args.fov_pixels,
        pixel_scale=args.pixel_scale,
        upsample=args.upsample,
    )

    # Convolution with the new PSF
    conv = ifftshift(ifft2(fft2(psf_obs.clone()) * fft2(gt.clone()))).real  

    # Downsample images to desired pixel scale.
    # avoid conv, psf down_scaling as there are only intermediate steps useful for debug
    gt_full = gt.clone()
    gt  = down_sample(gt_full, args.upsample)
    conv = down_sample(conv.clone(), args.upsample)

    # Add noise
    sigma =args.sigma_noise_max * args.rng()  # sigma of adittional noise, U(0;sigm_max) 
    obs = conv + torch.normal(
        mean=torch.zeros_like(conv), std=sigma * torch.ones_like(conv)
    )
    
    if debug:
        return gt, obs, gt_full, conv, psf_obs, sigma
    else:
        return gt, obs


class GalaxyDataset(Dataset):

    def __init__(
        self,
        settings,
        all_gal,
        all_psf,
        all_noise,
        all_info,
        sequence,
        debug=False
    ):
        """
        settings: run settings
        all_gal: list of galaxy FITS files
        all_psf: list of HST PSF files
        all_noise: list of noise files 
        all_info: list of additional information files        
        sequence: indexes of galaxies in the dataset
        """
        self.debug = debug
        self.settings = settings
        #
        self.all_gal   = all_gal
        self.all_psf   = all_psf
        self.all_noise = all_noise
        self.all_info  = all_info
        #
        n_gal = len(self.all_gal)
        self.seq = sequence
        
        assert n_gal >= len(self.seq), "pb n_gal < sequence length" 
        
        print("GalaxyDataset: size",len(self.seq),' among ',n_gal,'galaxies')

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        """
        Build ground truth and observation from HST galaxy
        idx : index < len(self.seq)
        """
        
        i = self.seq[idx]

        # Read out real galaxy from the catalog and the correspondig HST PSF
        gal_orig = galsim.fits.read(self.all_gal[i]) # original galaxy image
        psf      = galsim.fits.read(self.all_psf[i]) # original HST PSF
        noise    = galsim.fits.read(self.all_noise[i]) # original noise image
        with open(self.all_info[i], "r") as f:
            info = json.load(f)
            pixel_scale = info["pixel_scale"]  # original pixel_scale
            var = info['var']                  # original noise variance

        #print("GalaxyDataset gal_orig",gal_orig.array.min(),gal_orig.array.max(),gal_orig.array.sum())
        # Genetare a couple of ground truth and obersation with new PSF and snr
        psf.array = psf.array/psf.array.sum() # adjust flux to 1.0 for HST PSF
        psf_hst   = galsim.InterpolatedImage(psf)
        gal_rg    = galsim.RealGalaxy((gal_orig,psf,noise,pixel_scale,var))

        if self.debug:
             gt, obs, gt_full, conv, psf_obs, sigma = generator(self.settings, gal_rg, psf_hst, debug=True)
        else:
            gt, obs = generator(self.settings, gal_rg, psf_hst, debug=False)        
        
        # transform to CHW with C=1
        gt  = gt.unsqueeze(0)
        obs = obs.unsqueeze(0)

        if self.debug:
            return  gt, obs, gt_full, conv, psf_obs, sigma
        else:
            return gt, obs
