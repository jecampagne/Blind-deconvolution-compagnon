import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import matplotlib.lines as lines
mpl.rcParams["font.size"] = 16
mpl.rcParams['axes.titlesize'] = 16
plt.rcParams['image.cmap'] = 'RdBu_r'
from matplotlib.ticker import MultipleLocator

import os
import sys
module_path = os.path.abspath(os.path.join('../code')) # path to your source code
sys.path.insert(0, module_path)

from types import SimpleNamespace
import multiprocessing
import time
import regex as re
import glob
import json
import yaml
import random

import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
from torch.fft import fft2, ifft2, fftshift, ifftshift

from skimage.metrics import structural_similarity as ssim
from scipy.optimize import minimize_scalar

import galsim


def psnr(x,ref, max_I=1.0):
    mse = ((x-ref)**2).mean()
    return 10*(np.log10(max_I**2) - np.log10(mse))

def crop_center(arr, npix=64):
    return arr[arr.shape[0]//2-npix//2:arr.shape[0]//2+npix//2,
        arr.shape[1]//2-npix//2:arr.shape[1]//2+npix//2]


remove_im_mean = lambda data : data - data.mean(dim=(1,2,3),keepdims=True )

def im_set_corr(set1, set2, remove_mean=True):
    '''
    set1,set2: tensors of size (N,C,H,W)
    but if not (N,C,H,W)
        if (H,W) => (1,1,H,W)
        if (C,H,W) => (1,C,H,W)
    return numpy matrix (N of set1, N of set2)

    matmul error if (C,H,W) of both sets do not coincide
    '''
        
    if isinstance(set1,np.ndarray):
        set1 = torch.from_numpy(set1)
    if isinstance(set2,np.ndarray):
        set2 = torch.from_numpy(set2)

    if len(set1.shape) == 2:
        set1=set1[np.newaxis,np.newaxis]
    if len(set1.shape) == 3:
            set1=set1[np.newaxis]
    
    if len(set2.shape) == 2:
        set2=set2[np.newaxis,np.newaxis]
    if len(set2.shape) == 3:
        set2=set2[np.newaxis]
    

    if len(set1.shape) != 4 or len(set2.shape) != 4:
        raise ValueError('Input shape error (1)')
    if set1.shape[1] != set2.shape[1] or set1.shape[2] != set2.shape[2] or set1.shape[3] != set2.shape[3] : 
            raise ValueError('Input shape error (2)')

    #for matmul
    set1 = set1.to(torch.float64)
    set2 = set2.to(torch.float64)
    

    if remove_mean: 
        set1 = remove_im_mean(set1)
        set2 = remove_im_mean(set2)

    norms1 = set1.norm(dim=(2,3), keepdim=True).norm(dim=1, keepdim=True)
    norms1[norms1 == 0 ] = .001 # to avoid dividing by 0 for blank images 
    norms2 = set2.norm(dim=(2,3), keepdim=True).norm(dim=1, keepdim=True)
    norms2[norms2 == 0 ] = .001 # to avoid dividing by 0 for blank images 

    cosine =  torch.matmul(((set1/norms1).flatten(start_dim=1)),
             (set2/norms2).flatten(start_dim=1).T)
    
    return cosine.cpu().numpy()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

# update settings for galaxy simulator (used in GalaxyDataset)
def reset_randoms(args):
    set_seed(args.seed)
    args.rng_base = galsim.BaseDeviate(seed=args.seed)
    args.rng = galsim.UniformDeviate(seed=args.seed)  # U(0,1).
    args.rng_defocus = galsim.GaussianDeviate(
    args.rng_base, mean=0.0, sigma=args.sigma_defocus)
    args.rng_gaussian = galsim.GaussianDeviate(
        args.rng_base, mean=0.0, sigma=args.sigma_opt_psf
    ) 
    args.rng_gal_shear = galsim.DistDeviate(
        seed=args.rng, function=lambda x: x, x_min=args.min_shear, x_max=args.max_shear
    )



file_pattern = re.compile(r".*?(\d+).*?")


def get_order(file):
    match = file_pattern.match(os.path.basename(file))
    if not match:
        return math.inf
    return int(match.groups()[-1])


def get_list(loc, pattern):
    # dir = pathlib.Path(dir)
    a = list(glob.glob(loc + "/" + pattern))
    return sorted(a, key=get_order)


def func(param,obs_img,psf_obs_img,upsample):
    image_deconv =  deconv_Wiener(obs_img[np.newaxis,np.newaxis,...],psf_obs_img[np.newaxis,...],
                              sampling_factor=upsample,
                                 balance=param)
    wiener_deconv_img =resize_conserve_flux(image_deconv.squeeze(),size=gt_img.shape).squeeze()
    return psnr(wiener_deconv_img,gt_img)


import math

def find_best_grid(N):
    """
    Find the best grid arrangement (nlines, ncols) for N plots,
    prioritizing a layout as close to a square as possible.
    If no exact divisor pair exists, it will choose the closest pair
    where nlines * ncols >= N and ncols is as close as possible to nlines.
    """
    # Default solution: 1 row, N columns
    best_pair = (1, N)
    min_diff = float('inf')  # Initialize with infinity to ensure any valid pair will replace it

    # Iterate over possible values for nlines (number of rows)
    for nlines in range(1, int(math.sqrt(N)) + 2):
        # Calculate the minimum number of columns needed to fit N plots
        ncols = math.ceil(N / nlines)

        # Check if this pair is closer to a square than the current best
        current_diff = abs(ncols - nlines)
        if current_diff < min_diff:
            min_diff = current_diff
            best_pair = (nlines, ncols)

    return best_pair


import numpy as np
import matplotlib.pyplot as plt

def hbprof(x, y, n_bins=20, x_range=None, error_type='std', ax=None, label=None, color='k'):
    """
    Trace un profile histogramme (moyenne de y par bin en x) avec barres d'erreur.

    Paramètres
    ----------
    x : array-like
        Vecteur des valeurs en x.
    y : array-like
        Vecteur des valeurs en y.
    n_bins : int, optionnel
        Nombre de bins en x (par défaut 20).
    x_range : tuple (min, max), optionnel
        Plage de valeurs pour les bins en x. Si None, utilise (min(x), max(x)).
    error_type : str, optionnel
        Type de barre d'erreur : 'std' (écart-type) ou 'sem' (erreur standard sur la moyenne).
    ax : matplotlib Axes, optionnel
        Axe sur lequel tracer le graphique. Si None, crée une nouvelle figure.

    Retourne
    -------
    ax : matplotlib Axes
        L'axe avec le profile histogramme tracé.
    """
    if ax is None:
        fig, ax = plt.subplots()

    if x_range is None:
        x_min, x_max = min(x), max(x)
    else:
        x_min, x_max = x_range

    bins = np.linspace(x_min, x_max, n_bins)
    bin_indices = np.digitize(x, bins) - 1

    means = []
    errors = []
    bin_centers = []

    for i in range(len(bins) - 1):
        mask = (bin_indices == i)
        if np.sum(mask) > 0:
            y_in_bin = y[mask]
            means.append(np.mean(y_in_bin))
            if error_type == 'std':
                errors.append(np.std(y_in_bin))
            elif error_type == 'sem':
                errors.append(np.std(y_in_bin) / np.sqrt(np.sum(mask)))
            bin_centers.append((bins[i] + bins[i+1]) / 2)

    ax.errorbar(bin_centers, means, yerr=errors, fmt='o', 
                 mfc=color, mec=color, c=color, capsize=5, label=label)
    return ax

