import argparse
import yaml
import json
import os
import random
import pathlib
import pickle
from types import SimpleNamespace
import multiprocessing
import time
import regex as re
import glob

import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
from torch.fft import fft2, ifft2, fftshift, ifftshift

import numpy as np

import galsim
from model import *
from galaxy_img_utils import *

#################
# Training model for COSMOS galaxy with observational features
################

###########
# dataset utils
###########

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


################
# Utils
################
def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

#miltiscale loss
class MultiScaleLoss(nn.Module):
	def __init__(self, scales=3, norm='L1'):
		super(MultiScaleLoss, self).__init__()
		self.scales = scales
		if norm == 'L1':
			self.loss = nn.L1Loss()
		if norm == 'L2':
			self.loss = nn.MSELoss()

		self.weights = torch.FloatTensor([1/(2**scale) for scale in range(self.scales)])
		self.multiscales = [nn.AvgPool2d(2**scale, 2**scale) for scale in range(self.scales)]

	def forward(self, output, target):
		loss = 0
		for i in range(self.scales):
			output_i, target_i = self.multiscales[i](output), self.multiscales[i](target)
			loss += self.weights[i]*self.loss(output_i, target_i)
		return loss



################
# train/test 1-epoch
################
def train(args, model, criterion, train_loader, transforms, optimizer, epoch):

    # train mode
    model.train()

    loss_sum = 0  # to get the mean loss over the dataset
    print("Train.... start")

    t0 = time.time()
    for i_batch, imgs in enumerate(train_loader):

        #if i_batch==0 or i_batch%10 ==0:
        #    print("train batch:",i_batch,"...start at",time.time()-t0)
        
        gt, obs = imgs # ground truth and observaton
        
        gt = gt.to(args.device)
        obs = obs.to(args.device)

        # train step
        optimizer.zero_grad()
        output = model(obs)

        loss = criterion(output, gt)
        loss_sum += loss.item()
        # backprop to compute the gradients
        loss.backward()
        # perform an optimizer step to modify the weights
        optimizer.step()

    print(f"train epoch ...stop= {time.time()-t0:.2f}")

    return loss_sum / (i_batch + 1)


def test(args, model, criterion, test_loader, transforms, epoch):

    # test mode
    model.eval()

    t0 = time.time()

    loss_sum = 0  # to get the mean loss over the dataset
    with torch.no_grad():
        for i_batch, imgs in enumerate(test_loader):

            #if i_batch==0 or i_batch%10 ==0:
            #    print("test batch:",i_batch,"...start at",time.time()-t0)

            gt, obs = imgs # ground truth and observaton
        
            gt = gt.to(args.device)
            obs = obs.to(args.device)

            output = model(obs)

            loss = criterion(output, gt)
            loss_sum += loss.item()

    print(f"test epoch ...stop= {time.time()-t0:.2f}")
    return loss_sum / (i_batch + 1)


################
# Main: init & loop on epochs
################


def main():
    t0 = time.time()

    # Training config
    parser = argparse.ArgumentParser(
        description="Blind Deconvolution: variable PSF and noise"
    )
    parser.add_argument("--file", help="Config file")
    args0 = parser.parse_args()
    
    ## Load yaml configuration file
    with open(args0.file, "r") as config:
        settings_dict = yaml.safe_load(config)
    args = SimpleNamespace(**settings_dict)


    #default non swapping (5nov25)
    if "swap" not in args.__dict__:
        args.swap = False

    
    # check number of num_workers
    NUM_CORES = multiprocessing.cpu_count()
    if args.num_workers >= NUM_CORES:
        print("Info: # workers set to", NUM_CORES // 2)
        args.num_workers = NUM_CORES // 2

    # where to put all model training stuff
    #JEC 5nov25 add swapped
    args.out_root_dir = args.out_root_dir + "/" + args.run_tag \
        + "_" + str(args.n_train) + "train" \
        + "_" + str(args.num_kernels) + "kernels" \
        + "_" + str(args.num_blocks) + "blocks" \
        + "_" + args.loss
    if args.swap:
         args.out_root_dir =  args.out_root_dir + "_swapped"
    args.out_root_dir =  args.out_root_dir + "/"

    try:
        os.makedirs(
            args.out_root_dir, exist_ok=False
        )  # avoid erase the existing directories (exit_ok=False)
    except OSError:
        pass

    print("Info: outdir is ", args.out_root_dir)

    # device cpu/gpu...
    args.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    # seeding random generator others than Galsim
    set_seed(args.seed)

    # update settings for galaxy simulator (used in GalaxyDataset)
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


    # dataset & dataloader
    if "input_gal_file_list" in args.__dict__:
        # in case the original COSMOS files have been filtered (eg. minimum pixel size required)
        fname = args.input_data_dir + args.input_gal_file_list
        print("read input: ",fname)
        try:
            with open(fname, "r") as f:
                info = json.load(f)
                all_gal = info["all_gal"]
                all_psf = info["all_psf"]
                all_noise = info["all_noise"]
                all_info = info["all_info"]
        except FileNotFoundError:
            print(f"The file {fname} was not found: look for location.")
        except IOError:
            print(f"An error occurred while reading the file {fname}.")
    else:
        # read the whole set of COSMOS files
        ds_path = args.input_data_dir + "/" + args.input_dataset + "/"
        all_gal   = get_list(ds_path,"gal_*.fits")
        all_psf   = get_list(ds_path,"psf_*.fits")
        all_noise = get_list(ds_path,"noise_*.fits")
        all_info  = get_list(ds_path,"info*.json")

    n_total = len(all_gal)
    assert n_total == len(all_psf), "pb n_total neq n_psf"
    assert n_total == len(all_noise), "pb n_total neq n_noise"
    assert n_total == len(all_info), "pb n_total neq n_info"

    n_train = args.n_train
    n_val = args.n_val  # never used dureing trainig only to make the final plots
    assert n_train + n_val < n_total, "check sizes failed (n_train+n_val)"
    if "n_test" in args.__dict__:
        n_test = args.n_test
        assert n_train + n_val + n_test < n_total, "check sizes failed (n_train)"
    else:
        n_test = n_total - (n_train + n_val)

    sequence = np.arange(0, n_total)  # Generate random sequence for dataset.
    ####np.random.shuffle(sequence)

    #JEC 5nov25 add swapped
    if args.swap:
        train_seq = sequence[n_test : n_train + n_test]
        test_seq  = sequence[0 : n_test]
    else:
        train_seq = sequence[0 : n_train]
        test_seq  = sequence[n_train : n_train + n_test]
    assert len(train_seq) == n_train, "check training sequence size"
    assert len(test_seq)  == n_test, "check testing sequence size"


    print(f"{n_total} gals splitted in {n_train}/{n_test}/{n_val} for train/test/val ")
    print("train seq:",train_seq)
    print("test  seq:",test_seq)

    
    #print("args:",type(args),"\n",args)
    
    ds_train = GalaxyDataset(args,
                             all_gal,
                             all_psf,
                             all_noise,
                             all_info,
                             train_seq)

    ds_test  = GalaxyDataset(args,
                             all_gal,
                             all_psf,
                             all_noise,
                             all_info,
                             test_seq)

    train_loader = DataLoader(
        dataset=ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=ds_test,
        batch_size=args.test_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    # transformation: data augmentation and to torch tensor
    # nb: at least end with ToTensor()
    train_transforms = None
    test_transforms = None

    # model instantiation
    if args.archi == "Unet-Full":
        model = UNet(args)
    else:
        print("Error: ", args.archi, "unknown")
        return

    # check ouptut of model is ok. Allow to determine the model config done at run tile
    fake_input = torch.rand(1,1,args.fov_pixels,args.fov_pixels)
    out = model(fake_input)
    assert out.shape == fake_input.shape
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of parameters:{n_params/10**6:.1f} millions")

    # put model to device before loading scheduler/optimizer parameters
    model.to(args.device)

    # optimizer & scheduler

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_init,
        # amsgrad=True,
        # eps=1e-8, # by default is 1e-8
        # weight_decay=1e-3   # default is 0
    )

    if args.use_scheduler:

        if args.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, args.num_epochs, eta_min=1e-5
            )
        elif args.scheduler == "reduce":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=args.lr_decay,
                patience=args.patience,
                min_lr=1e-5,
            )
        elif args.scheduler == "steplr":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=args.patience, gamma=args.lr_decay
            )
        else:
            print("FATAL: not known scheduler...")
            return

    # check for resume session: load model/optim/scheduler dictionnaries
    start_epoch = 0

    train_loss_history = []
    test_loss_history = []

    if args.resume:
        args.checkpoint_file = args.out_root_dir + args.checkpoint_file
        args.history_loss_cpt_file = args.out_root_dir + args.history_loss_cpt_file

        # load checkpoint of model/scheduler/optimizer
        if os.path.isfile(args.checkpoint_file):
            print("=> loading checkpoint '{}'".format(args.checkpoint_file))
            checkpoint = torch.load(args.checkpoint_file)
            # the first epoch for the new training
            start_epoch = checkpoint["epoch"]
            # model update state
            model.load_state_dict(checkpoint["model_state_dict"])
            if args.resume_scheduler:
                # optizimer update state
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                # scheduler update state
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            else:
                print("=>>> scheduler not resumed")
                if args.resume_optimizer:
                    # optizimer update state
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                else:
                    print("=>>> optimizer not resumed")
            print("=> loaded checkpoint")
        else:
            print("=> FATAL no  checkpoint '{}'".format(args.checkpoint_file))
            return

        # load previous history of losses
        if os.path.isfile(args.history_loss_cpt_file):
            loss_history = np.load(args.history_loss_cpt_file, allow_pickle=True)
            train_loss_history = loss_history[0].tolist()
            test_loss_history = loss_history[1].tolist()

        else:
            print(
                "=> FATAL no history loss checkpoint '{}'".format(
                    args.history_loss_cpt_file
                )
            )
            return

    else:
        print("=> no checkpoints then Go as fresh start")

    # loss
    if args.loss == "MSE":
        print("Info: MSE is the criterion")
        criterion = nn.MSELoss(reduction="mean")
    elif args.loss == "L1":
        print("Info: L1 is the criterion")
        criterion = nn.L1Loss(reduction="mean")
    elif args.loss == "MultiScale":
        print(f"Info: MultiScale is the criterion with scales={args.multiscale_scale}, norm={args.multScale_norm}")
        criterion = MultiScaleLoss(scales=args.multiscale_scale, norm=args.multScale_norm)
    else:
        print("FATAL loss criterion not well defined")
        return
    
    # loop on epochs
    t0 = time.time()
    best_test_loss = np.inf

    print("The current args:", args)

    for epoch in range(start_epoch, args.num_epochs + 1):

        print("Start epoch",epoch)
        
        # training
        train_loss = train(
            args, model, criterion, train_loader, train_transforms, optimizer, epoch
        )
        # test
        test_loss = test(
            args, model, criterion, test_loader, test_transforms, epoch
        )

        # print & book keeping
        print(
            f"Epoch {epoch}, Losses train: {train_loss:.6f}",
            f"test {test_loss:.6f}, LR= {scheduler.get_last_lr()[0]:.2e}",
            f"time {time.time()-t0:.2f}",
        )
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)

        # update scheduler
        if args.use_scheduler:
            if args.scheduler == "reduce":
                # Warning ReduceLROnPlateau needs a metric
                scheduler.step(test_loss)
            else:
                scheduler.step()

        # save state at each epoch to be able to reload and continue the optimization
        if args.use_scheduler:
            state = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
        else:
            state = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }

        torch.save(
            state,
            args.out_root_dir + "/" + args.archi + "_last_state.pth",
        )
        # save intermediate history
        np.save(
            args.out_root_dir + "/" + args.archi + "_last_history.npy",
            np.array((train_loss_history, test_loss_history)),
        )

        # if better loss update best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
        torch.save(
            state,
            args.out_root_dir + "/" + args.archi + "_best_state.pth",
        )

    # Bye
    tf = time.time()
    print("all done!", tf - t0)


################################
if __name__ == "__main__":
    main()
