from tqdm import trange, tqdm
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import UnetDataTransform
from PerformanceMetrics import ssim, nmse, psnr
from FastMriDataModule import FastMriDataModule
from argparse import ArgumentParser
from unet_model import Unet
from torch import nn
from matplotlib import pyplot as plt
import torch.distributed as dist
from generator import Generator
import pytorch_ssim
import torch

# deprecate warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def ssim_criterion(output, target):
    ssim_loss = pytorch_ssim.SSIM()
    return ((1-ssim_loss(output, target)) / 2)

def init_model(args):
    # initialize model with given args
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    # this creates a k-space mask for transforming input data
    mask = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations
    )
    # use random masks for train transform, fixed masks for val transform
    train_transform = UnetDataTransform('singlecoil', mask_func=mask, use_seed=False)
    val_transform = UnetDataTransform('singlecoil', mask_func=mask)
    test_transform = UnetDataTransform('singlecoil')
    # Initialize Process Group
    dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)
    # define the data loaders
    batch_size = args.batch_size
    # create object for data module
    data_module = FastMriDataModule(
        data_path=args.data_path,
        challenge='singlecoil',
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_split='test',
        test_path=args.data_path+'/singlecoil_test',
        sample_rate=1,
        batch_size=batch_size,
        # may can use multiple workers here with linux?
        num_workers=0,
        distributed_sampler="ddp",
    )
    # save data to dataloader
    dataloader_tr = data_module.train_dataloader()
    dataloader_val = data_module.val_dataloader()
    dataloader_test = data_module.test_dataloader()

    return dataloader_tr, dataloader_val, dataloader_test, device

# train function
def train_model(args, train_loader, val_loader, epochs, device, tol):
    # Number of channels in the training images. For color images this is 3
    nc = 1
    # Size of z latent vector (i.e. size of generator input)
    nz = 100
    # Size of feature maps in generator
    ngf = 64
    # create model, optimizer and criterion
    # R
    model_R = Unet(in_chans=1,
                   out_chans=1,
                   chans=64,
                   num_pool_layers=5,
                   drop_prob=0,
                  )
    optimizer_R = torch.optim.AdamW(model_R.parameters(), lr=args.learning_rate)
    # G
    model_G = Generator(nc, nz, ngf)
    optimizer_G = torch.optim.AdamW(model_G.parameters(), lr=1e-4)
    # use multiple GPUs
    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs")
        model_R = nn.DataParallel(model_R)
        model_G = nn.DataParallel(model_G)
    model_R.to(device)
    model_G.to(device)
    # set objects for storing metrics
    tr_losses_R = []
    tr_losses_G = []
    val_losses = []
    tr_ssims = []
    val_ssims = []
    tr_psnrs = []
    val_psnrs = []
    tr_nmses = []
    val_nmses = []
    alpha_1 = args.alpha_1
    alpha_2 = args.alpha_2
    # track history of validation loss to perform early stopping
    # set number of epochs to track
    tol = tol
    # Train model
    for epoch in range(1, epochs+1):
        # training
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        print(f'Epoch {epoch}:')
        print('Train:')
        tr_loss, nb_tr_steps, ssim_val, tot_ssim, tot_psnr, tot_nmse, tot_loss_G, tot_loss_R = \
            0, 0, 0, 0, 0, 0, 0, 0
        model_R.train()
        model_G.train()
        for batch_idx, sample in loop:
            data, target = sample[0].unsqueeze(1).to(device), sample[1].unsqueeze(1).to(device)
            ### Generator
            # Create batch of latent vectors that we will use to visualize
            # the progression of the generator
            optimizer_G.zero_grad()
            latent_noise = torch.randn(data.shape[0], nz, 1, 1, device=device) * 1e-3
            perturbation = model_G(latent_noise)
            output_p = model_R(data + perturbation)
            # calculate regularization hinge loss
            epsilon = args.epsilon * args.batch_size
            # hinge loss
            hinge_loss = torch.clamp(torch.norm(perturbation)**2 - epsilon, min=0)
            # get best perturbation
            loss_G = -(ssim_criterion(output_p, target) * alpha_1 + hinge_loss * alpha_2)
            loss_G.backward()
            optimizer_G.step()
            ### End
            ### Reconstructor
            optimizer_R.zero_grad()
            output = model_R(data)
            perturbation = model_G(latent_noise)
            output_p = model_R(data + perturbation.detach())
            # calcualte regular loss
            loss_1 = ssim_criterion(output, target)
            # calculate loss with perturbation
            loss_2 = ssim_criterion(output_p, target)
            loss_R = loss_1 + loss_2 * alpha_1
            # calculate gradients
            loss_R.backward()
            # optimizer/scheduler "step"
            optimizer_R.step()
            ### End
            pred = output
            # calculate performance metrics
            ssim_tr = ssim(pred.squeeze(1).detach().cpu().numpy(), target.squeeze(1).detach().cpu().numpy())
            psnr_tr = psnr(pred.squeeze(1).detach().cpu().numpy(), target.squeeze(1).detach().cpu().numpy())
            nmse_tr = nmse(pred.squeeze(1).detach().cpu().numpy(), target.squeeze(1).detach().cpu().numpy())
            #print(ssim_tr)
            tot_ssim += ssim_tr
            tot_psnr += psnr_tr
            tot_nmse += nmse_tr
            tr_losses_G.append(loss_G)
            tr_losses_R.append(loss_R)
            nb_tr_steps += 1
            # update progress bar
            loop.set_description(f'Epoch [{epoch}/{epochs}]')
            loop.set_postfix(loss_G = loss_G.item(), loss_R = loss_R.item(), ssim = ssim_tr, pert = \
                             (torch.norm(perturbation)**2).item() / args.batch_size, psnr = psnr_tr, nmse = nmse_tr)
        tr_ssim = tot_ssim / nb_tr_steps
        tr_psnr = tot_psnr / nb_tr_steps
        tr_nmse = tot_nmse / nb_tr_steps
        tr_ssims.append(tr_ssim)
        tr_psnrs.append(tr_psnr)
        tr_nmses.append(tr_nmse)
        print(f'Train SSIM: {tr_ssim}')
        print(f'Train PSNR: {tr_psnr}')
        print(f'Train NMSE: {tr_nmse}')
        # validation
        model_R.eval()
        val_loss, nb_val_steps, ssim_val, tot_ssim, tot_psnr, tot_nmse = 0, 0, 0, 0, 0, 0
        print('Validation:')
        with torch.no_grad():
            for sample in val_loader:
                data, target = sample[0].unsqueeze(1).to(device), sample[1].unsqueeze(1).to(device)
                output = model_R(data)
                loss = ssim_criterion(output, target)
                pred = output
                # calculate performance metrics
                ssim_val = ssim(pred.squeeze(1).detach().cpu().numpy(), target.squeeze(1).detach().cpu().numpy())
                psnr_val = psnr(pred.squeeze(1).detach().cpu().numpy(), target.squeeze(1).detach().cpu().numpy())
                nmse_val = nmse(pred.squeeze(1).detach().cpu().numpy(), target.squeeze(1).detach().cpu().numpy())
                tot_ssim += ssim_val
                tot_psnr += psnr_val
                tot_nmse += nmse_val
                val_loss += loss.item()
                nb_val_steps += 1
        val_ssim = tot_ssim / nb_val_steps
        val_psnr = tot_psnr / nb_val_steps
        val_nmse = tot_nmse / nb_val_steps
        val_loss = val_loss / nb_val_steps
        val_losses.append(val_loss)
        val_ssims.append(val_ssim)
        val_nmses.append(val_nmse)
        val_psnrs.append(val_psnr)
        print(f'Validation SSIM: {val_ssim}')
        print(f'Validation PSNR: {val_psnr}')
        print(f'Validation NMSE: {val_nmse}')
        print(f'Validation Loss: {val_loss}')
        # check validation loss history for early stopping
        if len(val_losses) > tol:
            losses_diff_hist = []
            tracked_loss = val_losses[len(val_losses)-tol-1]
            # get last 'tol' tolerance index and calculate loss difference history
            for i in range(1, tol+1):
                losses_diff_hist.append(val_losses[len(val_losses)-i] - tracked_loss)
            print(losses_diff_hist)
            # if all histories are larger than or equal previous tracked loss, stop training
            # larger than 0 means the losses are not decreasing
            if sum([loss_diff >= 0 for loss_diff in losses_diff_hist]) == tol:
                print(sum([loss_diff >= 0 for loss_diff in losses_diff_hist]))
                break
    # save model
    torch.save(model_R.state_dict(), "unet_model_R.pt")
    torch.save(model_G.state_dict(), "unet_model_G.pt")
    return tr_ssims, tr_psnrs, tr_nmses, val_ssims, val_psnrs, val_nmses, tr_losses_G, tr_losses_R

# function for plotting
def plot_tr_val(tr_ssims, tr_psnrs, tr_nmses, val_ssims, val_psnrs, val_nmses, tr_losses_G, tr_losses_R):
    # plot train/validation accuracy/losses
    # "Performance Metrics"
    plt.figure(0)
    plt.plot(tr_ssims, 'blue')
    plt.plot(val_ssims, 'red')
    plt.title('Model SSIM')
    plt.ylabel('ssim')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('ssim.png')

    plt.figure(1)
    plt.plot(tr_psnrs, 'blue')
    plt.plot(val_psnrs, 'red')
    plt.title('Model PSNR')
    plt.ylabel('psnr')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('psnr.png')

    plt.figure(2)
    plt.plot(tr_nmses, 'blue')
    plt.plot(val_nmses, 'red')
    plt.title('Model NMSE')
    plt.ylabel('nmse')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('nmse.png')

    # "Loss"
    plt.figure(3)
    plt.plot(tr_losses_R, 'blue')
    plt.title('Model R Loss')
    plt.ylabel('loss R')
    plt.xlabel('iteration')
    plt.legend(['losses_R'], loc='upper left')
    plt.savefig('loss_r.png')

    plt.figure(4)
    plt.plot(tr_losses_G, 'red')
    plt.title('Model G Loss')
    plt.ylabel('loss G')
    plt.xlabel('iteration')
    plt.legend(['losses_G'], loc='upper left')
    plt.savefig('loss_g.png')

    plt.figure(5)
    plt.plot(tr_losses_G, 'red')
    plt.plot(tr_losses_R, 'blue')
    plt.title('Model Loss')
    plt.ylabel('loss R')
    plt.xlabel('iteration')
    plt.legend(['losses_G', 'losses_R'], loc='upper left')
    plt.savefig('loss_rg.png')

def build_args():
    parser = ArgumentParser()
    # client arguments
    parser.add_argument(
        "--data_path",
        required = True,
        type=str,
        help="Data path of dataset",
    )
    # model params
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Batch size of model",
    )
    # learning rate
    parser.add_argument(
        "--learning_rate",
        default=1e-4,
        type=float,
        help="Learning rate of optimizer",
    )
    # data transform params
    parser.add_argument(
        "--mask_type",
        choices=("random", "equispaced"),
        default="random",
        type=str,
        help="Type of k-space mask",
    )
    # center fraction
    parser.add_argument(
        "--center_fractions",
        nargs="+",
        default=[0.08],
        type=float,
        help="Number of center lines to use in mask",
    )
    # acceleration
    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=[4],
        type=int,
        help="Acceleration rates to use for masks",
    )
    parser.add_argument(
        "--alpha_1",
        default=3,
        type=float,
        help="alpha_1 for objective",
    )
    parser.add_argument(
        "--alpha_2",
        default=-1,
        type=float,
        help="alpha_2 for objective",
    )
    parser.add_argument(
        "--epsilon",
        default=1.5e4,
        type=float,
        help="each sample epsilon for regularize norm of perturbation",
    )

    args = parser.parse_args()
    return args

def main():
    args = build_args()
    dataloader_tr, dataloader_val, dataloader_test, device = init_model(args)
    epochs = 5
    # train
    tr_ssims, tr_psnrs, tr_nmses, val_ssims, val_psnrs, val_nmses, tr_losses_G, tr_losses_R = train_model(args,
                                                                                                          dataloader_tr,
                                                                                                          dataloader_val,
                                                                                                          epochs,
                                                                                                          device,
                                                                                                          3)
    # plot
    plot_tr_val(tr_ssims, tr_psnrs, tr_nmses, val_ssims, val_psnrs, val_nmses, tr_losses_G, tr_losses_R)

if __name__ == "__main__":
    main()
