# refer to Neighbor2Neighbor
from arch_unet import UNet

import os
import glob
import datetime
import argparse
import numpy as np
import cv2
from PIL import Image
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

parser = argparse.ArgumentParser()
parser.add_argument("--noisetype", type=str, default=" ")  # Simulated noise (Gaussian, Poisson or custom speckle, etc)
parser.add_argument('--data_dir', type=str, default=' ')  # Loaded acoustic camera image
parser.add_argument('--val_dirs', type=str, default=' ')  # Val set
parser.add_argument('--save_model_path', type=str, default='')  # Save trained model
parser.add_argument('--log_name', type=str, default=' ')
parser.add_argument('--gpu_devices', default='0', type=str)
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--n_feature', type=int, default=48)
parser.add_argument('--n_channel', type=int, default=3)
parser.add_argument('--lr', type=float, default=2e-4)  # Adjustable parameters
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--n_epoch', type=int, default=1000)  # Adjustable parameters
parser.add_argument('--interval', type=int, default=100)  # Adjustable parameters
parser.add_argument('--n_snapshot', type=int, default=1)
parser.add_argument('--batchsize', type=int, default=4)  # Adjustable parameters
parser.add_argument('--patchsize', type=int, default=256)
parser.add_argument("--Lambda1", type=float, default=1.0)
parser.add_argument("--Lambda2", type=float, default=1.0)
parser.add_argument("--increase_ratio", type=float, default=2.0)
parser.add_argument('--log_dir', type=str, default=' ')  # log

opt, _ = parser.parse_known_args()
systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
operation_seed_counter = 0
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_devices


def calculate_ssim(img1, img2):
    # Constants for SSIM calculation
    C1 = (0.01 ** 2)
    C2 = (0.03 ** 2)

    # Convert images to grayscale
    if len(img1.shape) > 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) > 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compute mean values
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)

    # Compute variances
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)

    # Compute covariance
    sigma12 = np.cov(img1.flatten(), img2.flatten())[0, 1]

    # Compute SSIM
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_value = numerator / denominator

    return ssim_value


def calculate_psnr(img1, img2):
    img1 = np.array(img1, dtype=np.float32)
    img2 = np.array(img2, dtype=np.float32)
    diff = img1 - img2
    psnr = 10.0 * np.log10(255.0 * 255.0 / np.mean(np.square(diff)))
    return psnr


# Save acoustic denoising model checkpoints
def checkpoint(net, epoch, name):
    save_model_path = os.path.join(opt.save_model_path, opt.log_name, systime)
    os.makedirs(save_model_path, exist_ok=True)
    model_name = 'epoch_{}_{:03d}.pth'.format(name, epoch)
    save_model_path = os.path.join(save_model_path, model_name)
    torch.save(net.state_dict(), save_model_path)
    print('Checkpoint saved to {}'.format(save_model_path))


# Get CUDA random generator
def get_generator():
    global operation_seed_counter
    operation_seed_counter += 1
    # Create a CUDA generator with manual seed
    g_cuda_generator = torch.Generator(device="cuda")
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator


# Acoustic data augmentation with noise
class AugmentNoise(object):
    def __init__(self, style):
        self.style = style
        if style.startswith('gauss'):
            self.params = [float(p) / 255.0 for p in style.replace('gauss', '').split('_')]
        elif style.startswith('poisson'):
            self.params = [float(p) for p in style.replace('poisson', '').split('_')]

    def add_train_noise(self, x):
        shape = x.shape
        if self.style.startswith("gauss"):
            std = self.params[0]
            if len(self.params) == 1:
                noise = torch.normal(0.0, std, size=shape, device=x.device)
            elif len(self.params) == 2:
                min_std, max_std = self.params
                std = torch.rand(shape[0], 1, 1, 1, device=x.device) * (max_std - min_std) + min_std
                noise = torch.normal(0.0, std, size=shape, device=x.device)
            return x + noise
        elif self.style.startswith("poisson"):
            lam = self.params[0]
            if len(self.params) == 1:
                return torch.poisson(lam * x, generator=get_generator()) / lam
            elif len(self.params) == 2:
                min_lam, max_lam = self.params
                lam = torch.rand(shape[0], 1, 1, 1, device=x.device) * (max_lam - min_lam) + min_lam
                return torch.poisson(lam * x, generator=get_generator()) / lam

    def add_valid_noise(self, x):
        if self.style.startswith("gauss"):
            std = self.params[0]
            noise = np.random.normal(scale=std, size=x.shape).astype(np.float32)
            return x + noise
        elif self.style.startswith("poisson"):
            lam = self.params[0]
            return np.random.poisson(lam * x).astype(np.float32)


def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = F.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size ** 2, h // block_size, w // block_size)


def generate_mask_pair(img):
    n, c, h, w = img.shape
    mask1 = torch.zeros(n * h // 2 * w // 2 * 4, dtype=torch.bool, device=img.device)
    mask2 = torch.zeros_like(mask1)
    # collect random mask pairs
    idx_pair = torch.tensor([[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]], dtype=torch.int64,
                            device=img.device)
    rd_idx = torch.randint(0, 8, (n * h // 2 * w // 2,), device=img.device)
    rd_pair_idx = idx_pair[rd_idx] + torch.arange(0, n * h // 2 * w // 2 * 4, 4, device=img.device).unsqueeze(1)
    mask1[rd_pair_idx[:, 0]] = 1
    mask2[rd_pair_idx[:, 1]] = 1
    return mask1, mask2


def generate_subimages(img, mask):
    n, c, h, w = img.shape
    subimage = torch.zeros(n, c, h // 2, w // 2, dtype=img.dtype, device=img.device)
    for i in range(c):
        img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=2)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        subimage[:, i:i + 1, :, :] = img_per_channel[mask].reshape(n, h // 2, w // 2, 1).permute(0, 3, 1, 2)
    return subimage


class DataLoader_Imagenet_val(Dataset):
    def __init__(self, data_dir, patch=256):
        super(DataLoader_Imagenet_val, self).__init__()
        self.data_dir = data_dir
        self.patch = patch
        self.train_fns = glob.glob(os.path.join(self.data_dir, "*"))
        print('fetch {} samples for training'.format(len(self.train_fns)))

    def __getitem__(self, index):
        fn = self.train_fns[index]
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        H, W, _ = im.shape
        if H - self.patch > 0:
            xx = np.random.randint(0, H - self.patch)
            im = im[xx:xx + self.patch, :, :]
        if W - self.patch > 0:
            yy = np.random.randint(0, W - self.patch)
            im = im[:, yy:yy + self.patch, :]
        transformer = transforms.Compose([transforms.ToTensor()])
        im = transformer(im)
        return im

    def __len__(self):
        return len(self.train_fns)


def validation_acoustic(dataset_dir):
    fns = glob.glob(os.path.join(dataset_dir, "test", "*"))
    images = [np.array(Image.open(fn), dtype=np.float32) for fn in fns]
    return images


# Load the acousitc images training dataset
TrainingDataset = DataLoader_Imagenet_val(opt.data_dir, patch=opt.patchsize)
TrainingLoader = DataLoader(dataset=TrainingDataset, num_workers=0, batch_size=opt.batchsize, shuffle=True,
                            pin_memory=False, drop_last=True)

# Prepare the acoustic images validation datasets
valid_dict = {"acoustic": validation_acoustic(os.path.join(opt.val_dirs, " "))}

noise_adder = AugmentNoise(style=opt.noisetype)

# Network setup
network = UNet(in_nc=opt.n_channel, out_nc=opt.n_channel, n_feature=opt.n_feature)
if opt.parallel:
    network = torch.nn.DataParallel(network)
network = network.cuda() if torch.cuda.is_available() else network

# Training parameters
num_epoch = opt.n_epoch
interval = opt.interval
optimizer = optim.Adam(network.parameters(), lr=opt.lr)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(20 * num_epoch / interval) - 1,
                                                            int(40 * num_epoch / interval) - 1,
                                                            int(60 * num_epoch / interval) - 1,
                                                            int(80 * num_epoch / interval) - 1],
                                     gamma=opt.gamma)
print("Batchsize={}, number of training epoch={}".format(opt.batchsize, opt.n_epoch))
checkpoint(network, 0, "model")

# Initialize SummaryWriter
writer = SummaryWriter(log_dir=opt.log_dir)

# Training loop of acoustic denoised model
for epoch in range(1, opt.n_epoch + 1):
    for param_group in optimizer.param_groups:
        current_lr = param_group['lr']
    print("LearningRate of Epoch {} = {}".format(epoch, current_lr))
    loss_now = 0

    network.train()
    for iteration, clean in enumerate(TrainingLoader):
        clean = clean / 255.0
        clean = clean.cuda()

        noisy = noise_adder.add_train_noise(clean)
        optimizer.zero_grad()

        mask1, mask2 = generate_mask_pair(noisy)
        noisy_sub1 = generate_subimages(noisy, mask1)
        noisy_sub2 = generate_subimages(noisy, mask2)
        with torch.no_grad():
            noisy_denoised = network(noisy)
        noisy_sub1_denoised = generate_subimages(noisy_denoised, mask1)
        noisy_sub2_denoised = generate_subimages(noisy_denoised, mask2)
        noisy_output = network(noisy_sub1)
        noisy_target = noisy_sub2

        Lambda = epoch / opt.n_epoch * opt.increase_ratio
        diff = noisy_output - noisy_target
        exp_diff = noisy_sub1_denoised - noisy_sub2_denoised

        loss1 = torch.mean(diff ** 2)
        loss2 = Lambda * torch.mean((diff - exp_diff) ** 2)
        loss_all = opt.Lambda1 * loss1 + opt.Lambda2 * loss2

        loss_all.backward()
        loss_now += loss_all.item()
        optimizer.step()

        print(
            '{:04d} {:05d} Loss1={:.6f}, Lambda={}, Loss2={:.6f}, Loss_Full={:.6f}'
            .format(epoch, iteration, loss1.item(), Lambda, loss2.item(), loss_all.item()))

    scheduler.step()
    writer.add_scalar("Loss/train", loss_now / len(TrainingLoader), epoch)

    if epoch % opt.n_snapshot == 0 or epoch == opt.n_epoch:
        network.eval()
        checkpoint(network, epoch, "model")
        save_model_path = os.path.join(opt.save_model_path, opt.log_name)
        validation_path = os.path.join(save_model_path, "validation")
        os.makedirs(validation_path, exist_ok=True)

        valid_repeat_times = {"acoustic": 1}

        for valid_name, valid_images in valid_dict.items():
            repeat_times = valid_repeat_times[valid_name]
            ssim_result = []
            psnr_result = []

            for i in range(repeat_times):
                for idx, im in enumerate(valid_images):
                    print(
                        f"Image {idx}: Shape={im.shape}, dtype={im.dtype}, "
                        f"min_value={im.min()}, max_value={im.max()}")
                    original_ac = im.astype(np.uint8)
                    im = im.astype(np.float32) / 255.0
                    noisy_im = noise_adder.add_valid_noise(im)

                    H, W, _ = noisy_im.shape
                    val_size = (max(H, W) + 31) // 32 * 32
                    noisy_im = np.pad(noisy_im, [(0, val_size - H), (0, val_size - W), (0, 0)], 'reflect')

                    transformer = transforms.Compose([transforms.ToTensor()])
                    noisy_im = transformer(noisy_im)
                    noisy_im = torch.unsqueeze(noisy_im, 0)
                    noisy_im = noisy_im.cuda()

                    with torch.no_grad():
                        prediction = network(noisy_im)
                        prediction = prediction[:, :, :H, :W]

                    prediction = prediction.permute(0, 2, 3, 1)
                    prediction = prediction.cpu().data.clamp(0, 1).numpy()
                    prediction = prediction.squeeze()

                    pred_ac = np.clip(prediction * 255.0 + 0.5).astype(np.uint8)
                    # Calculating evaluation metrics
                    img_psnr = calculate_psnr(original_ac.astype(np.float32), pred_ac.astype(np.float32))
                    img_ssim = calculate_ssim(original_ac.astype(np.float32), pred_ac.astype(np.float32))
                    psnr_result.append(img_psnr)
                    ssim_result.append(img_psnr)

                    # Visualize the acoustic image
                    # if i == 0 and epoch == opt.n_snapshot:
                    #     save_path = os.path.join(validation_path,
                    #                              "{}_{:03d}-{:03d}_clean.png".format(valid_name, idx, epoch))
                    #     Image.fromarray(original_ac).convert('RGB').save(save_path)
                    #
                    #     save_path = os.path.join(validation_path,
                    #                              "{}_{:03d}-{:03d}_noisy.png".format(valid_name, idx, epoch))
                    #     Image.fromarray(noisy_im.astype(np.uint8).squeeze()).convert('RGB').save(save_path)
                    #
                    # if i == 0:
                    #     save_path = os.path.join(validation_path,
                    #                              "{}_{:03d}-{:03d}_denoised.png".format(valid_name, idx, epoch))
                    #     Image.fromarray(pred_ac.squeeze()).convert('RGB').save(save_path)

## The trained model weight with the highest score is selected for subsequent inference.
