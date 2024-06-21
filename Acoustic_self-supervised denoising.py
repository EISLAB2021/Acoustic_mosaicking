from __future__ import division
import os
import time
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
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from arch_unet import UNet

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--noisetype", type=str, default="gauss25")
parser.add_argument('--data_dir', type=str, default=' ')
parser.add_argument('--val_dirs', type=str, default='')
parser.add_argument('--save_model_path', type=str, default='')
parser.add_argument('--log_name', type=str, default='')
parser.add_argument('--gpu_devices', default='0', type=str)
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--n_feature', type=int, default=48)
parser.add_argument('--n_channel', type=int, default=3)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--n_epoch', type=int, default=125)
parser.add_argument('--n_snapshot', type=int, default=1)
parser.add_argument('--batchsize', type=int, default=4)
parser.add_argument('--patchsize', type=int, default=256)
parser.add_argument("--Lambda1", type=float, default=1.0)
parser.add_argument("--Lambda2", type=float, default=1.0)
parser.add_argument("--increase_ratio", type=float, default=2.0)
parser.add_argument('--log_dir', type=str, default='./directory')

opt, _ = parser.parse_known_args()
systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
operation_seed_counter = 0
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_devices


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
    g_cuda_generator = torch.Generator(device="cuda")
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator


# Acoustic data augmentation with noise
class AugmentNoise(object):
    def __init__(self, style):
        print(style)
        if style.startswith('gauss'):
            self.params = [float(p) / 255.0 for p in style.replace('gauss', '').split('_')]
            if len(self.params) == 1:
                self.style = "gauss_fix"
            elif len(self.params) == 2:
                self.style = "gauss_range"
        elif style.startswith('poisson'):
            self.params = [float(p) for p in style.replace('poisson', '').split('_')]
            if len(self.params) == 1:
                self.style = "poisson_fix"
            elif len(self.params) == 2:
                self.style = "poisson_range"

    def add_train_noise(self, x):
        shape = x.shape
        if self.style == "gauss_fix":
            std = self.params[0]
            std = std * torch.ones((shape[0], 1, 1, 1), device=x.device)
            noise = torch.cuda.FloatTensor(shape, device=x.device)
            torch.normal(mean=0.0, std=std, generator=get_generator(), out=noise)
            return x + noise
        elif self.style == "gauss_range":
            min_std, max_std = self.params
            std = torch.rand(size=(shape[0], 1, 1, 1), device=x.device) * (max_std - min_std) + min_std
            noise = torch.cuda.FloatTensor(shape, device=x.device)
            torch.normal(mean=0, std=std, generator=get_generator(), out=noise)
            return x + noise
        elif self.style == "poisson_fix":
            lam = self.params[0]
            lam = lam * torch.ones((shape[0], 1, 1, 1), device=x.device)
            noised = torch.poisson(lam * x, generator=get_generator()) / lam
            return noised
        elif self.style == "poisson_range":
            min_lam, max_lam = self.params
            lam = torch.rand(size=(shape[0], 1, 1, 1), device=x.device) * (max_lam - min_lam) + min_lam
            noised = torch.poisson(lam * x, generator=get_generator()) / lam
            return noised

    def add_valid_noise(self, x):
        shape = x.shape
        if self.style == "gauss_fix":
            std = self.params[0]
            return np.array(x + np.random.normal(size=shape) * std, dtype=np.float32)
        elif self.style == "gauss_range":
            min_std, max_std = self.params
            std = np.random.uniform(low=min_std, high=max_std, size=(1, 1, 1))
            return np.array(x + np.random.normal(size=shape) * std, dtype=np.float32)
        elif self.style == "poisson_fix":
            lam = self.params[0]
            return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)
        elif self.style == "poisson_range":
            min_lam, max_lam = self.params
            lam = np.random.uniform(low=min_lam, high=max_lam, size=(1, 1, 1))
            return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)


# Reshape tensor with space-to-depth transformation
def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size ** 2, h // block_size, w // block_size)


# Generate acoustic image mask pairs
def generate_mask_pair(img):
    n, c, h, w = img.shape
    mask1 = torch.zeros(size=(n * h // 2 * w // 2 * 4,), dtype=torch.bool, device=img.device)
    mask2 = torch.zeros(size=(n * h // 2 * w // 2 * 4,), dtype=torch.bool, device=img.device)

    idx_pair = torch.tensor([[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]], dtype=torch.int64,
                            device=img.device)
    rd_idx = torch.zeros(size=(n * h // 2 * w // 2,), dtype=torch.int64, device=img.device)
    torch.randint(low=0, high=8, size=(n * h // 2 * w // 2,), generator=get_generator(), out=rd_idx)
    rd_pair_idx = idx_pair[rd_idx]
    rd_pair_idx += torch.arange(start=0, end=n * h // 2 * w // 2 * 4, step=4, dtype=torch.int64,
                                device=img.device).reshape(-1, 1)

    mask1[rd_pair_idx[:, 0]] = 1
    mask2[rd_pair_idx[:, 1]] = 1
    return mask1, mask2


# Generate subimages based on masks (Sub-sampler)
def generate_subimages(img, mask):
    n, c, h, w = img.shape
    subimage = torch.zeros(n, c, h // 2, w // 2, dtype=img.dtype, layout=img.layout, device=img.device)

    for i in range(c):
        img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=2)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        subimage[:, i:i + 1, :, :] = img_per_channel[mask].reshape(n, h // 2, w // 2, 1).permute(0, 3, 1, 2)
    return subimage


# Apply guided filter to sub-images
def apply_guided_filter(subsample, guide_image, d=100, eps=10):
    subsample = subsample.cpu().permute(0, 2, 3, 1).numpy()
    guide_image = guide_image.cpu().permute(0, 2, 3, 1).numpy()

    imgGuidedFilters = []
    for imgGuide in guide_image:
        imgGuide = np.uint8(imgGuide * 255)
        imgGuide_resized = cv2.resize(imgGuide, (128, 128))
        imgGuidedFilter = cv2.ximgproc.guidedFilter(imgGuide_resized, subsample, d, eps, -1)
        imgGuidedFilters.append(imgGuidedFilter)

    imgGuidedFilters = np.array(imgGuidedFilters)
    imgGuidedFilters = torch.from_numpy(imgGuidedFilters).permute(0, 3, 1, 2).float().to('cuda')

    return imgGuidedFilters


# Custom Dataset -- acoustic images
class MyDataset(Dataset):
    def __init__(self, image_dir, patch_size, transform=None):
        self.image_paths = sorted(glob.glob(image_dir))
        self.patch_size = patch_size
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        h, w = image.size[1], image.size[0]
        patch = transforms.RandomCrop((self.patch_size, self.patch_size))(image)
        return patch


# Training process
def train(net, train_loader, optimizer, scheduler, epoch, loss_fn, device):
    net.train()
    epoch_loss = 0
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        images = data.to(device)
        noisy_images = add_noise(images)
        output = net(noisy_images)
        loss = loss_fn(output, images)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    scheduler.step()
    return epoch_loss / len(train_loader)


# Adding noise to acoustic images
def add_noise(images):
    augment_noise = AugmentNoise(opt.noisetype)
    return augment_noise.add_train_noise(images)


# Main function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = UNet(n_channels=opt.n_channel, n_classes=opt.n_channel, bilinear=True)
    if torch.cuda.device_count() > 1 and opt.parallel:
        net = torch.nn.DataParallel(net)
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=opt.gamma)
    loss_fn = torch.nn.MSELoss()
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = MyDataset(opt.data_dir, opt.patchsize, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=opt.batchsize, shuffle=True, num_workers=4)
    log_dir = os.path.join(opt.log_dir, opt.log_name, systime)
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(opt.n_epoch):
        epoch_loss = train(net, train_loader, optimizer, scheduler, epoch, loss_fn, device)
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        if (epoch + 1) % opt.n_snapshot == 0:
            checkpoint(net, epoch + 1, 'train')

    writer.close()


if __name__ == "__main__":
    main()
