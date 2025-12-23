import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import argparse
import os
import csv

from Net import DMEGDNet
from dataload import Haze1kDataset, RRSHIDDataset, DHIDDataset, LHIDDataset
from metrics import calculate_ssim, calculate_psnr_tensor


def tensor_to_numpy(t):
    t = t.detach().cpu().clamp(0, 1)
    if t.ndim == 3:
        t = t.permute(1, 2, 0)
    elif t.ndim == 2:
        pass
    else:
        raise ValueError(f"Unexpected tensor shape: {t.shape}")
    return t.numpy()


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description="测试集去雾图")
    parser.add_argument("--data_path", type=str,
                        default='./dataset/Haze1k/Haze1k_thin/dataset/test',
                        help="测试集路径（包含input/target文件夹）")
    parser.add_argument("--ckpt", type=str,
                        default='./checkpoints/Haze1k/Thin.pth',
                        help="模型权重路径")
    parser.add_argument("--img_size", type=int, default=(512, 512))
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"加载模型权重: {args.ckpt}")
    model = DMEGDNet().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    dataset = Haze1kDataset(path=args.data_path, train=False, size=args.img_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=False)

    psnr_val = []
    ssim_val = []
    for haze, clear, id in dataloader:
        haze, clear = haze.to(device), clear.to(device)
        with torch.no_grad():
            dehazed = model(haze)
        psnr = calculate_psnr_tensor(dehazed, clear)
        ssim = calculate_ssim(dehazed, clear)
        fade = calculate_fade(dehazed)
        psnr_val.append(psnr)
        ssim_val.append(ssim)

    psnr_val = sum(psnr_val)/len(psnr_val)
    ssim_val = sum(ssim_val)/len(ssim_val)

    print(f"PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")

