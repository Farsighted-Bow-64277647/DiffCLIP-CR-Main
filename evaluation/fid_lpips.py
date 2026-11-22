import os
from glob import glob
import numpy as np
import lpips
import torch
import subprocess
import sys
from tqdm import tqdm


def calculate_fid(src, dst):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if sys.platform == "win32":
        num_workers = 0
    else:
        num_workers = 8
    subprocess.run(
        f"python -m pytorch_fid  {src} {dst} --device {device} --batch-size 8 --num-workers {num_workers}",
    )


def calculate_lpips(src, dst):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn_alex = lpips.LPIPS(net='alex', verbose=False).to(device)
    lpips_list = []
    for i, j in tqdm(zip(glob(src + "/*"), glob(dst + "/*")), total=len(glob(src + "/*"))):
        img1 = lpips.im2tensor(lpips.load_image(i)).to(device)
        img2 = lpips.im2tensor(lpips.load_image(j)).to(device)
        lpips_list.append(loss_fn_alex(img1, img2).item())
    print(f"LPIPS:{np.mean(lpips_list)}")


if __name__ == "__main__":
    src = r"C:\Marcus\G10-Diffusion\DiffCR-main (1)\IEEE-TGSRS-Submits\DiffClip-CR-main\experiments\test_ours_double_encoder_splitcaCond_splitcaUnet_sigmoid_260819_092644\results\test\0\GT"
    dst = r"C:\Marcus\G10-Diffusion\DiffCR-main (1)\IEEE-TGSRS-Submits\DiffClip-CR-main\experiments\test_ours_double_encoder_splitcaCond_splitcaUnet_sigmoid_260819_092644\results\test\0\Out"
    src_files = glob(src + "/*")
    dst_files = glob(dst + "/*")

    print("SRC:", len(src_files))
    print("DST:", len(dst_files))
    print(src_files[:5])
    print(dst_files[:5])
    calculate_lpips(src, dst)
