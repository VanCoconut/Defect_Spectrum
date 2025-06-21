# import necessary packages
import torch
import torchvision as tv
from PIL import Image
import random
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch import nn
import os
import sys
import torchvision
import yaml
#from data.datasets import  Dataset
from torch.utils.data import DataLoader
from utils_new.dist_util import *
from utils_new.util import *
import argparse
from torchinfo import summary
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def tensor_to_image(x):
    ndarr = x.squeeze().mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    image = Image.fromarray(ndarr)
    return image


def im_save(dir, image_input, image_name, is_img):
    directory = os.path.join(dir)
    isExist = os.path.exists(directory)
    if not isExist:
        os.makedirs(directory)
        print("The new directory is created!")
    save_path = os.path.join(directory, image_name)
    if is_img:
        torchvision.utils.save_image(
            image_input,
            save_path,
            normalize=True, range=(-1, 1)
        )
    else:
        im = Image.fromarray(image_input)
        im.save(save_path)


def create_argparser():
    defaults = dict(
        local_rank=0,
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def make_work_dir(save_dir):
    sample_dir = os.path.join(save_dir, 'images/')
    mask_dir = os.path.join(save_dir, 'masks/')
    mask_converted_dir = os.path.join(save_dir, 'masks_converted_dir/')

    if get_rank() == 0:
        os.makedirs(sample_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(mask_converted_dir, exist_ok=True)
    return sample_dir, mask_dir, mask_converted_dir


def normalize_tensor(tensor, min_val=1, max_val=5):
    t_min = tensor.min()
    t_max = tensor.max()
    tensor_normalized = (tensor - t_min) / (t_max - t_min)
    tensor_scaled = (tensor_normalized * (max_val - min_val) + min_val).round()
    return tensor_scaled


def semantic_mask_to_rgb(mask):
    colors = [
        (0, 0, 0),       # 0: Black
        (0, 0, 255),     # 1: Blue
        (0, 255, 0),     # 2: Green
        (255, 0, 0),     # 3: Red
        (0, 255, 255),   # 4: Yellow
        (255, 0, 255),   # 5: Magenta
        (255, 255, 0),   # 6: Cyan
        (128, 0, 0),     # 7: Dark Red
        (0, 128, 0),     # 8: Dark Green
        (0, 0, 128),     # 9: Dark Blue
        (128, 128, 128)  # 10: Gray
    ]

    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(11):
        rgb_mask[mask == i] = colors[i]
    return rgb_mask


def log_images(iter, image_dir, mask_dir, mask_converted_dir, model, diffusion, args, batch_size=4):
    model_kwargs = {}
    ch = args.num_defect + 4
    pred = torch.randn(batch_size, ch, 256, 256).to(next(model.parameters()).device)
    img_name = str(iter).zfill(6)

    # Large step sampling or only small, depends on diffusion.num_timesteps shape
    if hasattr(diffusion, 'num_timesteps') and diffusion.num_timesteps > args.step_inference:
        model_kwargs["num_timesteps"] = (1000, args.step_inference)
        generated_sampled_1, _ = diffusion.p_sample_loop(
            model=model,
            shape=pred.shape,
            progress=True,
            noise=None,
            return_intermediates=True,
            model_kwargs=model_kwargs,
            log_interval=diffusion.num_timesteps // 10
        )
        model_kwargs["num_timesteps"] = (args.step_inference, 0)
        # In case diffusion_small needed, but here we have only one model/diffusion passed
        # So just return the last output
        generated_sampled_2 = generated_sampled_1
    else:
        model_kwargs["num_timesteps"] = (args.step_inference, 0)
        generated_sampled_2, _ = diffusion.p_sample_loop(
            model=model,
            shape=pred.shape,
            progress=True,
            noise=None,
            return_intermediates=True,
            model_kwargs=model_kwargs,
            log_interval=diffusion.num_timesteps // 10
        )

    for i in range(generated_sampled_2.shape[0]):
        image_mask = generated_sampled_2[i]
        image = image_mask[:3, :, :]
        mask = image_mask[3:, :, :]

        softmax_output = F.softmax(mask, dim=0)
        argmax_depth = torch.argmax(softmax_output, dim=0)
        rgb_mask = semantic_mask_to_rgb(argmax_depth.cpu().numpy())

        im_save(image_dir, image, f'samples_image_{img_name}_{i}_{dist.get_rank()}.png', is_img=True)
        im_save(mask_dir, rgb_mask, f'samples_mask_{img_name}_{i}_{dist.get_rank()}.png', is_img=False)
        converted_mask = np.array(argmax_depth.cpu().numpy(), dtype=np.uint8)
        im_save(mask_converted_dir, converted_mask, f'samples_converted_mask_{img_name}_{i}_{dist.get_rank()}.png', is_img=False)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--step_inference", type=int, default=50)
    parser.add_argument("--sample_dir", type=str, default="./output")
    parser.add_argument("--large_recep", type=str, default="")
    parser.add_argument("--small_recep", type=str, required=True)
    parser.add_argument("--num_defect", type=int, default=1)
    parser.add_argument("--large_recep_config", type=str, default="")
    parser.add_argument("--small_recep_config", type=str, required=True)
    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = n_gpu > 1
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    batch_size = 4

    # Load small model config & checkpoint
    with open(args.small_recep_config, 'r', encoding='utf-8') as f_small:
        d_small = yaml.safe_load(f_small)

    diffusion_small = instantiate_from_config(d_small['diffusion']).to(device)
    model_small = instantiate_from_config(d_small['model']).to(device)

    ckpt_small = torch.load(args.small_recep, map_location=device)
    model_small.load_state_dict(ckpt_small['model'], strict=False)

    # Distributed wrap small model if needed
    if distributed:
        model_small = nn.parallel.DistributedDataParallel(
            model_small,
            device_ids=[args.local_rank],
            output_device=args.local_rank
        )

    # Decide if use large model or just small
    use_large = bool(args.large_recep and args.large_recep_config)

    if use_large:
        with open(args.large_recep_config, 'r', encoding='utf-8') as f_large:
            d_large = yaml.safe_load(f_large)

        diffusion_large = instantiate_from_config(d_large['diffusion']).to(device)
        model_large = instantiate_from_config(d_large['model']).to(device)

        ckpt_large = torch.load(args.large_recep, map_location=device)
        model_large.load_state_dict(ckpt_large['model'], strict=False)

        if distributed:
            model_large = nn.parallel.DistributedDataParallel(
                model_large,
                device_ids=[args.local_rank],
                output_device=args.local_rank
            )

    root_dir = args.sample_dir
    images_dir, mask_dir, mask_converted_dir = make_work_dir(root_dir)

    for i in range(10):
        with torch.no_grad():
            if use_large:
                # Run with large and small (you can add your two-step logic here if needed)
                # For simplicity just use large here
                log_images(i, images_dir, mask_dir, mask_converted_dir, model_large, diffusion_large, args, batch_size)
            else:
                # Use only small model and diffusion
                log_images(i, images_dir, mask_dir, mask_converted_dir, model_small, diffusion_small, args, batch_size)
