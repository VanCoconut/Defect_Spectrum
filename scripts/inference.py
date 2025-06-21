import torch
import torchvision as tv
from PIL import Image
import os
import sys
import yaml
import argparse
import numpy as np
from torch import nn
from utils_new.dist_util import *
from utils_new.util import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def im_save(dir, image_input, image_name, is_img):
    """
    Save image or mask to a directory, creating it if needed.
    """
    os.makedirs(dir, exist_ok=True)
    save_path = os.path.join(dir, image_name)
    if is_img:
        torchvision.utils.save_image(
            image_input,
            save_path,
            normalize=True,
            range=(-1, 1)
        )
    else:
        im = Image.fromarray(image_input)
        im.save(save_path)


def make_work_dir(save_dir):
    """
    Create directories for images, masks, and converted masks.
    """
    sample_dir = os.path.join(save_dir, 'images/')
    mask_dir = os.path.join(save_dir, 'masks/')
    mask_converted_dir = os.path.join(save_dir, 'masks_converted/')
    if get_rank() == 0:
        os.makedirs(sample_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(mask_converted_dir, exist_ok=True)
    return sample_dir, mask_dir, mask_converted_dir


def semantic_mask_to_rgb(mask, num_classes=11):
    """
    Convert a grayscale mask to an RGB image using a fixed palette.
    """
    colors = [
        (0, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0),
        (0, 255, 255), (255, 0, 255), (255, 255, 0),
        (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 128)
    ]
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(num_classes):
        rgb[mask == i] = colors[i]
    return rgb


def log_images(iteration, image_dir, mask_dir, mask_converted_dir, model_small, diffusion_small, args, batch_size):
    """
    Generate samples using small receptive model and save images and masks.
    """
    model_small.eval()
    ch = args.num_defect + 4
    noise_shape = (batch_size, ch, 256, 256)
    noise = torch.randn(noise_shape, device='cuda')
    model_kwargs = {"num_timesteps": (args.step_inference, 0)}

    # Generate
    images_masks, _ = diffusion_small.p_sample_loop(
        model=model_small,
        shape=noise_shape,
        progress=True if get_rank() == 0 else False,
        noise=noise,
        return_intermediates=True,
        model_kwargs=model_kwargs,
        log_interval=diffusion_small.num_timesteps // 10
    )

    # Save outputs
    img_name = str(iteration).zfill(6)
    for idx in range(images_masks.shape[0]):
        sample = images_masks[idx]
        image = sample[:3]
        mask_tensor = sample[3:]
        softmax_output = torch.softmax(mask_tensor, dim=0)
        pred_mask = torch.argmax(softmax_output, dim=0).cpu().numpy().astype(np.uint8)
        rgb_mask = semantic_mask_to_rgb(pred_mask, num_classes=args.num_defect + 1)

        im_save(image_dir, image, f'sample_img_{img_name}_{idx}.png', is_img=True)
        im_save(mask_dir, rgb_mask, f'sample_mask_{img_name}_{idx}.png', is_img=False)
        im_save(mask_converted_dir, pred_mask, f'sample_mask_conv_{img_name}_{idx}.png', is_img=False)
    synchronize()
    model_small.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--step_inference", type=int, required=True)
    parser.add_argument("--sample_dir", type=str, required=True)
    parser.add_argument("--small_recep", type=str, required=True)
    parser.add_argument("--small_recep_config", type=str, required=True)
    parser.add_argument("--num_defect", type=int, required=True)
    args = parser.parse_args()

    # Distributed setup
    n_gpu = int(os.environ.get("WORLD_SIZE", 1))
    distributed = n_gpu > 1
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    device = "cuda"
    batch_size = 4

    # Load small config and model
    with open(args.small_recep_config, 'r') as f:
        cfg_small = yaml.safe_load(f)
    diffusion_small = instantiate_from_config(cfg_small['diffusion']).to(device)
    model_small = instantiate_from_config(cfg_small['model']).to(device)

    ckpt_small = torch.load(args.small_recep, map_location=device)
    model_small.load_state_dict(ckpt_small['model'], strict=False)

    if distributed:
        model_small = nn.parallel.DistributedDataParallel(
            model_small,
            device_ids=[args.local_rank],
            output_device=args.local_rank
        )

    # Prepare output dirs
    img_dir, m_dir, mc_dir = make_work_dir(args.sample_dir)

    # Run inference
    for i in range(10):
        with torch.no_grad():
            log_images(i, img_dir, m_dir, mc_dir, model_small, diffusion_small, args, batch_size)
