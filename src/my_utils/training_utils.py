import argparse
import json
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from glob import glob

import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from pathlib import Path
from torch.utils import data as data

from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop, triplet_random_crop
from basicsr.data.degradations import (
    random_add_gaussian_noise_pt,
    random_add_poisson_noise_pt,
    random_add_speckle_noise_pt,
    random_add_saltpepper_noise_pt,
    bivariate_Gaussian,
)

from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


def parse_args_paired_training_diff(input_args=None):
    """
    Parses command-line arguments used for configuring an paired session (pix2pix-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    # args for the loss function

    parser.add_argument(
        "--ram_path", default="assets/mm-realsr/tag2text_swin_14m.pth", type=str
    )
    # parser.add_argument(
    #     "--pos_prompt",
    #     default="Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations.",
    #     type=str,
    # )
    parser.add_argument(
        "--pos_prompt",
        default="",
        type=str,
    )

    parser.add_argument("--scale_factor", default=16, type=int)
    parser.add_argument("--finetune", default=False, action="store_true")
    parser.add_argument("--gan_disc_type", default="vagan")
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid_s")
    parser.add_argument("--lambda_clipiqa", default=1, type=float)
    parser.add_argument("--lambda_musiq", default=0.01, type=float)
    parser.add_argument("--lambda_lr", default=1, type=float)
    parser.add_argument("--lambda_gan", default=0, type=float)
    parser.add_argument("--lambda_lpips", default=5.0, type=float)
    parser.add_argument("--lambda_l2", default=2.0, type=float)
    parser.add_argument("--base_config", default="./configs/tadm_train.yaml", type=str)

    # validation eval args
    parser.add_argument("--eval_freq", default=500, type=int)
    parser.add_argument("--save_val", default=True, action="store_false")
    parser.add_argument(
        "--num_samples_eval",
        type=int,
        default=100,
        help="Number of samples to use for all evaluation",
    )

    parser.add_argument(
        "--viz_freq",
        type=int,
        default=500,
        help="Frequency of visualizing the outputs.",
    )

    # details about the model architecture
    parser.add_argument("--sd_path")
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
    )
    parser.add_argument("--dfrm_net_path", type=str, default=None)
    parser.add_argument("--lora_rank_unet", default=32, type=int)
    parser.add_argument("--lora_rank_vae", default=32, type=int)

    # training details
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--seed", type=int, default=3042, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_training_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=1e5,
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
    )
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "piecewise_constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=0.1,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--lr_step_rules",
        type=str,
        default="1:1e4,0.5:1e4,0.5:1e4,0.5",
        help="Learning rate change rules for piecewise_constant. Format: 'step:lr,...'",
    )

    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def parse_args_paired_training_dfrm(input_args=None):
    """
    Parses command-line arguments used for configuring an paired session (pix2pix-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    # args for the loss function
    parser.add_argument("--latent_channel", default=4, type=int)
    parser.add_argument("--pixel_channel", default=3, type=int)
    parser.add_argument("--scale_factor", default=16, type=int)
    parser.add_argument("--lambda_z", default=0.5, type=float)
    parser.add_argument("--lambda_lr", default=5.0, type=float)
    parser.add_argument("--base_config", default="./configs/tadm_train.yaml", type=str)

    # validation eval args
    parser.add_argument("--eval_freq", default=500, type=int)
    parser.add_argument("--save_val", default=True, action="store_false")
    parser.add_argument(
        "--num_samples_eval",
        type=int,
        default=100,
        help="Number of samples to use for all evaluation",
    )

    parser.add_argument(
        "--viz_freq",
        type=int,
        default=500,
        help="Frequency of visualizing the outputs.",
    )

    # details about the model architecture
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
    )

    # training details
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_training_epochs", type=int, default=50)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=500000,
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "piecewise_constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=0.1,
        help="Power factor of the polynomial scheduler.",
    )

    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

def randn_cropinput(lq, gt, base_size=[64, 128, 256, 512]):
    cur_size_h = random.choice(base_size)
    cur_size_w = random.choice(base_size)
    init_h = lq.size(-2) // 2
    init_w = lq.size(-1) // 2
    lq = lq[
        :,
        :,
        init_h - cur_size_h // 2 : init_h + cur_size_h // 2,
        init_w - cur_size_w // 2 : init_w + cur_size_w // 2,
    ]
    gt = gt[
        :,
        :,
        init_h - cur_size_h // 2 : init_h + cur_size_h // 2,
        init_w - cur_size_w // 2 : init_w + cur_size_w // 2,
    ]
    assert lq.size(-1) >= 64
    assert lq.size(-2) >= 64
    return [lq, gt]
