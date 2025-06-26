import sys
sys.path.append("/mnt/massive/wangce/ArbiRescale/S3Diff")
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from matplotlib import pyplot as plt
import gc
import tqdm
import math
import lpips
import pyiqa
import argparse
import clip
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import time

from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms

# from tqdm.auto import tqdm

import diffusers
from src.dataset.SingleImageDataset import SingleImageDataset
from modules.ae_arch_inv_cnn import DFRM
import utils.misc as misc

from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

from tadm_tile import TADM_Tile
from my_utils.testing_utils import parse_args_rescale_testing, PlainDataset, lr_proc
from utils.util_image import ImageSpliterTh
from my_utils.utils import instantiate_from_config
from pathlib import Path
from utils import util_image
from utils.wavelet_color import wavelet_color_fix, adain_color_fix


def evaluate(in_path, ref_path, ntest):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    metric_dict = {}
    metric_dict["clipiqa"] = pyiqa.create_metric("clipiqa").to(device)
    metric_dict["musiq"] = pyiqa.create_metric("musiq").to(device)
    metric_dict["niqe"] = pyiqa.create_metric("niqe").to(device)
    metric_dict["maniqa"] = pyiqa.create_metric("maniqa").to(device)
    metric_paired_dict = {}

    in_path = Path(in_path) if not isinstance(in_path, Path) else in_path
    assert in_path.is_dir()

    ref_path_list = None
    if ref_path is not None:
        ref_path = Path(ref_path) if not isinstance(ref_path, Path) else ref_path
        ref_path_list = sorted([x for x in ref_path.glob("*.[jpJP][pnPN]*[gG]")])
        if ntest is not None:
            ref_path_list = ref_path_list[:ntest]

        metric_paired_dict["psnr"] = pyiqa.create_metric(
            "psnr", test_y_channel=True, color_space="ycbcr"
        ).to(device)
        metric_paired_dict["lpips"] = pyiqa.create_metric("lpips-vgg").to(device)
        metric_paired_dict["dists"] = pyiqa.create_metric("dists").to(device)
        metric_paired_dict["ssim"] = pyiqa.create_metric(
            "ssim", test_y_channel=True, color_space="ycbcr"
        ).to(device)

    lr_path_list = sorted([x for x in in_path.glob("*.[jpJP][pnPN]*[gG]")])
    if ntest is not None:
        lr_path_list = lr_path_list[:ntest]

    print(f"Find {len(lr_path_list)} images in {in_path}")
    result = {}
    for i in tqdm.tqdm(range(len(lr_path_list))):
        _in_path = lr_path_list[i]
        _ref_path = ref_path_list[i] if ref_path_list is not None else None

        im_in = util_image.imread(_in_path, chn="rgb", dtype="float32")  # h x w x c
        im_in_tensor = util_image.img2tensor(im_in).cuda()  # 1 x c x h x w
        for key, metric in metric_dict.items():
            with torch.cuda.amp.autocast():
                result[key] = result.get(key, 0) + metric(im_in_tensor).item()

        if ref_path is not None:
            im_ref = util_image.imread(
                _ref_path, chn="rgb", dtype="float32"
            )  # h x w x c
            im_ref_tensor = util_image.img2tensor(im_ref).cuda()
            for key, metric in metric_paired_dict.items():
                result[key] = (
                    result.get(key, 0) + metric(im_in_tensor, im_ref_tensor).item()
                )

    if ref_path is not None:
        fid_metric = pyiqa.create_metric("fid").to(device)
        result["fid"] = fid_metric(in_path, ref_path)

    print_results = []
    for key, res in result.items():
        if key == "fid":
            print(f"{key}: {res:.2f}")
            print_results.append(f"{key}: {res:.2f}")
        else:
            print(f"{key}: {res/len(lr_path_list):.5f}")
            print_results.append(f"{key}: {res/len(lr_path_list):.5f}")
    return print_results


def main(args):
    config = OmegaConf.load(args.test_config)

    if args.sd_path is None:
        from huggingface_hub import snapshot_download

        sd_path = snapshot_download(repo_id="stabilityai/stable-diffusion-2-1")
    else:
        sd_path = args.sd_path

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    # initialize net_sr
    net_sr = TADM_Tile(
        sd_path=sd_path,
        pretrained_path=args.pretrained_path,
        args=args,
    )
    net_sr.set_eval()

    # initialize degradation estimation network
    net_dfrm = DFRM(4, 3, args.scale_factor)
    dfrm_state_dict = torch.load(args.dfrm_net_path, map_location=torch.device("cpu"))
    net_dfrm.load_state_dict(dfrm_state_dict, strict=True)
    net_dfrm = net_dfrm.cuda()
    net_dfrm.eval()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_sr.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available, please install it by running `pip install xformers`"
            )

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    dataset_test = SingleImageDataset(config.test)
    dl_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0
    )

    # Prepare everything with our `accelerator`.
    net_sr, net_dfrm = accelerator.prepare(net_sr, net_dfrm)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move al networksr to device and cast to weight_dtype
    net_sr.to(accelerator.device, dtype=weight_dtype)
    net_dfrm.to(accelerator.device, dtype=weight_dtype)

    out_sr_path = os.path.join(args.output_dir, "sr")
    os.makedirs(out_sr_path, exist_ok=True)
    out_lr_path = os.path.join(args.output_dir, "lr")
    os.makedirs(out_lr_path, exist_ok=True)

    for step, batch_test in enumerate(dl_test):
        gt_path = batch_test["gt_path"][0]
        (path, name) = os.path.split(gt_path)

        x_tgt = (
            batch_test["gt"]
            .cuda()
            .to(dtype=weight_dtype, memory_format=torch.contiguous_format)
            .contiguous()
        )

        ori_h, ori_w = x_tgt.shape[2:]

        x_tgt = torch.clamp(x_tgt, -1.0, 1.0)

        pad_h = (math.ceil(ori_h / 64)) * 64 - ori_h
        pad_w = (math.ceil(ori_w / 64)) * 64 - ori_w
        x_tgt = F.pad(x_tgt, pad=(0, pad_w, 0, pad_h), mode="reflect")

        B = x_tgt.size(0)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                # forward pass
                start_time = time.time()
                x_latent = (
                    net_sr.vae.encode(x_tgt).latent_dist.sample()
                    * net_sr.vae.config.scaling_factor
                )
                res_dict = net_dfrm(x_tgt, x_latent)
                recov_latent = res_dict["z_back_recon"]
                recov_lr = res_dict["lr"]

                x_tgt_pred = accelerator.unwrap_model(net_sr)(
                    recov_latent, batch_test["txt"]
                )
                end_time = time.time()
                print(f"Time: {end_time - start_time} s")
                
                x_tgt_pred = x_tgt_pred[:, :, :ori_h, :ori_w]
                out_img_sr = (x_tgt_pred * 0.5 + 0.5).cpu().detach()
                out_img_lr = (recov_lr * 0.5 + 0.5).cpu().detach()

        output_sr_pil = transforms.ToPILImage()(out_img_sr[0])
        output_lr_pil = transforms.ToPILImage()(out_img_lr[0])

        fname, ext = os.path.splitext(name)
        outf_sr = os.path.join(out_sr_path, fname + ".png")
        output_sr_pil.save(outf_sr)
        outf_lr = os.path.join(out_lr_path, fname + ".png")
        output_lr_pil.save(outf_lr)

    print_results = evaluate(out_sr_path, config.test.hq_dataroot, None)
    out_t = os.path.join(args.output_dir, "results.txt")
    with open(out_t, "w", encoding="utf-8") as f:
        for item in print_results:
            f.write(f"{item}\n")
            
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_args_rescale_testing()
    main(args)
