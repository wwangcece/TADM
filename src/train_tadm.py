import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys

sys.path.append("/mnt/massive/wangce/ArbiRescale/S3Diff")
import pyiqa
import gc
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from DISTS_pytorch import DISTS
import lpips

from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import set_seed
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration
from modules.ae_arch_inv_cnn import DFRM
from tadm import TADM
from my_utils.training_utils import (
    parse_args_paired_training_diff,
    PairedDataset,
    degradation_proc,
)
from src.dataset.SingleImageDataset import SingleImageDataset


def main(args):

    # init and save configs
    config = OmegaConf.load(args.base_config)

    if args.sd_path is None:
        from huggingface_hub import snapshot_download

        sd_path = snapshot_download(repo_id="stabilityai/stable-diffusion-2-1")
    else:
        sd_path = args.sd_path

    os.makedirs(args.output_dir, exist_ok=True)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    pro_config = ProjectConfiguration(project_dir=args.output_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=args.output_dir,
        project_config=pro_config,
        kwargs_handlers=[kwargs],
    )
    accelerator.init_trackers("logs")

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    # initialize degradation estimation network
    net_dfrm = DFRM(4, 3, args.scale_factor)
    dfrm_state_dict = torch.load(args.dfrm_net_path, map_location=torch.device("cpu"))
    net_dfrm.load_state_dict(dfrm_state_dict, strict=True)
    net_dfrm = net_dfrm.cuda()
    if args.finetune:
        net_dfrm.requires_grad_(True)
        net_dfrm.train()
    else:
        net_dfrm.requires_grad_(False)
        net_dfrm.eval()

    # initialize net_sr
    net_sr = TADM(
        lora_rank_unet=args.lora_rank_unet,
        lora_rank_vae=args.lora_rank_vae,
        sd_path=sd_path,
        pretrained_path=args.pretrained_path,
    )
    net_sr.set_train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_sr.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available, please install it by running `pip install xformers`"
            )

    if args.gradient_checkpointing:
        net_sr.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    net_lpips = lpips.LPIPS(net="vgg").cuda()
    net_lpips.requires_grad_(False)
    net_dists = DISTS().cuda()
    net_dists.requires_grad_(False)

    # make the optimizer
    layers_to_opt = []
    for n, _p in net_sr.unet.named_parameters():
        if "lora" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    layers_to_opt += list(net_sr.unet.conv_in.parameters())

    for n, _p in net_sr.vae.named_parameters():
        if "lora" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)

    if hasattr(net_sr, "timemapping"):
        for n, _p in net_sr.timemapping.named_parameters():
            assert _p.requires_grad
            layers_to_opt.append(_p)

    if hasattr(net_sr, "refiner"):
        for n, _p in net_sr.refiner.named_parameters():
            assert _p.requires_grad
            layers_to_opt.append(_p)

    layers_to_opt += list(net_sr.unet.conv_in.parameters())
    if args.finetune:
        for n, _p in net_dfrm.named_parameters():
            if _p.requires_grad:
                layers_to_opt.append(_p)
    total_params = sum(p.numel() for p in layers_to_opt)
    print(
        f"**********total trainable parameters: {total_params // 1e6} M **************"
    )

    dataset_train = SingleImageDataset(config.train)
    dl_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )
    dataset_val = SingleImageDataset(config.validation)
    dl_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, shuffle=False, num_workers=0
    )

    optimizer = torch.optim.AdamW(
        layers_to_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        step_rules=args.lr_step_rules,
    )

    # Prepare everything with our `accelerator`.
    (
        net_sr,
        optimizer,
        dl_train,
        lr_scheduler,
    ) = accelerator.prepare(
        net_sr,
        optimizer,
        dl_train,
        lr_scheduler,
    )
    net_dfrm, net_lpips, net_dists = accelerator.prepare(
        net_dfrm, net_lpips, net_dists
    )
    # # renorm with image net statistics
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move al networksr to device and cast to weight_dtype
    net_sr.to(accelerator.device, dtype=weight_dtype)
    net_dfrm.to(accelerator.device, dtype=weight_dtype)
    net_lpips.to(accelerator.device, dtype=weight_dtype)
    net_dists.to(accelerator.device, dtype=weight_dtype)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=0,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # start the training loop
    global_step = 0
    with accelerator.autocast():
        for epoch in range(0, args.num_training_epochs):
            for step, batch in enumerate(dl_train):
                l_acc = [net_sr]
                with accelerator.accumulate(*l_acc):
                    x_tgt = batch["gt"].to(accelerator.device)
                    x_latent = batch["hq_latent"].to(accelerator.device)
                    x_lr = (
                        F.interpolate(
                            x_tgt,
                            size=(
                                int(x_tgt.shape[-2] // args.scale_factor),
                                int(x_tgt.shape[-1] // args.scale_factor),
                            ),
                            mode="bicubic",
                        )
                        .to(accelerator.device)
                        .clamp(-1, 1)
                    )
                    B, C, H, W = x_tgt.shape

                    if args.finetune:
                        res_dict = net_dfrm(x_tgt, x_latent)
                        recov_latent = res_dict["z_back_recon"]
                        recov_lr = res_dict["lr"]
                    else:
                        with torch.no_grad():
                            res_dict = net_dfrm(x_tgt, x_latent)
                            recov_latent = res_dict["z_back_recon"]
                            recov_lr = res_dict["lr"]

                    """
                    Generator loss: fool the discriminator
                    """
                    tag_prompt = [args.pos_prompt for _ in range(B)]

                    x_tgt_pred, x_latent_denoised, timesteps = net_sr(
                        recov_latent, tag_prompt
                    )
                    loss_l2 = (
                        F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean")
                        * args.lambda_l2
                    )
                    loss_lpips = (
                        net_lpips(x_tgt_pred.float(), x_tgt.float()).mean()
                        * args.lambda_lpips
                    )
                    loss_dists = (
                        net_dists(
                            x_tgt_pred.float(),
                            x_tgt.float(),
                            require_grad=True,
                            batch_average=True,
                        )
                        * args.lambda_lpips
                    )
                    loss_lr = (
                        (
                            F.mse_loss(x_lr.float(), recov_lr.float(), reduction="mean")
                            * args.lambda_lr
                        )
                        if args.finetune
                        else 0
                    )
                    loss_latent1 = (
                        (
                            F.mse_loss(
                                recov_latent.float() * 0.18215,
                                x_latent.float(),
                                reduction="mean",
                            )
                            * args.lambda_lr
                        )
                        if args.finetune
                        else 0
                    )

                    loss = (
                        loss_l2
                        + loss_lpips
                        + loss_lr
                        + loss_latent1
                        + loss_dists
                    )

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if accelerator.is_main_process:
                        logs = {}
                        logs["loss_l2"] = loss_l2.detach().item()
                        logs["loss_lpips"] = loss_lpips.detach().item()
                        logs["loss_dists"] = loss_dists.detach().item()
                        if args.finetune:
                            logs["loss_latent1"] = loss_latent1.detach().item()
                            logs["loss_lr"] = loss_lr.detach().item()
                        progress_bar.set_postfix(**logs)

                        # checkpoint the model
                        if global_step % args.checkpointing_steps == 1:
                            outf_sr = os.path.join(
                                args.output_dir,
                                "checkpoints",
                                f"model_sr_{global_step}.pkl",
                            )
                            outf_dfrm = os.path.join(
                                args.output_dir,
                                "checkpoints",
                                f"model_dfrm_{global_step}.pkl",
                            )
                            accelerator.unwrap_model(net_sr).save_model(outf_sr)
                            if args.finetune:
                                accelerator.unwrap_model(net_dfrm).save_model(outf_dfrm)

                        # compute validation set FID, L2, LPIPS, CLIP-SIM
                        if global_step % args.eval_freq == 1:
                            print("**********evaluating!**********")
                            l_l2, l_lpips, l_dists = (
                                [],
                                [],
                                [],
                            )

                            val_count = 0
                            for step, batch_val in enumerate(dl_val):
                                if step >= args.num_samples_eval:
                                    break
                                x_tgt = batch_val["gt"].to(accelerator.device)
                                x_latent = (
                                    batch_val["hq_latent"].to(accelerator.device)
                                    * 0.18215
                                )
                                B, C, H, W = x_tgt.shape
                                assert B == 1, "Use batch size 1 for eval."
                                with torch.no_grad():
                                    res_dict = net_dfrm(x_tgt, x_latent)
                                    recov_latent = res_dict["z_back_recon"]
                                    recov_lr = res_dict["lr"]

                                    tag_prompt = [args.pos_prompt for _ in range(B)]

                                    x_tgt_pred, _, _ = accelerator.unwrap_model(net_sr)(
                                        recov_latent, tag_prompt
                                    )
                                    # compute the reconstruction losses
                                    loss_l2 = F.mse_loss(
                                        x_tgt_pred.float(),
                                        x_tgt.detach().float(),
                                        reduction="mean",
                                    )
                                    loss_lpips = net_lpips(
                                        x_tgt_pred.float(), x_tgt.detach().float()
                                    ).mean()
                                    loss_dists = net_dists(
                                        x_tgt_pred.float(), x_tgt.detach().float()
                                    ).mean()

                                    l_l2.append(loss_l2.item())
                                    l_lpips.append(loss_lpips.item())
                                    l_dists.append(loss_dists.item())

                                if args.save_val and val_count < 10:
                                    x_tgt = x_tgt.cpu().detach() * 0.5 + 0.5
                                    x_tgt_pred = x_tgt_pred.cpu().detach() * 0.5 + 0.5
                                    x_recov_lr = recov_lr.cpu().detach() * 0.5 + 0.5

                                    combined = torch.cat([x_tgt_pred, x_tgt], dim=3)
                                    output_pil = transforms.ToPILImage()(combined[0])
                                    lr_pil = transforms.ToPILImage()(x_recov_lr[0])

                                    img_out_path = os.path.join(
                                        args.output_dir, "eval", str(global_step)
                                    )
                                    os.makedirs(img_out_path, exist_ok=True)
                                    outf = os.path.join(
                                        img_out_path, f"val_{step}_sr.png"
                                    )
                                    output_pil.save(outf)
                                    outf = os.path.join(
                                        img_out_path, f"val_{step}_lr.png"
                                    )
                                    lr_pil.save(outf)

                                    val_count += 1

                            logs["val/l2"] = np.mean(l_l2)
                            logs["val/lpips"] = np.mean(l_lpips)
                            logs["val/dists"] = np.mean(l_dists)

                            gc.collect()
                            torch.cuda.empty_cache()
                        accelerator.log(logs, step=global_step)


if __name__ == "__main__":
    args = parse_args_paired_training_diff()
    main(args)
