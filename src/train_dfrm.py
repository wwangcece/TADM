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
from my_utils.training_utils import (
    parse_args_paired_training_dfrm,
)
from src.dataset.SingleImageDataset import SingleImageDataset


def main(args):

    # init and save configs
    config = OmegaConf.load(args.base_config)

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
    if args.pretrained_path is not None:
        dfrm_state_dict = torch.load(
            args.pretrained_path, map_location=torch.device("cpu")
        )
        net_dfrm.load_state_dict(dfrm_state_dict, strict=True)
    net_dfrm = net_dfrm.cuda()
    net_dfrm.requires_grad_(True)
    net_dfrm.train()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # make the optimizer
    layers_to_opt = list(net_dfrm.parameters())
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
        net_dfrm,
        optimizer,
        dl_train,
        lr_scheduler,
    ) = accelerator.prepare(
        net_dfrm,
        optimizer,
        dl_train,
        lr_scheduler,
    )
    net_dfrm = accelerator.prepare(net_dfrm)
    # # renorm with image net statistics
    weight_dtype = torch.float32

    # Move al networksr to device and cast to weight_dtype
    net_dfrm.to(accelerator.device, dtype=weight_dtype)

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
                l_acc = [net_dfrm]
                with accelerator.accumulate(*l_acc):
                    x_tgt = batch["gt"].to(accelerator.device)
                    z_origin = batch["hq_latent"].to(accelerator.device)
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
                    recov_lr, z_forward_recon = net_dfrm(x_tgt, z_origin, rev=False, train=True)
                    z_back_recon = net_dfrm(
                        recov_lr, z_origin, rev=True, train=True
                    )

                    loss_lr = (
                        (
                            F.mse_loss(x_lr.float(), recov_lr.float(), reduction="mean")
                            * args.lambda_lr
                        )
                    )
                    loss_z_forward = (
                        (
                            F.mse_loss(
                                z_forward_recon.float(),
                                z_origin.float(),
                                reduction="mean",
                            )
                            * args.lambda_z
                        )
                    )
                    loss_z_back = (
                        (
                            F.mse_loss(
                                z_back_recon.float(),
                                z_origin.float(),
                                reduction="mean",
                            )
                            * args.lambda_z
                        )
                    )

                    loss = loss_lr + loss_z_forward + loss_z_back

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
                        logs["loss_lr"] = loss_lr.detach().item()
                        logs["loss_z_forward"] = loss_z_forward.detach().item()
                        logs["loss_z_back"] = loss_z_back.detach().item()
                        progress_bar.set_postfix(**logs)

                        # checkpoint the model
                        if global_step % args.checkpointing_steps == 1:
                            outf_dfrm = os.path.join(
                                args.output_dir,
                                "checkpoints",
                                f"model_dfrm_{global_step}.pkl",
                            )
                            accelerator.unwrap_model(net_dfrm).save_model(outf_dfrm)

                        # compute validation set FID, L2, LPIPS, CLIP-SIM
                        if global_step % args.eval_freq == 1:
                            print("**********evaluating!**********")
                            l_l2, l_z_forward, l_z_back = (
                                [],
                                [],
                                [],
                            )

                            val_count = 0
                            for step, batch_val in enumerate(dl_val):
                                if step >= args.num_samples_eval:
                                    break
                                x_tgt = batch_val["gt"].to(accelerator.device)
                                z_origin = (
                                    batch_val["hq_latent"].to(accelerator.device)
                                )
                                B, C, H, W = x_tgt.shape
                                assert B == 1, "Use batch size 1 for eval."
                                with torch.no_grad():
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
                                    recov_lr, z_forward_recon = net_dfrm(x_tgt, z_origin, rev=False, train=False)
                                    z_back_recon = net_dfrm(
                                        recov_lr, z_origin, rev=True, train=False
                                    )

                                    # compute the reconstruction losses
                                    loss_lr = (
                                        (
                                            F.mse_loss(x_lr.float(), recov_lr.float(), reduction="mean")
                                            * args.lambda_lr
                                        )
                                    )
                                    loss_z_forward = (
                                        (
                                            F.mse_loss(
                                                z_forward_recon.float(),
                                                z_origin.float(),
                                                reduction="mean",
                                            )
                                            * args.lambda_z
                                        )
                                    )
                                    loss_z_back = (
                                        (
                                            F.mse_loss(
                                                z_back_recon.float(),
                                                z_origin.float(),
                                                reduction="mean",
                                            )
                                            * args.lambda_z
                                        )
                                    )

                                    loss = loss_lr + loss_z_forward + loss_z_back

                                    l_l2.append(loss_lr.item())
                                    l_z_forward.append(loss_z_forward.item())
                                    l_z_back.append(loss_z_back.item())

                                if args.save_val and val_count < 10:
                                    z_origin = z_origin.cpu().detach() * 0.5 + 0.5
                                    z_back_recon = z_back_recon.cpu().detach() * 0.5 + 0.5
                                    x_recov_lr = recov_lr.cpu().detach() * 0.5 + 0.5

                                    combined = torch.cat(
                                        [z_origin, z_back_recon], dim=3
                                    )
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
                            logs["val/lpips"] = np.mean(l_z_forward)
                            logs["val/dists"] = np.mean(l_z_back)

                            gc.collect()
                            torch.cuda.empty_cache()
                        accelerator.log(logs, step=global_step)


if __name__ == "__main__":
    args = parse_args_paired_training_dfrm()
    main(args)
