import os
import re
import requests
import sys
import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig, get_peft_model
from model import make_1step_sched, my_lora_fwd
from my_utils.vaehook import VAEHook
from numpy import pi, exp, sqrt
import numpy as np
import torch.nn.functional as F
from tpm_patch import Refiner, TimeMapping


def get_layer_number(module_name):
    base_layers = {"down_blocks": 0, "mid_block": 4, "up_blocks": 5}

    if module_name == "conv_out":
        return 9

    base_layer = None
    for key in base_layers:
        if key in module_name:
            base_layer = base_layers[key]
            break

    if base_layer is None:
        return None

    additional_layers = int(
        re.findall(r"\.(\d+)", module_name)[0]
    )  # sum(int(num) for num in re.findall(r'\d+', module_name))
    final_layer = base_layer + additional_layers
    return final_layer


class TADM_Tile(torch.nn.Module):

    def __init__(
        self,
        sd_path=None,
        pretrained_path=None,
        lora_rank_unet=48,
        lora_rank_vae=48,
        args=None,
    ):
        super().__init__()
        self.args = args
        self.latent_tiled_size = args.latent_tiled_size
        self.latent_tiled_overlap = args.latent_tiled_overlap

        self.tokenizer = AutoTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            sd_path, subfolder="text_encoder"
        )
        self.sched = make_1step_sched(sd_path)

        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet")
        timemapping = TimeMapping()
        refiner = Refiner()

        target_modules_unet = [
            "to_k",
            "to_q",
            "to_v",
            "to_out.0",
            "conv",
            "conv1",
            "conv2",
            "conv_shortcut",
            "conv_out",
            "proj_in",
            "proj_out",
            "ff.net.2",
            "ff.net.0.proj",
        ]
        target_modules_vae = r"^decoder\..*(conv1|conv2|conv_in|conv_shortcut|conv|conv_out|to_k|to_q|to_v|to_out\.0)$"

        if pretrained_path is not None:
            # resume training, including lora params
            sd = torch.load(pretrained_path, map_location="cpu")
            # sd_time = torch.load(
            #     "experiments/diff-final-x16-resume/checkpoints/model_sr_1.pkl",
            #     map_location="cpu",
            # )

            unet_lora_config = LoraConfig(
                r=sd["rank_unet"],
                init_lora_weights="gaussian",
                target_modules=sd["unet_lora_target_modules"],
            )
            unet.add_adapter(unet_lora_config)

            vae_lora_config = LoraConfig(
                r=sd["rank_vae"],
                init_lora_weights="gaussian",
                target_modules=sd["vae_lora_target_modules"],
            )
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")

            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)

            _sd_timemapping = timemapping.state_dict()
            for k in sd["state_dict_timemapping"]:
                _sd_timemapping[k] = sd["state_dict_timemapping"][k]
            timemapping.load_state_dict(_sd_timemapping)

            _sd_refiner = refiner.state_dict()
            for k in sd["state_dict_refiner"]:
                _sd_refiner[k] = sd["state_dict_refiner"][k]
            refiner.load_state_dict(_sd_refiner)
        else:
            # training RescaleDiff from scratch
            print("Initializing model with random weights")
            unet_lora_config = LoraConfig(
                r=lora_rank_unet,
                init_lora_weights="gaussian",
                target_modules=target_modules_unet,
            )
            unet.add_adapter(unet_lora_config)

            vae_lora_config = LoraConfig(
                r=lora_rank_vae,
                init_lora_weights="gaussian",
                target_modules=target_modules_vae,
            )
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")

        self.lora_rank_vae = lora_rank_vae
        self.lora_rank_unet = lora_rank_unet
        self.target_modules_vae = target_modules_vae
        self.target_modules_unet = target_modules_unet

        unet.to("cuda")
        vae.to("cuda")
        timemapping.to("cuda")
        refiner.to("cuda")
        self.unet, self.vae, self.timemapping, self.refiner = (
            unet,
            vae,
            timemapping,
            refiner,
        )
        self.timesteps = torch.tensor([50], device="cuda").long()
        self.text_encoder.requires_grad_(False)

        # vae tile
        self._init_tiled_vae(
            encoder_tile_size=args.vae_encoder_tiled_size,
            decoder_tile_size=args.vae_decoder_tiled_size,
        )

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def _init_tiled_vae(
        self,
        encoder_tile_size=256,
        decoder_tile_size=256,
        fast_decoder=False,
        fast_encoder=False,
        color_fix=False,
        vae_to_gpu=True,
    ):
        # save original forward (only once)
        if not hasattr(self.vae.encoder, "original_forward"):
            setattr(self.vae.encoder, "original_forward", self.vae.encoder.forward)
        if not hasattr(self.vae.decoder, "original_forward"):
            setattr(self.vae.decoder, "original_forward", self.vae.decoder.forward)

        encoder = self.vae.encoder
        decoder = self.vae.decoder

        self.vae.encoder.forward = VAEHook(
            encoder,
            encoder_tile_size,
            is_decoder=False,
            fast_decoder=fast_decoder,
            fast_encoder=fast_encoder,
            color_fix=color_fix,
            to_gpu=vae_to_gpu,
        )
        self.vae.decoder.forward = VAEHook(
            decoder,
            decoder_tile_size,
            is_decoder=True,
            fast_decoder=fast_decoder,
            fast_encoder=fast_encoder,
            color_fix=color_fix,
            to_gpu=vae_to_gpu,
        )

    def set_train(self):
        self.unet.train()

        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True

        self.unet.conv_in.requires_grad_(True)

    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        """Generates a gaussian mask of weights for tile contributions"""

        latent_width = tile_width
        latent_height = tile_height

        var = 0.01
        midpoint = (
            latent_width - 1
        ) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [
            exp(
                -(x - midpoint)
                * (x - midpoint)
                / (latent_width * latent_width)
                / (2 * var)
            )
            / sqrt(2 * pi * var)
            for x in range(latent_width)
        ]
        midpoint = latent_height / 2
        y_probs = [
            exp(
                -(y - midpoint)
                * (y - midpoint)
                / (latent_height * latent_height)
                / (2 * var)
            )
            / sqrt(2 * pi * var)
            for y in range(latent_height)
        ]

        weights = np.outer(y_probs, x_probs)
        return torch.tile(
            torch.tensor(weights), (nbatches, self.unet.config.in_channels, 1, 1)
        )

    def forward(self, c_t, prompt=""):

        # encode the text prompt
        caption_tokens = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.cuda()
        caption_enc = self.text_encoder(caption_tokens)[0]

        lq_latent = c_t * self.vae.config.scaling_factor

        ## add tile function
        b, c, h, w = lq_latent.size()
        tile_size, tile_overlap = (self.latent_tiled_size, self.latent_tiled_overlap)
        if h * w <= tile_size * tile_size:
            print(f"[Tiled Latent]: the input size is tiny and unnecessary to tile.")
            timesteps = self.timemapping(lq_latent)
            print(f"sampling from {timesteps[0]}")

            model_pred = self.unet(
                lq_latent, timesteps, encoder_hidden_states=caption_enc
            ).sample
            x_denoised_learned = self.refiner(
                lq_latent,
                model_pred,
                timesteps,
            )
            x_denoised_sched = self.sched.step(
                model_pred, self.timesteps, lq_latent, return_dict=True
            ).prev_sample

            x_denoised_pred = x_denoised_learned + x_denoised_sched
            timesteps_pred = (
                torch.zeros((b, c, h, w), device=lq_latent.device) + timesteps
            )
        else:
            print(
                f"[Tiled Latent]: the input size is {c_t.shape[-2]}x{c_t.shape[-1]}, need to tiled"
            )
            # tile_weights = self._gaussian_weights(tile_size, tile_size, 1).to()
            tile_size = min(tile_size, min(h, w))
            tile_weights = self._gaussian_weights(tile_size, tile_size, 1).to(
                c_t.device
            )

            grid_rows = 0
            cur_x = 0
            while cur_x < lq_latent.size(-1):
                cur_x = (
                    max(grid_rows * tile_size - tile_overlap * grid_rows, 0) + tile_size
                )
                grid_rows += 1

            grid_cols = 0
            cur_y = 0
            while cur_y < lq_latent.size(-2):
                cur_y = (
                    max(grid_cols * tile_size - tile_overlap * grid_cols, 0) + tile_size
                )
                grid_cols += 1

            input_list = []
            noise_preds = []
            timesteps_list = []
            x_denoised_list = []
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if col < grid_cols - 1 or row < grid_rows - 1:
                        # extract tile from input image
                        ofs_x = max(row * tile_size - tile_overlap * row, 0)
                        ofs_y = max(col * tile_size - tile_overlap * col, 0)
                        # input tile area on total image
                    if row == grid_rows - 1:
                        ofs_x = w - tile_size
                    if col == grid_cols - 1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    # input tile dimensions
                    input_tile = lq_latent[
                        :, :, input_start_y:input_end_y, input_start_x:input_end_x
                    ]
                    input_list.append(input_tile)

                    if len(input_list) == 1 or col == grid_cols - 1:
                        input_list_t = torch.cat(input_list, dim=0)
                        timesteps = self.timemapping(input_list_t)
                        # print(f"sampling from {timesteps[0]}")
                        # predict the noise residual
                        noise_pred = self.unet(
                            input_list_t,
                            timesteps,
                            encoder_hidden_states=caption_enc,
                        ).sample
                        x_denoised_learned = self.refiner(
                            input_list_t,
                            noise_pred,
                            timesteps,
                        )
                        x_denoised_sched = self.sched.step(
                            noise_pred, self.timesteps, input_list_t, return_dict=True
                        ).prev_sample
                        x_denoised = x_denoised_learned + x_denoised_sched

                        input_list = []
                    noise_preds.append(noise_pred)
                    if len(timesteps.shape) == 1:
                        timesteps = timesteps.view(1, 1, 1, 1)
                    timesteps = F.interpolate(
                        timesteps,
                        size=(noise_pred.shape[2], noise_pred.shape[3]),
                        mode="nearest",
                    )
                    timesteps_list.append(timesteps)
                    x_denoised_list.append(x_denoised)

            # Stitch noise predictions for all tiles
            x_denoised_pred = torch.zeros(lq_latent.shape, device=lq_latent.device)
            contributors = torch.zeros(lq_latent.shape, device=lq_latent.device)
            timesteps_pred = torch.zeros((b, c, h, w), device=lq_latent.device)
            timesteps_contri = torch.zeros((b, c, h, w), device=lq_latent.device)
            # Add each tile contribution to overall latents
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if col < grid_cols - 1 or row < grid_rows - 1:
                        # extract tile from input image
                        ofs_x = max(row * tile_size - tile_overlap * row, 0)
                        ofs_y = max(col * tile_size - tile_overlap * col, 0)
                        # input tile area on total image
                    if row == grid_rows - 1:
                        ofs_x = w - tile_size
                    if col == grid_cols - 1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    x_denoised_pred[
                        :, :, input_start_y:input_end_y, input_start_x:input_end_x
                    ] += (x_denoised_list[row * grid_cols + col] * tile_weights)
                    timesteps_pred[
                        :,
                        :,
                        input_start_y:input_end_y,
                        input_start_x:input_end_x,
                    ] += timesteps_list[row * grid_cols + col]
                    contributors[
                        :, :, input_start_y:input_end_y, input_start_x:input_end_x
                    ] += tile_weights
                    timesteps_contri[
                        :,
                        :,
                        input_start_y:input_end_y,
                        input_start_x:input_end_x,
                    ] += 1

            # Average overlapping areas with more than 1 contributor
            x_denoised_pred /= contributors
            timesteps_pred /= timesteps_contri

        output_image = (
            self.vae.decode(x_denoised_pred / self.vae.config.scaling_factor).sample
        ).clamp(-1, 1)

        return output_image

    def save_model(self, outf):
        sd = {}
        sd["unet_lora_target_modules"] = self.target_modules_unet
        sd["vae_lora_target_modules"] = self.target_modules_vae

        sd["rank_vae"] = self.lora_rank_vae
        sd["rank_unet"] = self.lora_rank_unet

        sd["state_dict_vae"] = {
            k: v
            for k, v in self.vae.state_dict().items()
            if "lora" in k or "skip_conv" in k
        }
        sd["state_dict_unet"] = {
            k: v
            for k, v in self.unet.state_dict().items()
            if "lora" in k or "conv_in" in k
        }

        sd["state_dict_timemapping"] = {
            k: v for k, v in self.timemapping.state_dict().items()
        }
        sd["state_dict_refiner"] = {k: v for k, v in self.refiner.state_dict().items()}

        torch.save(sd, outf)
