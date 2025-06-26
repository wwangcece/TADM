import torch
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, UNet2DModel
from peft import LoraConfig, get_peft_model
from model import make_1step_sched, my_lora_fwd
from tpm_patch import Refiner, TimeMapping

# model = UNet2DModel(in_channels=8, out_channels=4, down_block_types=("DownBlock2D","DownBlock2D", "DownBlock2D")


class TADM(torch.nn.Module):

    def __init__(
        self,
        sd_path=None,
        pretrained_path=None,
        lora_rank_unet=32,
        lora_rank_vae=16,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            sd_path, subfolder="text_encoder"
        ).cuda()
        self.sched = make_1step_sched(sd_path)

        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet")
        timemapping = TimeMapping(out_channels=1)
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

            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)

            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

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

        self.lora_rank_unet = lora_rank_unet
        self.lora_rank_vae = lora_rank_vae
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

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()

        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)

        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True

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
        timesteps = self.timemapping(c_t)

        encoded_control = c_t * self.vae.config.scaling_factor
        model_pred = self.unet(
            encoded_control,
            timesteps,
            encoder_hidden_states=caption_enc,
        ).sample

        # denoise with fixed and learned scheduler
        x_denoised_learned = self.refiner(
            encoded_control,
            model_pred,
            timesteps,
        )
        x_denoised_sched = self.sched.step(
            model_pred, self.timesteps, encoded_control, return_dict=True
        ).prev_sample
        x_denoised = x_denoised_learned + x_denoised_sched

        output_image = (
            self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample
        ).clamp(-1, 1)

        return output_image, x_denoised / self.vae.config.scaling_factor, timesteps

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
