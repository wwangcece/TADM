import os
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import argparse
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from diffusers.models import AutoencoderKL
from tqdm import tqdm


def load_image(path, image_size):
    image = Image.open(path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),  # Converts to [0,1]
            transforms.Normalize([0.5], [0.5]),  # Normalize to [-1,1]
        ]
    )
    return transform(image).unsqueeze(0)  # Shape: [1, 3, H, W]


def extract_latents(model, image_tensor):
    with torch.no_grad():
        image_tensor = image_tensor.to(model.device)
        latents = model.encode(image_tensor).latent_dist.sample()
        latents = latents * 0.18215  # scale factor used in SD
    return latents.squeeze(0).cpu().numpy()  # [C, H, W]


def main(args):
    # Load pretrained VAE (from Stable Diffusion)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").cuda().eval()

    os.makedirs(args.out_dir, exist_ok=True)

    # List all images
    image_paths = []
    if os.path.isfile(args.in_path):
        image_paths = [args.in_path]
    else:
        for fname in os.listdir(args.in_path):
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                image_paths.append(os.path.join(args.in_path, fname))
        image_paths.sort()

    for img_path in tqdm(image_paths, desc="Processing images"):
        try:
            image_tensor = load_image(img_path, args.image_size)
            latent = extract_latents(vae, image_tensor)
            # Save .npy
            basename = os.path.splitext(os.path.basename(img_path))[0]
            np.save(os.path.join(args.out_dir, f"{basename}.npy"), latent)
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_path",
        type=str,
        required=True,
        help="Path to an image or a folder of images",
    )
    parser.add_argument(
        "--out_dir", type=str, required=True, help="Directory to save .npy latent files"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Image resolution to resize (default: 512)",
    )

    args = parser.parse_args()
    main(args)
