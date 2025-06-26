import os
from typing import Dict
import time
import numpy as np
import torch
from torch.utils import data
from PIL import Image
from .image import augment, random_crop_arr, center_crop_arr

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
    ".npy",
    ".txt"
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_paths_from_images(path):
    assert os.path.isdir(path), "{:s} is not a valid directory".format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, "{:s} has no valid image file".format(path)
    return sorted(images)


class SingleImageDataset(data.Dataset):
    """
    # TODO: add comment
    """

    def __init__(
        self,
        opt
    ) -> "SingleImageDataset":
        super(SingleImageDataset, self).__init__()
        self.hq_paths = get_paths_from_images(opt['hq_dataroot'])[:opt['data_len']]
        self.hq_latent_paths = (
            get_paths_from_images(opt['hq_latent_dataroot'])[:opt['data_len']]
            if opt['hq_latent_dataroot'] != ""
            else None
        )
        self.hq_text_paths = (
            get_paths_from_images(opt['hq_text_dataroot'])[:opt['data_len']]
            if opt['hq_text_dataroot'] != ""
            else None
        )
        self.lq_paths = (
            get_paths_from_images(opt["lq_dataroot"])[: opt["data_len"]]
            if opt["lq_dataroot"] != ""
            else None
        )

        self.crop_type = opt['crop_type']
        assert self.crop_type in [
            "center",
            "random",
            "none",
        ], f"invalid crop type: {self.crop_type}"
        self.use_hflip = opt['use_hflip']
        self.use_rot = opt['use_rot']
        self.out_size = opt['out_size']
        self.data_len = opt['data_len']

    @torch.no_grad()
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # -------------------------------- Load hq images and lq images-------------------------------- #
        hq_path = self.hq_paths[index]
        hq_latent_path = self.hq_latent_paths[index] if self.hq_latent_paths is not None else None
        hq_text_path = self.hq_text_paths[index] if self.hq_text_paths is not None else None
        lq_path = (
            self.lq_paths[index] if self.lq_paths is not None else None
        )
        success = False
        for _ in range(3):
            try:
                hq_img = Image.open(hq_path).convert("RGB")
                lq_img = Image.open(lq_path).convert("RGB") if lq_path is not None else None
                if hq_latent_path:
                    hq_latent = np.load(hq_latent_path)
                    if hq_latent.shape[0] == 4:
                        hq_latent = hq_latent.transpose(1, 2, 0)
                else:
                    hq_latent = None
                success = True
                break
            except:
                time.sleep(1)
        assert success, f"failed to load image {hq_path}"

        if self.crop_type == "random":
            img_list = [hq_img, lq_img] if lq_path is not None else [hq_img]
            img_list, crop_y, crop_x = random_crop_arr(img_list, self.out_size)
            hq_img, lq_img = img_list if lq_path is not None else (img_list[0], None)
            hq_latent = (
                hq_latent[
                    crop_y // 8 : (crop_y + self.out_size) // 8,
                    crop_x // 8 : (crop_x + self.out_size) // 8,
                    :
                ]
                if hq_latent is not None
                else None
            )
        elif self.crop_type == "center":
            hq_img = center_crop_arr(hq_img, self.out_size)
            lq_img = center_crop_arr(lq_img, self.out_size) if lq_path is not None else None
            hq_latent = (
                center_crop_arr(hq_latent, self.out_size // 8) if hq_latent is not None else None
            )
        # self.crop_type is "none"
        else:
            hq_img = np.array(hq_img)
            lq_img = np.array(lq_img) if lq_path is not None else None
            # assert hq_img.shape[:2] == (self.out_size, self.out_size)

        # hwc, rgb to bgr, [0, 255] to [0, 1], float32
        img_hq = (hq_img[..., ::-1] / 255.0).astype(np.float32)
        img_lq = (
            (lq_img[..., ::-1] / 255.0).astype(np.float32)
            if lq_path is not None
            else None
        )
        hq_latent = hq_latent.astype(np.float32) if hq_latent is not None else None
        if hq_text_path is not None:
            with open(hq_text_path, 'r', encoding='utf-8') as file:
                hq_text = file.read()
        else:
            hq_text = ""

        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        if hq_latent is not None:
            # no augmentation
            pass
        else:
            img_hq = augment(img_hq, self.use_hflip, self.use_rot)

        # [0, 1], BGR to RGB, HWC to CHW
        img_hq = (
            torch.from_numpy(img_hq[..., ::-1].copy())
            .float()
            .transpose(2, 1)
            .transpose(1, 0)
        )
        img_lq = (
            torch.from_numpy(img_lq[..., ::-1].copy())
            .float()
            .transpose(2, 1)
            .transpose(1, 0)
        ) if lq_path is not None else None
        hq_latent = torch.from_numpy(hq_latent).float().transpose(2, 1).transpose(1, 0) if hq_latent is not None else None

        res_dict = {"gt": 2 * img_hq - 1, "txt": hq_text, "gt_path": hq_path}
        if hq_latent is not None:
            res_dict["hq_latent"] = hq_latent
        if lq_path is not None:
            res_dict["lq"] = 2 * img_lq - 1

        return res_dict

    def __len__(self) -> int:
        if self.data_len < 0:
            return len(self.hq_paths)
        else:
            return min(len(self.hq_paths), self.data_len)
