import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from pathlib import Path
import torch
from utils import util_image
import tqdm
import torch.nn.functional as F
import pyiqa

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def evaluate(in_path, ref_path, out_path, ntest):
    metric_paired_dict = {
        "psnr": pyiqa.create_metric(
            "psnr", test_y_channel=True, color_space="ycbcr"
        ).to(device),
        "ssim": pyiqa.create_metric(
            "ssim", test_y_channel=True, color_space="ycbcr"
        ).to(device),
    }

    in_path = Path(in_path)
    ref_path = Path(ref_path)
    lr_path_list = sorted([x for x in in_path.glob("*.[jpJP][pnPN]*[gG]")])
    ref_path_list = sorted([x for x in ref_path.glob("*.[jpJP][pnPN]*[gG]")])

    if ntest is not None:
        lr_path_list = lr_path_list[:ntest]
        ref_path_list = ref_path_list[:ntest]

    print(f"Find {len(lr_path_list)} images in {in_path}")
    result = {}

    for i in tqdm.tqdm(range(len(lr_path_list))):
        im_in = util_image.imread(lr_path_list[i], chn="rgb", dtype="float32")
        im_ref = util_image.imread(ref_path_list[i], chn="rgb", dtype="float32")

        im_in_tensor = util_image.img2tensor(im_in).to(device)
        im_ref_tensor = util_image.img2tensor(im_ref).to(device)

        if im_in_tensor.shape[-2:] != im_ref_tensor.shape[-2:]:
            im_in_tensor = F.interpolate(
                im_in_tensor,
                size=im_ref_tensor.shape[-2:],
                mode="bicubic",
                align_corners=False,
            )

        for key, metric in metric_paired_dict.items():
            with torch.cuda.amp.autocast():
                result[key] = (
                    result.get(key, 0) + metric(im_in_tensor, im_ref_tensor).item()
                )

    print_results = []
    for key, res in result.items():
        print(f"{key}: {res / len(lr_path_list):.5f}")
        print_results.append(f"{key}: {res / len(lr_path_list):.5f}")

    os.makedirs(out_path, exist_ok=True)
    out_t = os.path.join(out_path, "results.txt")
    with open(out_t, "w", encoding="utf-8") as f:
        for item in print_results:
            f.write(f"{item}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--in_path",
        type=str,
        default="/mnt/massive/wangce/Rescaling/diffbir/experiments-abla/jpeg-2",
    )
    parser.add_argument(
        "-r", "--ref_path", type=str, default="/mnt/massive/wangce/dataset/DIV2K/val/HR"
    )
    parser.add_argument("-o", "--out_path", type=str, default="logs/jpeg-2")
    parser.add_argument("--ntest", type=int, default=None)
    args = parser.parse_args()
    evaluate(args.in_path, args.ref_path, args.out_path, args.ntest)
