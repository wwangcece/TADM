import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import pyiqa
import argparse
from pathlib import Path
import torch
from utils import util_image
import tqdm
import torch.nn.functional as F

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# print(pyiqa.list_models())


def evaluate(in_path, ref_path, out_path, ntest):
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
        im_in_tensor = util_image.img2tensor(im_in).to(device)  # 1 x c x h x w
        for key, metric in metric_dict.items():
            with torch.cuda.amp.autocast():
                result[key] = result.get(key, 0) + metric(im_in_tensor).item()

        if ref_path is not None:
            im_ref = util_image.imread(
                _ref_path, chn="rgb", dtype="float32"
            )  # h x w x c
            im_ref_tensor = util_image.img2tensor(im_ref).to(device)
            in_height = im_in_tensor.shape[-2]
            in_width = im_in_tensor.shape[-1]
            ref_height = im_ref_tensor.shape[-2]
            ref_width = im_ref_tensor.shape[-1]
            if in_height != ref_height or in_width != ref_width:
                im_in_tensor = F.interpolate(
                    im_in_tensor,
                    size=(im_ref_tensor.shape[-2], im_ref_tensor.shape[-1]),
                    mode="bicubic",
                    align_corners=False,
                )
            for key, metric in metric_paired_dict.items():
                result[key] = (
                    result.get(key, 0) + metric(im_in_tensor, im_ref_tensor).item()
                )

    if ref_path is not None:
        fid_metric = pyiqa.create_metric("fid")
        result["fid"] = fid_metric(in_path, ref_path)

    print_results = []
    for key, res in result.items():
        if key == "fid":
            print(f"{key}: {res:.2f}")
            print_results.append(f"{key}: {res:.2f}")
        else:
            print(f"{key}: {res/len(lr_path_list):.5f}")
            print_results.append(f"{key}: {res/len(lr_path_list):.5f}")

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
        default="/mnt/massive/wangce/NTIRE-2024/StableSR-main/results/rescale_x32_512/Urban100",
    )
    parser.add_argument(
        "-r",
        "--ref_path",
        type=str,
        default="/mnt/massive/wangce/dataset/Urban100/HR",
    )
    parser.add_argument(
        "-o",
        "--out_path",
        type=str,
        default="./logs/x32_Urban100",
    )
    parser.add_argument("--ntest", type=int, default=None)
    args = parser.parse_args()
    evaluate(args.in_path, args.ref_path, args.out_path, args.ntest)
