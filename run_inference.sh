accelerate launch --num_processes=1 --gpu_ids="0" --main_process_port 29300 src/inference_tadm.py \
    --test_config "./configs/tadm_test.yaml" \
    --enable_xformers_memory_efficient_attention \
    --scale_factor 16 \
    --dfrm_net_path "path/to/pretrained/dfrm" \
    --pretrained_path "path/to/pretrained/tadm" \
    --output_dir "./path/to/dfrm/results" \
    --latent_tiled_size 96 \
    --latent_tiled_overlap 32

    # x16: /mnt/massive/wangce/ArbiRescale/S3Diff/assets/mm-realsr/model_sr_x16.pkl
    # x32: /mnt/massive/wangce/ArbiRescale/S3Diff/assets/mm-realsr/model_sr_x32.pkl