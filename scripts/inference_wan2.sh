export PYTHONPATH=$(pwd)


# three2one, synthetic scenarios
# CUDA_VISIBLE_DEVICES=1 python src/wan2_inference.py --config=configs/wan2-2_lora_three2one_synthetic.yaml \
#                         --ckpt_path="ckpts/wan2-2_lora_three2one_synthetic.ckpt" \
#                         --original_video_root="examples/synthetic_scenarios/Third_Video" \
#                         --pred_path="pred_results/three2one_synthetic" \
#                         --seed=1234


# one2three, synthetic scenarios
# CUDA_VISIBLE_DEVICES=1 python src/wan2_inference.py --config=configs/wan2-2_lora_one2three_synthetic.yaml \
#                         --ckpt_path "ckpts/wan2-2_lora_one2three_synthetic.ckpt" \
#                         --original_video_root="examples/synthetic_scenarios/First_Video" \
#                         --ref_image_root="examples/synthetic_scenarios/Reference_Image" \
#                         --pred_path "pred_results/one2three_synthetic" \
#                         --seed=1234


# three2one, real-world scenarios
# CUDA_VISIBLE_DEVICES=1 python src/wan2_inference.py --config=configs/wan2-2_lora_three2one_realworld.yaml \
#                         --ckpt_path="ckpts/wan2-2_lora_three2one_realworld.ckpt" \
#                         --original_video_root="examples/realworld_scenarios/Third_Video" \
#                         --pred_path="pred_results/three2one_realworld" \
#                         --seed=1234


# one2three, real-world scenarios
CUDA_VISIBLE_DEVICES=6,7 python src/wan2_inference.py --config=configs/wan2-2_lora_one2three_realworld.yaml \
                        --ckpt_path "ckpts/wan2-2_lora_one2three_realworld.ckpt" \
                        --original_video_root="examples/realworld_scenarios/First_Video" \
                        --ref_image_root="examples/realworld_scenarios/Reference_Image" \
                        --pred_path "pred_results/one2three_realworld" \
                        --seed=1234
