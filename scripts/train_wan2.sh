export PYTHONPATH=$(pwd)


# three2one, synthetic scenarios
# CUDA_VISIBLE_DEVICES=0,1,2,3 python src/wan2_trainer.py --config=configs/wan2-2_lora_three2one_synthetic.yaml --seed=1234


# one2three, synthetic scenarios
# CUDA_VISIBLE_DEVICES=0,1,2,3 python src/wan2_trainer.py --config=configs/wan2-2_lora_one2three_synthetic.yaml --seed=1234


# three2one, real-world scenarios
# CUDA_VISIBLE_DEVICES=0,1,2,3 python src/wan2_trainer.py --config=configs/wan2-2_lora_three2one_realworld.yaml --seed=1234


# one2three, real-world scenarios
# CUDA_VISIBLE_DEVICES=0,1,2,3 python src/wan2_trainer.py --config=configs/wan2-2_lora_one2three_realworld.yaml --seed=1234
CUDA_VISIBLE_DEVICES=4,5 python src/wan2_trainer.py --config=configs/my_train.yaml --seed=1234 --resume_path=outputs/wan2-2_one2three_realworld/rank-80_stride-2_v1/checkpoints/step=6800.ckpt