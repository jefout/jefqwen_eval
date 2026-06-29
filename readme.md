# Evaluation code I use
python local_eval_fast.py \
  --model \
  --eval-config eval/local_eval_datasets.json \
  --n-sequences 32 --seq-len 2048 --batch-size 8 --gpus auto


# current training pipeline
## 1. Download the training datasets with data_sampler.py
## 2. training with axolotl with the config in axolotl_config_v0