CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch \
  --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1234 \
    trainers/train_scale.py \
  --model-config ./configs/scale/fane_tune.yaml
