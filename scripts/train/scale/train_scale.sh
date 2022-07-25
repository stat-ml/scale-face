#python3 trainers/train_scale.py \
#  --model-config ./configs/scale/01_activation_selection/test.yaml \
#  --debug --tmp

#CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch \
#  --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1234 \
#    trainers/train_scale.py \
#  --model-config ./configs/scale/01_activation_selection/test.yaml \
#  --debug --tmp

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch \
  --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1234 \
    trainers/train_scale.py \
  --model-config ./configs/scale/01_activation_selection/test.yaml \
  --debug --tmp