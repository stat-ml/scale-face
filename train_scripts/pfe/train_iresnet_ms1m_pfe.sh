python3 -m torch.distributed.launch --nproc_per_node 1 trainers/train_pfe.py \
--model-config configs/models/iresnet_ms1m_pfe.yaml --tmp