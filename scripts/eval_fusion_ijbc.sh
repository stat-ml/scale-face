## Fusion with PFE
#python3 ./face_lib/utils/fusion.py \
#  --checkpoint_path=/gpfs/gpfs0/r.karimov/models/pfe/first_ms1m_pfe/sota.pt \
#  --dataset_path=/gpfs/gpfs0/r.karimov/IJB/aligned_data_for_fusion/big \
#  --protocol_path=/gpfs/gpfs0/r.karimov/IJB/IJB-C/protocols/archive \
#  --config_path=./configs/models/iresnet_ms1m_pfe.yaml \
#  --protocol=ijbc

# Fusion with ProbFace
#python3 ./face_lib/utils/fusion.py \
#  --checkpoint_path=/gpfs/gpfs0/r.zainulin/sota.pth \
#  --dataset_path=/gpfs/gpfs0/r.karimov/IJB/aligned_data_for_fusion/big \
#  --protocol_path=/gpfs/gpfs0/r.karimov/IJB/IJB-C/protocols/archive \
#  --config_path=./configs/models/iresnet_ms1m_probface.yaml \
#  --protocol=ijbc

## Evaluating probface on fusion IJBC
#python3 ./face_lib/utils/fusion.py \
#  --checkpoint_path=/gpfs/gpfs0/r.zainulin/sota.pth \
#  --dataset_path=/gpfs/gpfs0/r.karimov/IJB/aligned_data_for_fusion/debug \
#  --protocol_path=/gpfs/gpfs0/r.karimov/IJB/IJB-C/protocols/archive \
#  --config_path=/gpfs/gpfs0/r.zainulin/git_repos/face-evaluation/configs/models/iresnet_ms1m_probface.yaml \
#  --protocol=ijbc

# Fusion with PFE
python3 ./face_lib/utils/fusion.py \
  --checkpoint_path=/trinity/home/r.kail/Faces/face-evaluation/exman/runs/000086-2021-08-16-15-39-24/checkpoints/sota.pth \
  --dataset_path=/gpfs/gpfs0/r.karimov/IJB/aligned_data_for_fusion/big \
  --protocol_path=/gpfs/gpfs0/r.karimov/IJB/IJB-C/protocols/archive \
  --config_path=./configs/models/iresnet_ms1m_pfe_normalized.yaml \
  --protocol=ijbc