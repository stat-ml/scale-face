## Fusion with PFE
#python3 ./face_lib/utils/fusion.py \
#  --checkpoint_path=/gpfs/gpfs0/r.karimov/models/pfe/first_ms1m_pfe/sota.pt \
#  --dataset_path=/gpfs/gpfs0/r.karimov/IJB/aligned_data_for_fusion/big \
#  --protocol_path=/gpfs/gpfs0/r.karimov/IJB/IJB-C/protocols/archive \
#  --config_path=./configs/models/iresnet_ms1m_pfe.yaml \
#  --protocol=ijbc

# Fusion with ProbFace
python3 ./face_lib/utils/fusion.py \
  --checkpoint_path=/gpfs/gpfs0/r.zainulin/sota.pth \
  --dataset_path=/gpfs/gpfs0/r.karimov/IJB/aligned_data_for_fusion/big \
  --protocol_path=/gpfs/gpfs0/r.karimov/IJB/IJB-C/protocols/archive \
  --config_path=./configs/models/iresnet_ms1m_probface.yaml \
  --protocol=ijbc