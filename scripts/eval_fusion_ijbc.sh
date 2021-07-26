python3 ./face_lib/utils/fusion.py \
  --checkpoint_path=/gpfs/gpfs0/r.karimov/models/pfe/first_ms1m_pfe/sota.pt \
  --dataset_path=/gpfs/gpfs0/r.kail/IJB/IJBC_aligned_good_big \
  --protocol_path=/gpfs/gpfs0/r.kail/IJB/raw_data/IJB/IJB-C/protocols/archive \
  --config_path=/trinity/home/r.kail/Faces/face-evaluation/configs/models/iresnet_ms1m_pfe.yaml \
  --protocol=ijbc

#python3 ./face_lib/utils/fusion.py \
#  --checkpoint_path=/gpfs/gpfs0/r.karimov/models/pfe/first_ms1m_pfe/sota.pt \
#  --dataset_path=/gpfs/gpfs0/r.kail/IJB/IJBC_aligned_debug \
#  --protocol_path=/gpfs/gpfs0/r.kail/IJB/raw_data/IJB/IJB-C/protocols/archive \
#  --config_path=/trinity/home/r.kail/Faces/face-evaluation/configs/models/iresnet_ms1m_pfe.yaml \
#  --protocol=ijbc

 # Evaluating probface on fusion IJBC
#python3 ./face_lib/utils/fusion.py \
#  --checkpoint_path=/gpfs/gpfs0/r.zainulin/sota.pth \
#  --dataset_path=/gpfs/gpfs0/r.kail/IJB/IJBC_aligned_big \
#  --protocol_path=/gpfs/gpfs0/r.kail/IJB/raw_data/IJB/IJB-C/protocols/archive \
#  --config_path=/trinity/home/r.kail/Faces/face-evaluation/configs/models/iresnet_ms1m_probface.yaml \
#  --protocol=ijbc
