python3 ./face_lib/utils/fusion/fusion_perm.py \
  --checkpoint_path=/gpfs/gpfs0/r.karimov/models/pfe/first_ms1m_pfe/sota.pt \
  --dataset_path=/gpfs/gpfs0/r.kail/IJB/IJBC_cropped_good_big \
  --protocol_path=/gpfs/gpfs0/r.kail/IJB/raw_data/IJB/IJB-C/protocols/archive \
  --config_path=/trinity/home/r.kail/Faces/face-evaluation/configs/models/iresnet_ms1m_pfe.yaml \
  --protocol=ijbc

#  --dataset_path=/gpfs/gpfs0/r.kail/IJB/IJB_cropped_imgs_only_prev/ \