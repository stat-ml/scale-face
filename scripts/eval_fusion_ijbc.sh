# Fusion with PFE
python3 ./face_lib/utils/fusion.py \
  --checkpoint_path=/gpfs/gpfs0/k.fedyanin/space/models/pfe/first_ms1m_pfe/sota.pt \
  --dataset_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/big \
  --protocol_path=/gpfs/gpfs0/k.fedyanin/space/IJB/IJB-C/protocols/archive \
  --config_path=./configs/models/iresnet_ms1m_pfe.yaml \
  --uncertainty_strategy=head \
  --protocol=ijbc \
  --fusion_distance_methods random_cosine mean_cosine min_cosine softmax_cosine PFE_cosine min_MLS PFE_MLS \
  --FARs 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 \
  --device_id=0 \
  --batch_size=64 \
  --save_table_path=/gpfs/gpfs0/r.kail/tables/res_PFE.pkl \
  --verbose

## Fusion with Probface
#python3 ./face_lib/utils/fusion.py \
#  --checkpoint_path=/gpfs/gpfs0/r.zainulin/sota.pth \
#  --dataset_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/big \
#  --protocol_path=/gpfs/gpfs0/k.fedyanin/space/IJB/IJB-C/protocols/archive \
#  --config_path=./configs/models/iresnet_ms1m_probface.yaml \
#  --uncertainty_strategy=head \
#  --protocol=ijbc \
#  --fusion_distance_methods random_cosine mean_cosine PFE_cosine \
#  --FARs 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 \
#  --device_id=0 \
#  --batch_size=64 \
#  --save_table_path=/gpfs/gpfs0/r.kail/tables/res_probface_good.pkl \
#  --verbose

## Fusion with PFE
#python3 ./face_lib/utils/fusion.py \
#  --checkpoint_path=/gpfs/data/gpfs0/r.karimov/models/pfe/normalized_pfe/sota.pth \
#  --dataset_path=/gpfs/gpfs0/r.karimov/IJB/aligned_data_for_fusion/big \
#  --protocol_path=/gpfs/gpfs0/r.karimov/IJB/IJB-C/protocols/archive \
#  --config_path=./configs/models/iresnet_ms1m_pfe_normalized.yaml \
#  --uncertainty_strategy=head \
#  --protocol=ijbc \
#  --fusion_distance_methods random_cosine mean_cosine min_cosine softmax_cosine PFE_cosine min_MLS PFE_MLS \
#  --FARs 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 \
#  --device_id=0 \
#  --batch_size=64 \
#  --save_table_path=/gpfs/gpfs0/r.kail/tables/res_PFE_normalized.pkl \
#  --verbose
