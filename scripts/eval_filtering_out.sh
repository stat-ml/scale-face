python3 ./face_lib/utils/reject_verification.py \
  --checkpoint_path=/gpfs/gpfs0/r.karimov/models/pfe/first_ms1m_pfe/sota.pt \
  --dataset_path=/gpfs/gpfs0/r.karimov/IJB/aligned_data_for_fusion/debug \
  --meta_path=/gpfs/gpfs0/r.karimov/IJB/aligned_data_for_fusion/metadata_refuse_verification/pairs_100_prob_0.5.csv \
  --config_path=./configs/models/iresnet_ms1m_pfe.yaml \
  --batch_size=64 \
  --rejected_portions 0.0001 0.001 0.01 0.1 0.2 0.5 \
  --device_id=0
