python3 ./face_lib/utils/reject_verification.py \
  --checkpoint_path=/gpfs/gpfs0/r.karimov/models/pfe/first_ms1m_pfe/sota.pt \
  --dataset_path=/gpfs/gpfs0/r.karimov/IJB/aligned_data_for_fusion/big \
  --pairs_table_path=/gpfs/gpfs0/r.karimov/IJB/aligned_data_for_fusion/metadata_refuse_verification/pairs_10000000_prob_0.5.csv \
  --config_path=./configs/models/iresnet_ms1m_pfe.yaml \
  --batch_size=64 \
  --FARs 0.0001 0.0005 0.001 0.005 0.01 0.05 \
  --rejected_portions 0.005 0.01 0.02 0.05 0.1 0.2 \
  --device_id=0 \
  --save_fig_path=/gpfs/gpfs0/r.kail/figures/rejected.png
