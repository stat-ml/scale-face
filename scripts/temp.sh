
python3 ./face_lib/evaluation/template_reject_verification.py \
  --checkpoint_path=/gpfs/data/gpfs0/k.fedyanin/space/models/scale/02_sigm_mul_selection/32/checkpoint.pth \
  --dataset_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/big \
  --protocol=ijbc \
  --protocol_path=/gpfs/gpfs0/k.fedyanin/space/IJB/IJB-C/protocols/test1 \
  --config_path=./configs/scale/01_activation_selection/sigm_mul.yaml \
  --batch_size=64 \
  --distaces_batch_size=8 \
  --uncertainty_strategy=scale \
  --uncertainty_mode=confidence \
  --FARs 0.0001 0.001 0.05 \
  --fusion_distance_uncertainty_metrics mean_cosine_mean mean_centered-cosine_mean \
  --rejected_portions $(seq 0 0.02 0.5) \
  --device_id=0 \
  --save_fig_path=/gpfs/gpfs0/k.fedyanin/space/figures/test \
  --verbose \
  --cached_embeddings \
  --equal_uncertainty_enroll
