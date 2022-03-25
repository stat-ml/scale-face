## Scale
#python3 ./face_lib/evaluation/template_reject_verification.py \
#  --checkpoint_path=/gpfs/gpfs0/k.fedyanin/space/models/scale/01_frozen/01_sigm_mul/checkpoint.pth \
#  --dataset_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/small \
#  --protocol=ijbc \
#  --protocol_path=/gpfs/gpfs0/k.fedyanin/space/IJB/IJB-C/protocols/archive \
#  --config_path=./configs/scale/sigm_mul.yaml \
#  --batch_size=64 \
#  --uncertainty_strategy=scale \
#  --uncertainty_mode=confidence \
#  --FARs 0.0001 0.0005 0.001 0.005 0.01 0.05 \
#  --rejected_portions $(seq 0 0.002 0.5) \
#  --fusion_distance_uncertainty_metrics mean_cosine_mean mean_cosine_harmonic-harmonic mean_cosine_mul \
#  --device_id=0 \
#  --save_fig_path=/beegfs/home/r.kail/faces/figures/test

## ScaleFace
#python3 ./face_lib/evaluation/template_reject_verification.py \
#  --checkpoint_path=/gpfs/data/gpfs0/k.fedyanin/space/models/scale/02_sigm_mul_selection/32/checkpoint.pth \
#  --dataset_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/big \
#  --protocol=ijbc \
#  --protocol_path=/gpfs/gpfs0/k.fedyanin/space/IJB/IJB-C/protocols/test1 \
#  --config_path=./configs/scale/01_activation_selection/sigm_mul.yaml \
#  --batch_size=64 \
#  --distaces_batch_size=8 \
#  --uncertainty_strategy=scale \
#  --uncertainty_mode=confidence \
#  --FARs 0.0001 0.001 0.05\
#  --rejected_portions $(seq 0 0.02 0.5) \
#  --fusion_distance_uncertainty_metrics first_cosine_mean mean_cosine_mean \
#  --device_id=0 \
#  --save_fig_path=/gpfs/gpfs0/k.fedyanin/space/figures/test \
#  --verbose \
#  --cached_embeddings

### alternative options for scale
#  --fusion_distance_uncertainty_metrics first_cosine_mean mean_cosine_mean mean_cosine_harmonic-harmonic mean_cosine_mul\
#  --checkpoint_path=/gpfs/gpfs0/k.fedyanin/space/models/scale/01_frozen/01_sigm_mul/checkpoint.pth \
#  --save_fig_path=/beegfs/home/r.kail/faces/figures/test
#  --protocol_path=/gpfs/gpfs0/k.fedyanin/space/IJB/IJB-C/protocols/archive \
#  --FARs 0.0001 0.0005 0.001 0.005 0.01 0.05 \
#  --dataset_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/small \
#  --dataset_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_images_identity \
#  --fusion_distance_uncertainty_metrics PFE_cosine_mean mean_cosine_mean mean_cosine_harmonic-harmonic mean_cosine_mul \

####  PFE normalized
#python3 ./face_lib/evaluation/template_reject_verification.py \
#--checkpoint_path=/gpfs/data/gpfs0/k.fedyanin/space/models/pfe/classic_normalized_pfe/sota.pth \
#--dataset_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/big \
#--protocol=ijbc \
#--protocol_path=/gpfs/gpfs0/k.fedyanin/space/IJB/IJB-C/protocols/test1 \
#--config_path=./configs/models/iresnet_ms1m_pfe_normalized.yaml \
#--batch_size=64 \
#--distaces_batch_size=8 \
#--uncertainty_strategy=head \
#--uncertainty_mode=uncertainty \
#--FARs 0.0001 0.001 0.05 \
#--rejected_portions $(seq 0 0.02 0.5) \
#--fusion_distance_uncertainty_metrics first_cosine_mean mean_cosine_mean PFE_cosine_mean PFE_MLS_harmonic-sum PFE_MLS_harmonic-harmonic PFE_MLS_mean \
#--device_id=0 \
#--save_fig_path=/gpfs/gpfs0/k.fedyanin/space/figures/test \
#--verbose \
#--cached_embeddings

#### MagFace
#python3 ./face_lib/evaluation/template_reject_verification.py \
#  --checkpoint_path=/gpfs/gpfs0/k.fedyanin/space/models/scale/01_frozen/01_sigm_mul/checkpoint.pth \
#  --dataset_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/big \
#  --protocol=ijbc \
#  --protocol_path=/gpfs/gpfs0/k.fedyanin/space/IJB/IJB-C/protocols/test1 \
#  --config_path=./configs/scale/01_activation_selection/sigm_mul.yaml \
#  --batch_size=64 \
#  --distaces_batch_size=8 \
#  --uncertainty_strategy=magface \
#  --uncertainty_mode=confidence \
#  --FARs 0.0001 0.001 0.05\
#  --rejected_portions $(seq 0 0.02 0.5) \
#  --fusion_distance_uncertainty_metrics first_cosine_mean mean_cosine_mean \
#  --device_id=0 \
#  --save_fig_path=/gpfs/gpfs0/k.fedyanin/space/figures/test \
#  --verbose \
#  --cached_embeddings
#
#python3 explore/visualize_templates.py --last_timestamp --test_folder=/gpfs/gpfs0/k.fedyanin/space/figures/test


## ScaleFace fine-tuned
python3 ./face_lib/evaluation/template_reject_verification.py \
  --checkpoint_path=/gpfs/data/gpfs0/k.fedyanin/space/models/scale/06_fine_tuning/sigm_32_lr_0.0003/checkpoint.pth \
  --dataset_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/big \
  --protocol=ijbc \
  --protocol_path=/gpfs/gpfs0/k.fedyanin/space/IJB/IJB-C/protocols/test1 \
  --config_path=./configs/scale/02_sigm_mul_coef_selection/32.yaml \
  --batch_size=64 \
  --distaces_batch_size=8 \
  --uncertainty_strategy=scale_finetuned \
  --uncertainty_mode=confidence \
  --FARs 0.0001 0.001 0.05\
  --rejected_portions $(seq 0 0.02 0.5) \
  --fusion_distance_uncertainty_metrics first_cosine_mean mean_cosine_mean softmax_cosine_mean \
  --device_id=0 \
  --save_fig_path=/gpfs/gpfs0/k.fedyanin/space/figures/test \
  --verbose #\
#  --cached_embeddings
#  --config_path=./configs/scale/01_activation_selection/sigm_mul.yaml \

## ScaleFace fusions
#python3 ./face_lib/evaluation/template_reject_verification.py \
#  --checkpoint_path=/gpfs/data/gpfs0/k.fedyanin/space/models/scale/02_sigm_mul_selection/32/checkpoint.pth \
#  --dataset_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/big \
#  --protocol=ijbc \
#  --protocol_path=/gpfs/gpfs0/k.fedyanin/space/IJB/IJB-C/protocols/test1 \
#  --config_path=./configs/scale/01_activation_selection/sigm_mul.yaml \
#  --batch_size=64 \
#  --distaces_batch_size=8 \
#  --uncertainty_strategy=scale \
#  --uncertainty_mode=confidence \
#  --FARs 0.0001 0.001 0.05\
#  --rejected_portions $(seq 0 0.02 0.5) \
#  --fusion_distance_uncertainty_metrics softmax_cosine_harmonic-harmonic stat-softmax_cosine_harmonic-harmonic softmax_cosine_harmonic-harmonic stat-softmax_cosine_harmonic-harmonic harmonic-harmonic_cosine_harmonic-harmonic stat-harmonic-harmonic_cosine_harmonic-harmonic \
#  --device_id=0 \
#  --save_fig_path=/gpfs/gpfs0/k.fedyanin/space/figures/test \
#  --verbose \
#  --cached_embeddings

#  --fusion_distance_uncertainty_metrics argmax_cosine_mean first_cosine_mean mean_cosine_mean softmax_cosine_mean \
#  --fusion_distance_uncertainty_metrics first_cosine_mean mean_cosine_mean softmax_cosine_mean argmax_cosine_mean PFE_cosine_mean \
