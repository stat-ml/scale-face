## PFE
#python3 ./face_lib/utils/reject_verification.py \
#  --checkpoint_path=/gpfs/gpfs0/k.fedyanin/space/models/pfe/first_ms1m_pfe/sota.pt \
#  --dataset_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/big \
#  --pairs_table_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/metadata_refuse_verification/pairs_1000000_prob_0.5.csv \
#  --config_path=./configs/models/iresnet_ms1m_pfe.yaml \
#  --batch_size=64 \
#  --uncertainty_strategy=head \
#  --FARs 0.0001 0.0005 0.001 0.005 0.01 0.05 \
#  --rejected_portions $(seq 0 0.002 0.5) \
#  --distance_uncertainty_metrics cosine_mean cosine_harmonic-sum cosine_harmonic-harmonic MLS_harmonic-sum MLS_harmonic-harmonic \
#  --device_id=0 \
#  --save_fig_path=/gpfs/gpfs0/r.kail/figures/rejected

## Probface
#python3 ./face_lib/utils/reject_verification.py \
#  --checkpoint_path=/gpfs/gpfs0/r.zainulin/sota.pth \
#  --dataset_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/big \
#  --pairs_table_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/metadata_refuse_verification/pairs_1000000_prob_0.5.csv \
#  --config_path=./configs/models/iresnet_ms1m_probface.yaml \
#  --batch_size=64 \
#  --uncertainty_strategy=head \
#  --FARs 0.0001 0.0005 0.001 0.005 0.01 0.05 \
#  --rejected_portions $(seq 0 0.002 0.5) \
#  --distance_uncertainty_metrics cosine_mean cosine_harmonic-sum cosine_harmonic-harmonic MLS_harmonic-sum MLS_harmonic-harmonic \
#  --device_id=0 \
#  --save_fig_path=/gpfs/gpfs0/r.kail/figures/rejected

## PFE normalized
#python3 ./face_lib/utils/reject_verification.py \
#  --checkpoint_path=/gpfs/data/gpfs0/k.fedyanin/space/models/pfe/normalized_pfe/sota.pth \
#  --dataset_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/big \
#  --pairs_table_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/metadata_refuse_verification/pairs_1000000_prob_0.5.csv \
#  --config_path=./configs/models/iresnet_ms1m_pfe_normalized.yaml \
#  --batch_size=64 \
#  --uncertainty_strategy=head \
#  --FARs 0.0001 0.0005 0.001 0.005 0.01 0.05 \
#  --rejected_portions $(seq 0 0.002 0.5) \
#  --distance_uncertainty_metrics cosine_mean cosine_harmonic-sum cosine_harmonic-harmonic MLS_harmonic-sum MLS_harmonic-harmonic \
#  --device_id=0 \
#  --save_fig_path=/gpfs/data/gpfs0/r.kail/figures/normalized_pfe_3/

## GAN
#python3 ./face_lib/utils/reject_verification.py \
#  --checkpoint_path=/gpfs/gpfs0/k.fedyanin/space/models/pfe/normalized_pfe/sota.pth \
#  --dataset_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/big \
#  --pairs_table_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/metadata_refuse_verification/pairs_1000000_prob_0.5.csv \
#  --config_path=./configs/models/iresnet_ms1m_pfe_normalized.yaml \
#  --batch_size=4 \
#  --uncertainty_strategy=GAN \
#  --FARs 0.0001 0.0005 0.001 0.005 0.01 0.05 \
#  --rejected_portions $(seq 0 0.002 0.5) \
#  --distance_uncertainty_metrics cosine_mean cosine_harmonic-harmonic\
#  --device_id=0 \
#  --save_fig_path=/gpfs/data/gpfs0/r.kail/figures/gan_bs4/ \
#  --discriminator_path=/gpfs/data/gpfs0/k.fedyanin/space/GAN/stylegan.pth

## Classifier
#python3 ./face_lib/utils/reject_verification.py \
#  --checkpoint_path=/gpfs/data/gpfs0/k.fedyanin/space/models/pair_classifiers/01_smart_cos \
#  --dataset_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/big \
#  --pairs_table_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/metadata_refuse_verification/pairs_1000000_prob_0.5.csv \
#  --config_path=./configs/models/pair_classifiers/smart_cosine.yaml \
#  --batch_size=64 \
#  --distaces_batch_size=100 \
#  --uncertainty_strategy=classifier \
#  --FARs 0.0001 0.0005 0.001 0.005 0.01 0.05 \
#  --rejected_portions $(seq 0 0.002 0.5) \
#  --distance_uncertainty_metrics classifier_classifier cosine_classifier MLS_classifier cosine_harmonic-sum \
#  --device_id=0 \
#  --save_fig_path=/gpfs/data/gpfs0/r.kail/figures/test
##  --save_fig_path=/gpfs/data/gpfs0/r.kail/figures/classifiers/classifier_smart_cosine/

## Classifier
#python3 ./face_lib/utils/reject_verification.py \
#  --checkpoint_path=/gpfs/data/gpfs0/k.fedyanin/space/models/pair_classifiers/02_bilinear/checkpoints/sota.pth \
#  --dataset_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/big \
#  --pairs_table_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/metadata_refuse_verification/pairs_1000000_prob_0.5.csv \
#  --config_path=./configs/models/pair_classifiers/bilinear.yaml \
#  --batch_size=64 \
#  --distaces_batch_size=100 \
#  --uncertainty_strategy=classifier \
#  --FARs 0.0001 0.0005 0.001 0.005 0.01 0.05 \
#  --rejected_portions $(seq 0 0.002 0.5) \
#  --distance_uncertainty_metrics classifier_classifier cosine_classifier MLS_classifier classifier_harmonic-sum cosine_harmonic-sum MLS_harmonic-sum \
#  --device_id=0 \
#  --save_fig_path=/gpfs/data/gpfs0/r.kail/figures/pair_classifiers/02_bilinear

### PFE spectral
#python3 ./face_lib/utils/reject_verification.py \
#  --checkpoint_path=/gpfs/data/gpfs0/k.fedyanin/space/models/pfe/spectral_normalized_pfe/sota.pth \
#  --dataset_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/big \
#  --pairs_table_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/metadata_refuse_verification/pairs_1000000_prob_0.5.csv \
#  --config_path=./configs/models/iresnet_ms1m_pfe_normalized_spectral.yaml \
#  --batch_size=64 \
#  --uncertainty_strategy=head \
#  --FARs 0.0001 0.0005 0.001 0.005 0.01 0.05 \
#  --rejected_portions $(seq 0 0.002 0.5) \
#  --distance_uncertainty_metrics cosine_mean cosine_harmonic-sum cosine_harmonic-harmonic MLS_harmonic-sum MLS_harmonic-harmonic \
#  --device_id=0 \
#  --save_fig_path=/gpfs/data/gpfs0/r.kail/figures/spectral/

### Scale
python3 ./face_lib/utils/reject_verification.py \
  --checkpoint_path=/gpfs/gpfs0/k.fedyanin/space/models/scale/01_frozen/01_sigm_mul/checkpoint.pth \
  --dataset_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/big \
  --pairs_table_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/metadata_refuse_verification/pairs_1000_prob_0.5.csv \
  --config_path=./configs/models/scale/sigm_mul.yaml \
  --batch_size=64 \
  --uncertainty_strategy=scale \
  --uncertainty_mode=confidence \
  --FARs 0.0001 0.0005 0.001 0.005 0.01 0.05 \
  --rejected_portions $(seq 0 0.002 0.5) \
  --distance_uncertainty_metrics cosine_mean cosine_harmonic-harmonic cosine_mul \
  --device_id=0 \
  --save_fig_path=/beegfs/home/r.kail/faces/figures/test
