## PFE
#python3 ./face_lib/utils/reject_verification.py \
#  --checkpoint_path=/gpfs/gpfs0/k.fedyanin/space/models/pfe/first_ms1m_pfe/sota.pt \
#  --dataset_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/big \
#  --pairs_table_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/metadata_refuse_verification/pairs_1000000_prob_0.5.csv \
#  --config_path=./configs/models/iresnet_ms1m_pfe.yaml \
#  --batch_size=64 \
#  --uncertainty_strategy=head \
#  --FARs 0.0001 0.0005 0.001 0.005 0.01 0.05 \
#  --rejected_portions $(seq 0 0.002 0.2) \
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
#  --rejected_portions $(seq 0 0.002 0.2) \
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
#  --rejected_portions $(seq 0 0.002 0.2) \
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
#  --rejected_portions $(seq 0 0.002 0.2) \
#  --distance_uncertainty_metrics cosine_mean cosine_harmonic-harmonic\
#  --device_id=0 \
#  --save_fig_path=/gpfs/data/gpfs0/r.kail/figures/gan_bs4/ \
#  --discriminator_path=/gpfs/data/gpfs0/k.fedyanin/space/GAN/stylegan.pth

## Classifier
#python3 ./face_lib/utils/reject_verification.py \
#  --checkpoint_path=/gpfs/data/gpfs0/k.fedyanin/space/models/pair_classifiers/first_pair_classifier_advanced.pth \
#  --dataset_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/big \
#  --pairs_table_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/metadata_refuse_verification/pairs_1000_prob_0.5.csv \
#  --config_path=./configs/models/iresnet_ms1m_face_classification.yaml \
#  --batch_size=64 \
#  --distaces_batch_size=100 \
#  --uncertainty_strategy=classifier \
#  --FARs 0.0001 0.0005 0.001 0.005 0.01 0.05 \
#  --rejected_portions $(seq 0 0.002 0.2) \
#  --distance_uncertainty_metrics classifier_classifier cosine_classifier MLS_classifier cosine_harmonic-sum MLS_harmonic-sum \
#  --device_id=0 \
#  --save_fig_path=/gpfs/data/gpfs0/r.kail/figures/classifiers/classifier_1/

### PFE spectral
#python3 ./face_lib/utils/reject_verification.py \
#  --checkpoint_path=/gpfs/data/gpfs0/k.fedyanin/space/models/pfe/spectral_normalized_pfe/sota.pth \
#  --dataset_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/big \
#  --pairs_table_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/metadata_refuse_verification/pairs_1000000_prob_0.5.csv \
#  --config_path=./configs/models/iresnet_ms1m_pfe_normalized_spectral.yaml \
#  --batch_size=64 \
#  --uncertainty_strategy=head \
#  --FARs 0.0001 0.0005 0.001 0.005 0.01 0.05 \
#  --rejected_portions $(seq 0 0.002 0.2) \
#  --distance_uncertainty_metrics cosine_mean cosine_harmonic-sum cosine_harmonic-harmonic MLS_harmonic-sum MLS_harmonic-harmonic \
#  --device_id=0 \
#  --save_fig_path=/gpfs/data/gpfs0/r.kail/figures/spectral/
