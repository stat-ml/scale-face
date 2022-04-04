## PFE
#python3 ./face_lib/evaluation/reject_verification.py \
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
#  --save_fig_path=/beegfs/home/r.kail/faces/figures/test

## Probface
#python3 ./face_lib/evaluation/reject_verification.py \
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
#  --save_fig_path=/beegfs/home/r.kail/faces/figures/test

### PFE normalized
#python3 ./face_lib/evaluation/reject_verification.py \
#  --checkpoint_path=/gpfs/data/gpfs0/k.fedyanin/space/models/pfe/normalized_pfe/sota.pth \
#  --dataset_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/big \
#  --pairs_table_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/metadata_refuse_verification/pairs_10000_prob_0.1.csv \
#  --config_path=./configs/models/iresnet_ms1m_pfe_normalized.yaml \
#  --batch_size=16 \
#  --uncertainty_strategy=head \
#  --FARs 0.0001 0.001 0.05 \
#  --rejected_portions $(seq 0 0.002 0.5) \
#  --distance_uncertainty_metrics cosine_mean cosine_harmonic-sum cosine_harmonic-harmonic MLS_harmonic-sum MLS_harmonic-harmonic \
#  --device_id=0 \
#  --save_fig_path=/gpfs/gpfs0/k.fedyanin/space/figures/test
##  --checkpoint_path=/gpfs/data/gpfs0/k.fedyanin/space/models/pfe/classic_normalized_pfe/sota.pth \

## GAN
#python3 ./face_lib/evaluation/reject_verification.py \
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
#  --discriminator_path=/gpfs/data/gpfs0/k.fedyanin/space/GAN/stylegan.pth
#  --save_fig_path=/beegfs/home/r.kail/faces/figures/test \

## Classifier
#python3 ./face_lib/evaluation/reject_verification.py \
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
#  --save_fig_path=/beegfs/home/r.kail/faces/figures/test

## Classifier
#python3 ./face_lib/evaluation/reject_verification.py \
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
#  --save_fig_path=/beegfs/home/r.kail/faces/figures/test

### PFE spectral
#python3 ./face_lib/evaluation/reject_verification.py \
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
#  --save_fig_path=/beegfs/home/r.kail/faces/figures/test

## Scale
#python3 ./face_lib/evaluation/reject_verification.py \
#  --checkpoint_path=/gpfs/gpfs0/k.fedyanin/space/models/scale/01_frozen/01_sigm_mul/checkpoint.pth \
#  --dataset_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/big \
#  --pairs_table_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/metadata_refuse_verification/pairs_1000_prob_0.5.csv \
#  --config_path=./configs/scale/sigm_mul.yaml \
#  --batch_size=64 \
#  --uncertainty_strategy=scale \
#  --uncertainty_mode=confidence \
#  --FARs 0.0001 0.0005 0.001 0.005 0.01 0.05 \
#  --rejected_portions $(seq 0 0.002 0.5) \
#  --distance_uncertainty_metrics cosine_mean cosine_harmonic-harmonic cosine_mul \
#  --device_id=0 \
#  --save_fig_path=/beegfs/home/r.kail/faces/figures/test

## L2 norm of vector
#python3 ./face_lib/evaluation/reject_verification.py \
#  --checkpoint_path=/gpfs/data/gpfs0/k.fedyanin/space/models/arcface/backbones/classic_packed.pth \
#  --dataset_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/big \
#  --pairs_table_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/metadata_refuse_verification/pairs_1000000_prob_0.5.csv \
#  --config_path=./configs/models/arcface_emb_norm.yaml \
#  --batch_size=32 \
#  --uncertainty_strategy=emb_norm \
#  --uncertainty_mode=confidence \
#  --FARs 0.0001 0.0005 0.001 0.005 0.01 0.05 \
#  --rejected_portions $(seq 0 0.002 0.5) \
#  --distance_uncertainty_metrics cosine_mean cosine_harmonic-harmonic cosine_mul cosine_squared-sum cosine_squared-harmonic \
#  --device_id=0 \
#  --save_fig_path=/beegfs/home/r.kail/faces/figures/test

## Arcface backbone + MagFace uncertainty
#python3 ./face_lib/evaluation/reject_verification.py \
#  --checkpoint_path=/gpfs/data/gpfs0/k.fedyanin/space/models/magface/ms1mv2_ir50_ddp/arcface+magface.pth\
#  --dataset_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/big \
#  --pairs_table_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/metadata_refuse_verification/pairs_1000000_prob_0.5.csv \
#  --config_path=./configs/magface/arcface+ir50.yaml \
#  --batch_size=16 \
#  --uncertainty_strategy=backbone+uncertainty_model \
#  --uncertainty_mode=confidence \
#  --FARs 0.0001 0.0005 0.001 0.005 0.01 0.05 \
#  --rejected_portions $(seq 0 0.002 0.5) \
#  --distance_uncertainty_metrics cosine_mean cosine_harmonic-harmonic cosine_mul cosine_squared-sum cosine_squared-harmonic \
#  --device_id=0 \
#  --save_fig_path=/trinity/home/r.kail/faces/figures/test

## Arcface backbone + MagFace uncertainty
#python3 ./face_lib/evaluation/reject_verification.py \
#  --checkpoint_path=/gpfs/data/gpfs0/k.fedyanin/space/models/magface/ms1mv2_ir50_ddp/arcface+magface.pth\
#  --dataset_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/big \
#  --pairs_table_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/metadata_refuse_verification/pairs_1000_prob_0.5.csv \
#  --config_path=./configs/magface/arcface+ir50.yaml \
#  --batch_size=16 \
#  --uncertainty_strategy=backbone+uncertainty_model \
#  --uncertainty_mode=confidence \
#  --FARs 0.0001 0.0005 0.001 0.005 0.01 0.05 \
#  --rejected_portions $(seq 0 0.002 0.5) \
#  --distance_uncertainty_metrics cosine_mean cosine_harmonic-harmonic cosine_mul cosine_squared-sum cosine_squared-harmonic \
#  --device_id=0 \
#  --save_fig_path=/beegfs/home/r.kail/faces/figures/test


## Cheaty version of distance function, which aims to proof concept, that uncertainty can improve distance
#python3 ./face_lib/evaluation/reject_verification.py \
#    --checkpoint_path=/gpfs/data/gpfs0/k.fedyanin/space/models/scale/02_sigm_mul_selection/64/checkpoint.pth \
#    --dataset_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/big \
#    --pairs_table_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/metadata_refuse_verification/pairs_1000000_prob_0.5.csv \
#    --config_path=./configs/scale/02_sigm_mul_coef_selection/64.yaml \
#    --batch_size=64 \
#    --uncertainty_strategy=scale \
#    --uncertainty_mode=confidence \
#    --FARs 0.0001 0.0005 0.001 0.005 0.01 0.05 \
#    --rejected_portions $(seq 0 0.002 0.5) \
#    --distance_uncertainty_metrics centered-cosine_harmonic-harmonic scale-mul-centered-cosine_harmonic-harmonic scale-harmonic-centered-cosine_harmonic-harmonic scale-sqrt-mul-centered-cosine_harmonic-harmonic scale-sqrt-harmonic-centered-cosine_harmonic-harmonic \
#    --device_id=0 \
#    --save_fig_path=/beegfs/home/r.kail/faces/figures/17_different_dist_funcs/cheat_scale_64sigm

## Evaluate MagFace
#python3 ./face_lib/evaluation/reject_verification.py \
#  --checkpoint_path=/gpfs/data/gpfs0/k.fedyanin/space/models/magface/ms1mv2_ir50_ddp/adapted.pth \
#  --dataset_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/big \
#  --pairs_table_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/metadata_refuse_verification/pairs_1000000_prob_0.5.csv \
#  --config_path=./configs/magface/ir50.yaml \
#  --batch_size=32 \
#  --uncertainty_strategy=magface_precalculated \
#  --uncertainty_mode=confidence \
#  --precalculated_path=/gpfs/data/gpfs0/k.fedyanin/space/IJB/IJB-C/magface_evaluation/features/1M_pairs/ir100 \
#  --distaces_batch_size=128 \
#  --FARs 0.0001 0.0005 0.001 0.005 0.01 0.05 \
#  --rejected_portions $(seq 0 0.002 0.5) \
#  --distance_uncertainty_metrics cosine_mean cosine_harmonic-harmonic cosine_mul cosine_squared-sum cosine_squared-harmonic \
#  --device_id=0 \
#  --save_fig_path=/trinity/home/r.kail/faces/figures/test

# ================================ Test_val_split ================================================================== #

# Estimate statistics on validation, use them on test
python3 ./face_lib/evaluation/reject_verification.py \
    --checkpoint_path=/gpfs/data/gpfs0/k.fedyanin/space/models/scale/02_sigm_mul_selection/32/checkpoint.pth \
    --dataset_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/big \
    --pairs_table_path=/gpfs/data/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/metadata_refuse_verification/val_test/test_pairs_1000_prob_0.5.csv \
    --val_pairs_table_path=/gpfs/data/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/metadata_refuse_verification/val_test/val_pairs_1000_prob_0.5.csv \
    --config_path=./configs/scale/02_sigm_mul_coef_selection/32.yaml \
    --batch_size=64 \
    --uncertainty_strategy=scale \
    --uncertainty_mode=confidence \
    --FARs 0.0001 0.0005 0.001 0.005 0.01 0.05 \
    --rejected_portions $(seq 0 0.002 0.5) \
    --distance_uncertainty_metrics \
        biased-cosine_harmonic-harmonic \
        scale-mul-biased-cosine_harmonic-harmonic \
        scale-harmonic-biased-cosine_harmonic-harmonic \
        scale-sqrt-mul-biased-cosine_harmonic-harmonic \
        scale-sqrt-harmonic-biased-cosine_harmonic-harmonic \
    --device_id=0 \
    --save_fig_path=/beegfs/home/r.kail/faces/figures/25_val_test/test

## L2 Norm
#python3 ./face_lib/evaluation/reject_verification.py \
#    --checkpoint_path=/gpfs/data/gpfs0/k.fedyanin/space/models/arcface/backbones/classic_packed.pth \
#    --dataset_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/big \
#    --pairs_table_path=/gpfs/data/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/metadata_refuse_verification/val_test/test_pairs_1000_prob_0.5.csv \
#    --val_pairs_table_path=/gpfs/data/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/metadata_refuse_verification/val_test/val_pairs_1000_prob_0.5.csv \
#    --config_path=./configs/models/arcface_emb_norm.yaml \
#    --batch_size=64 \
#    --uncertainty_strategy=emb_norm \
#    --uncertainty_mode=confidence \
#    --FARs 0.0001 0.0005 0.001 0.005 0.01 0.05 \
#    --rejected_portions $(seq 0 0.002 0.5) \
#    --distance_uncertainty_metrics \
#        biased-cosine_harmonic-harmonic \
#        scale-mul-biased-cosine_harmonic-harmonic \
#        scale-harmonic-biased-cosine_harmonic-harmonic \
#        scale-sqrt-mul-biased-cosine_harmonic-harmonic \
#        scale-sqrt-harmonic-biased-cosine_harmonic-harmonic \
#    --device_id=0 \
#    --save_fig_path=/beegfs/home/r.kail/faces/figures/25_val_test/norm
#
#
#python3 ./face_lib/evaluation/reject_verification.py \
#    --checkpoint_path=/gpfs/data/gpfs0/k.fedyanin/space/models/pfe/normalized_pfe/sota.pth \
#    --dataset_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/big \
#    --pairs_table_path=/gpfs/data/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/metadata_refuse_verification/val_test/test_pairs_1000_prob_0.5.csv \
#    --val_pairs_table_path=/gpfs/data/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/metadata_refuse_verification/val_test/val_pairs_1000_prob_0.5.csv \
#    --config_path=./configs/models/iresnet_ms1m_pfe_normalized.yaml \
#    --batch_size=16 \
#    --uncertainty_strategy=head \
#    --FARs 0.0001 0.0005 0.001 0.005 0.01 0.05 \
#    --rejected_portions $(seq 0 0.002 0.5) \
#    --distance_uncertainty_metrics \
#        biased-cosine_harmonic-harmonic \
#        pfe-mul-biased-cosine_harmonic-harmonic \
#        pfe-harmonic-biased-cosine_harmonic-harmonic \
#        pfe-sqrt-mul-biased-cosine_harmonic-harmonic \
#        pfe-sqrt-harmonic-biased-cosine_harmonic-harmonic \
#    --device_id=0 \
#    --save_fig_path=/beegfs/home/r.kail/faces/figures/25_val_test/PFE