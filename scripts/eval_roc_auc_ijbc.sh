## Eval ROC AUC
#python3 ./face_lib/utils/roc_auc_ijbc_evaluation.py \
#  --checkpoint_path=/gpfs/data/gpfs0/k.fedyanin/space/models/pair_classifiers/02_bilinear/checkpoints/sota.pth \
#  --dataset_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/debug \
#  --pairs_table_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/metadata_refuse_verification/pairs_100_prob_0.5_debug.csv \
#  --config_path=./configs/models/pair_classifiers/smart_cosine.yaml \
#  --batch_size=64 \
#  --pairs_distance_strategy=classifier \
#  --device_id=0 \
#  --save_results_path=/gpfs/gpfs0/r.zainulin/figures_test

python3 ./face_lib/utils/roc_auc_ijbc_evaluation.py \
  --checkpoint_path=/gpfs/data/gpfs0/k.fedyanin/space/models/pair_classifiers/02_bilinear/checkpoints/sota.pth \
  --dataset_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/big \
  --pairs_table_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/metadata_refuse_verification/pairs_1000000_prob_0.5.csv \
  --config_path=./configs/models/pair_classifiers/smart_cosine.yaml \
  --batch_size=64 \
  --pairs_distance_strategy=classifier \
  --device_id=0 \
  --save_results_path=/gpfs/gpfs0/r.zainulin/figures_test