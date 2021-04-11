python train_pfe_pipeline.py \
--env-config configs/evs/cpu.yaml \
--model-config configs/models/spherenet_pfe.yaml \
--optimizer-config configs/optims/sgd.yaml \
--dataset-config configs/datasets/casia_mtcnncaffe_aligned_nooverlap.yaml \
--evaluation-configs configs/evaluation/6000_pairs_lfw.yaml \
--debug \
--freeze-backbone