#python3 -m torch.distributed.launch --nproc_per_node 1 trainers/train_pair_face_classification.py \
#  --model-config configs/models/pair_classifiers/smart_cosine.yaml
#
#python3 -m torch.distributed.launch --nproc_per_node 1 trainers/train_pair_face_classification.py \
#  --model-config configs/models/pair_classifiers/smart_cosine_initialized.yaml
#
#python3 -m torch.distributed.launch --nproc_per_node 1 trainers/train_pair_face_classification.py \
#  --model-config configs/models/pair_classifiers/bilinear.yaml
#
#python3 -m torch.distributed.launch --nproc_per_node 1 trainers/train_pair_face_classification.py \
#  --model-config configs/models/pair_classifiers/bilinear_bigger_lr.yaml
#
#python3 -m torch.distributed.launch --nproc_per_node 1 trainers/train_pair_face_classification.py \
#  --model-config configs/models/pair_classifiers/3layer_perceptron.yaml