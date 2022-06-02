python main_pretrain.py --project kcc_pretrain \
--gpus 2 --arch resnet50 --comment 80_20_INET \
--dataset CIFAR100 --num_labeled_classes 80 --num_unlabeled_classes 20 \
--precision 16 --max_epochs 100 --base_lr 1e-1 --min_lr 1e-3 \
--data_dir /root/dataset/CIFAR \
--log_dir /root/dhk/category_discovery/logs \
--checkpoint_dir /root/dhk/category_discovery/checkpoints \
--backbone /root/dhk/category_discovery/backbones/img_resnet50_pretrain.pth

python main_pretrain.py --project kcc_pretrain \
--gpus 2 --arch resnet50 --comment 80_20_SUP \
--dataset CIFAR100 --num_labeled_classes 80 --num_unlabeled_classes 20 \
--precision 16 --max_epochs 100 --base_lr 1e-1 --min_lr 1e-3 \
--data_dir /root/dataset/CIFAR \
--log_dir /root/dhk/category_discovery/logs \
--checkpoint_dir /root/dhk/category_discovery/checkpoints \
--backbone /root/dhk/category_discovery/backbones/sup_resnet50_pretrain.pth

python main_pretrain.py --project kcc_pretrain \
--gpus 2 --arch resnet50 --comment 80_20_DINO \
--dataset CIFAR100 --num_labeled_classes 80 --num_unlabeled_classes 20 \
--precision 16 --max_epochs 100 --base_lr 1e-1 --min_lr 1e-3 \
--data_dir /root/dataset/CIFAR \
--log_dir /root/dhk/category_discovery/logs \
--checkpoint_dir /root/dhk/category_discovery/checkpoints \
--backbone /root/dhk/category_discovery/backbones/sup_resnet50_pretrain.pth