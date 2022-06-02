python /root/dhk/category_discovery/main_pretrain.py --project kcc_pretrain \
--gpus 2 --arch vit_base --comment 80_20_SUP \
--dataset CIFAR100 --num_labeled_classes 80 --num_unlabeled_classes 20 \
--precision 16 --max_epochs 300 --base_lr 1e-4 --min_lr 2e-5 \
--data_dir /root/dataset/CIFAR \
--log_dir /root/dhk/category_discovery/logs \
--checkpoint_dir /root/dhk/category_discovery/checkpoints \
--backbone /root/dhk/category_discovery/backbones/sup_cifar100_vitbase16_pretrain.pth

python /root/dhk/category_discovery/main_pretrain.py --project kcc_pretrain \
--gpus 2 --arch vit_base --comment 50_50_SUP \
--dataset CIFAR100 --num_labeled_classes 50 --num_unlabeled_classes 50 \
--precision 16 --max_epochs 300 --base_lr 1e-4 --min_lr 2e-5 \
--data_dir /root/dataset/CIFAR \
--log_dir /root/dhk/category_discovery/logs \
--checkpoint_dir /root/dhk/category_discovery/checkpoints \
--backbone /root/dhk/category_discovery/backbones/sup_cifar100_vitbase16_pretrain.pth