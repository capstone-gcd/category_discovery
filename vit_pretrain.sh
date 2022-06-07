python main_pretrain.py --entity dhk --project kcc_pretrain --comment 50_50_CE \
    --gpus 1 --arch vit_tiny --patch_size 4 --optim adamw  \
    --dataset CIFAR100 --num_labeled_classes 50 --num_unlabeled_classes 50 \
    --batch_size 512 --num_workers 8 --img_size 32 \
    --precision 16 --max_epochs 300 --base_lr 1e-4 --min_lr 2e-5 --temperature 0.07 \
    --data_dir /root/dhk/dataset \
    --log_dir /root/dhk/category_discovery/logs \
    --checkpoint_dir /root/dhk/category_discovery/checkpoints

python main_pretrain.py --entity dhk --project kcc_pretrain --comment 80_20_CE \
    --gpus 1 --arch vit_tiny --patch_size 4 --optim adamw  \
    --dataset CIFAR100 --num_labeled_classes 80 --num_unlabeled_classes 20 \
    --batch_size 512 --num_workers 8 --img_size 32 \
    --precision 16 --max_epochs 300 --base_lr 1e-4 --min_lr 2e-5 --temperature 0.07 \
    --data_dir /root/dhk/dataset \
    --log_dir /root/dhk/category_discovery/logs \
    --checkpoint_dir /root/dhk/category_discovery/checkpoints