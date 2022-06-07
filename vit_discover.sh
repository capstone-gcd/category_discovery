python main_discover.py --entity dhk --project kcc_discover --comment 50_50_CE \
    --gpus 1 --arch vit_tiny --patch_size 4 --num_heads 4 \
    --dataset CIFAR100 --num_labeled_classes 50 --num_unlabeled_classes 50 \
    --batch_size 256 --num_workers 8 --img_size 32 --multicrop\
    --precision 16 --max_epochs 500 --base_lr 1e-4 --min_lr 2e-5 --temperature 0.07 \
    --data_dir /root/dhk/dataset \
    --pretrained checkpoints/pretrain-vit_tiny_4-CIFAR100-50_50_CE.cp

python main_discover.py --entity dhk --project kcc_discover --comment 80_20_CE \
    --gpus 1 --arch vit_tiny --patch_size 4 --num_heads 4 \
    --dataset CIFAR100 --num_labeled_classes 80 --num_unlabeled_classes 20 \
    --batch_size 256 --num_workers 8 --img_size 32 --multicrop\
    --precision 16 --max_epochs 500 --base_lr 1e-4 --min_lr 2e-5 --temperature 0.07 \
    --data_dir /root/dhk/dataset \
    --pretrained checkpoints/pretrain-vit_tiny_4-CIFAR100-80_20_CE.cp