python main_pretrain.py --project kcc_pretrain \
    --dataset CIFAR100 --data_dir /root/default/dataset/CIFAR \
    --gpus 2 --precision 16 --max_epochs 200 \
    --num_labeled_classes 50 --num_unlabeled_classes 50 \
    --comment 50_50_none \
    --arch vit_small

python main_discover.py --project kcc_discover\
    --dataset CIFAR100 --data_dir /root/default/dataset/CIFAR \
    --gpus 2 --max_epochs 500 --batch_size 256 --num_heads 4 --precision 16 --multicrop --overcluster_factor 5 \
    --num_labeled_classes 50 --num_unlabeled_classes 50 \
    --comment 50_50_none \
    --arch vit_small \
    --pretrained ./checkpoints/pretrain-vit_small-CIFAR100-50_50_ssl.cp