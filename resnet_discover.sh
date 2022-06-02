python main_discover.py --project kcc_discover \
--gpus 2 --arch resnet50 --comment 50_50_INET \
--dataset CIFAR100 --num_labeled_classes 50 --num_unlabeled_classes 50 \
--batch_size 512 --max_epochs 500 \
--num_heads 4 --comment 50_50 --precision 16 --multicrop \
--data_dir /root/dataset/CIFAR \
--pretrained /root/dhk/category_discovery/checkpoints/pretrain-resnet50-CIFAR100-50_50_INET.cp

python main_discover.py --project kcc_discover \
--gpus 2 --arch resnet50 --comment 50_50_SUP \
--dataset CIFAR100 --num_labeled_classes 50 --num_unlabeled_classes 50 \
--batch_size 512 --max_epochs 500 \
--num_heads 4 --comment 50_50 --precision 16 --multicrop \
--data_dir /root/dataset/CIFAR \
--pretrained /root/dhk/category_discovery/checkpoints/pretrain-resnet50-CIFAR100-50_50_SUP.cp

python main_discover.py --project kcc_discover \
--gpus 2 --arch resnet50 --comment 50_50_DINO \
--dataset CIFAR100 --num_labeled_classes 50 --num_unlabeled_classes 50 \
--batch_size 512 --max_epochs 500 \
--num_heads 4 --comment 50_50 --precision 16 --multicrop \
--data_dir /root/dataset/CIFAR \
--pretrained /root/dhk/category_discovery/checkpoints/pretrain-resnet50-CIFAR100-50_50_DINO.cp