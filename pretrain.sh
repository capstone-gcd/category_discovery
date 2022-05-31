conda create --name cap python=3.8 -y
conda activate cap
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11.0 -c pytorch -y
pip install pytorch-lightning==1.1.3 lightning-bolts==0.3.0 wandb sklearn matplotlib gym
python main_pretrain.py --dataset CIFAR100 --gpus 2 --num_labeled_classes 50 --num_unlabeled_classes 50 --comment 50_50 \
    --precision 16 --max_epochs 200


python -m torch.distributed.launch --nproc_per_node=8 main_dino.py --arch resnet50 --optimizer sgd --lr 0.03 --weight_decay 1e-4 --weight_decay_end 1e-4 --global_crops_scale 0.14 1 --local_crops_scale 0.05 0.14 --data_path /path/to/imagenet/train --output_dir /path/to/saving_dir


python main_discover.py --dataset CIFAR10 --data_dir /root/dataset/CIFAR \
    --gpus 2 --max_epochs 500 --batch_size 256 --num_labeled_classes 5 --num_unlabeled_classes 5 \
    --pretrained ./checkpoints/backbone/dino_deitsmall8_pretrain.pth \
    --num_heads 4 --comment 5_5_vits8 --precision 16 --multicrop --overcluster_factor 10 \
    --arch vit_small