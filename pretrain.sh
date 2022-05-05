conda create --name cap python=3.8 -y
conda activate cap
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11.0 -c pytorch -y
pip install pytorch-lightning==1.1.3 lightning-bolts==0.3.0 wandb sklearn

python main_pretrain.py --gpus 2 --num_labeled_classes 5 --num_unlabeled_classes 5 --comment 5_5 \
    --precision 16 --max_epochs 200 \
    --pretrained /root/default/kcc/category_discovery/checkpoints/backbone/dino_backbone.pth