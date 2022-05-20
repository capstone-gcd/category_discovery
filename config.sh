conda create -n cap python=3.8
conda activate cap
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11.0 -c pytorch -y
pip install pytorch-lightning==1.1.3 lightning-bolts==0.3.0 wandb sklearn matplotlib gym
mkdir -p logs/wandb checkpoints
chmod +777 logs
chmod +755 *.sh