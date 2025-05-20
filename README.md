## Setup
See [README_DTP.md](README_DTP.md) for setup instructions.

After setting up the environment, see [FIX.md](FIX.md) for instructions on how to fix the pytorch version issues.

Generate additional depth data using the following command (takes ~20 min):
```bash
pip install transformers

python custom-tools/generate_depth.py --img_dir data/nightcity-fine/train/img --depth_dir data/nightcity-fine/train/depth
python custom-tools/generate_depth.py --img_dir data/nightcity-fine/val/img --depth_dir data/nightcity-fine/val/depth
```

## Depth training
Make sure depth data was generated and the fixes were applied (there are two). Then request a node for ~10 hours with 2 GPUs:
```bash
./start_node.sh 600 2
```
Then on the node, run the following command to start the training:
```bash
conda activate <env>
sh custom-tools/dist_train.sh configs/night_depth.py 2
```

Training results will be saved in `work_dirs/cfg_depth/`. After training, drag that folder into `results/` rename it appropriately and add a gitignore exception for the best checkpoint (see .gitignore), and push to save the results.

## Commands
```bash
# Train
python custom-tools/train.py configs/night_depth.py

# Distributed training
sh custom-tools/dist_train.sh configs/night_depth.py <num_gpus>

# Test
python custom-tools/test.py configs/night_depth.py <checkpoint> --eval mIoU --aug-test
```