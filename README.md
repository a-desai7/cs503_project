## Setup
See [README_DTP.md](README_DTP.md) for setup instructions.

After setting up the environment, see [FIX.md](FIX.md) for instructions on how to fix the pytorch version issues.

## Commands
```bash
# Train
python custom-tools/train.py checkpoints/night/cfg.py 

# Distributed training
sh custom-tools/dist_train.sh checkpoints/night/cfg.py <num_gpus>

# Test
python custom-tools/test.py checkpoints/night/cfg.py <checkpoint> --eval mIoU --aug-test
```