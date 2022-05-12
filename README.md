# Entity Neural Network Trainer

[![Actions Status](https://github.com/entity-neural-network/enn-trainer/workflows/Checks/badge.svg)](https://github.com/entity-neural-network/enn-trainer/actions)
[![PyPI](https://img.shields.io/pypi/v/enn-trainer.svg?style=flat-square)](https://pypi.org/project/enn-trainer/)
[![Documentation Status](https://readthedocs.org/projects/entity-gym/badge/?version=latest&style=flat-square)](https://enn-trainer.readthedocs.io/en/latest/?badge=latest)
[![Discord](https://img.shields.io/discord/913497968701747270?style=flat-square)](https://discord.gg/SjVqhSW4Qf)

PPO and behavioral cloning implementations compatible with [Entity Gym](https://github.com/entity-neural-network/entity-gym).

## Installation

```bash
pip install enn-trainer
pip install setuptools==59.5.0
pip install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
```

## Usage

Training policy for an entity-gym example environment:

```bash
python -m  enn_trainer.train env.id=Xor
```

List all available hyperparameters:

```bash
python -m enn_trainer.train --hps-info
```

Training policy for a custom entity-gym environment:

```python
import hyperstate
from enn_trainer.config import TrainConfig
from enn_trainer.train import train
from custom_env import CustomEnv


@hyperstate.command(TrainConfig)
def main(cfg: TrainConfig) -> None:
    train(cfg=cfg, env_cls=CustomEnv)

if __name__ == "__main__":
    main()
```

To run behavioral cloning on recorded samples:

```bash
# Download data (261MB)
# Larger 5GB file with 1M samples: https://www.dropbox.com/s/o7jf4r7m0xtm80p/enhanced250m-1m-v2.blob?dl=1
wget 'https://www.dropbox.com/s/es84ml3wltxdmnh/enhanced250m-60k.blob?dl=1' -O enhanced250m-60k.blob
# Run training
poetry run python enn_trainer/enn_trainer/supervised.py dataset_path=enhanced250m-60k.blob optim.batch_size=256 fast_eval_samples=256
```
