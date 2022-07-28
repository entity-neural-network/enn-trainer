# Entity Neural Network Trainer

[![Actions Status](https://github.com/entity-neural-network/enn-trainer/workflows/Checks/badge.svg)](https://github.com/entity-neural-network/enn-trainer/actions)
[![PyPI](https://img.shields.io/pypi/v/enn-trainer.svg?style=flat-square)](https://pypi.org/project/enn-trainer/)
[![Documentation Status](https://readthedocs.org/projects/entity-gym/badge/?version=latest&style=flat-square)](https://enn-trainer.readthedocs.io/en/latest/?badge=latest)
[![Discord](https://img.shields.io/discord/913497968701747270?style=flat-square)](https://discord.gg/SjVqhSW4Qf)

ENN Trainer allow you to train reinforcement learning agents for [Entity Gym](https://github.com/entity-neural-network/entity-gym) environments with PPO or behavioral cloning.

## Installation

```bash
pip install enn-trainer
pip install torch==1.12.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
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

Setting up a training script for a custom entity-gym environment (replace `TreasureHunt` with your environment):

```python
from enn_trainer import TrainConfig, State, init_train_state, train
from entity_gym.examples.tutorial import TreasureHunt
import hyperstate

@hyperstate.stateful_command(TrainConfig, State, init_train_state)
def main(state_manager: hyperstate.StateManager) -> None:
    train(state_manager=state_manager, env=TreasureHunt)

if __name__ == "__main__":
    main()
```

You can find more detailed guides and an API reference on the [documentation website](https://enn-trainer.readthedocs.io/en/latest/). If you run into issues or have a question, feel free to open an issue or ask on [discord](https://discord.gg/SjVqhSW4Qf).
