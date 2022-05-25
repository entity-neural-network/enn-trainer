from entity_gym.examples import ENV_REGISTRY
from hyperstate import StateManager
from rogue_net.relpos_encoding import RelposEncodingConfig
from rogue_net.rogue_net import RogueNetConfig

from enn_trainer.config import RolloutConfig
from enn_trainer.train import (
    EnvConfig,
    OptimizerConfig,
    PPOConfig,
    State,
    TrainConfig,
    init_train_state,
    train,
)


def _train(cfg: TrainConfig) -> float:
    sm = StateManager(TrainConfig, State, init_train_state, None)
    sm._config = cfg
    return train(sm, ENV_REGISTRY[cfg.env.id])


def test_multi_armed_bandit() -> None:
    cfg = TrainConfig(
        total_timesteps=500,
        cuda=False,
        ppo=PPOConfig(ent_coef=0.0, gamma=0.5, anneal_entropy=False),
        env=EnvConfig(id="MultiArmedBandit"),
        rollout=RolloutConfig(steps=16, processes=2, num_envs=4),
        net=RogueNetConfig(n_layer=0, d_model=16),
        optim=OptimizerConfig(lr=0.05, bs=16, update_epochs=4),
    )
    meanrew = _train(cfg)
    print(f"Final mean reward: {meanrew}")
    assert meanrew > 0.99 / 32


def test_minefield() -> None:
    cfg = TrainConfig(
        total_timesteps=256,
        cuda=False,
        net=RogueNetConfig(d_model=16),
        env=EnvConfig(id="Minefield"),
        rollout=RolloutConfig(steps=16, num_envs=4),
        optim=OptimizerConfig(bs=64),
        ppo=PPOConfig(),
    )
    meanrew = _train(cfg)
    print(f"Final mean reward: {meanrew}")
    assert meanrew >= 0.0


def test_multi_snake() -> None:
    cfg = TrainConfig(
        total_timesteps=256,
        cuda=False,
        net=RogueNetConfig(d_model=16),
        env=EnvConfig(id="MultiSnake"),
        rollout=RolloutConfig(steps=16, num_envs=4),
        optim=OptimizerConfig(bs=64),
        ppo=PPOConfig(),
    )
    meanrew = _train(cfg)
    print(f"Final mean reward: {meanrew}")
    assert meanrew >= 0.0


def test_not_hotdog() -> None:
    cfg = TrainConfig(
        total_timesteps=1000,
        cuda=False,
        net=RogueNetConfig(d_model=16, n_layer=1),
        env=EnvConfig(id="NotHotdog"),
        rollout=RolloutConfig(steps=16, num_envs=8),
        optim=OptimizerConfig(bs=16, lr=0.005),
        ppo=PPOConfig(ent_coef=0.0, gamma=0.5),
    )
    meanrew = _train(cfg)
    print(f"Final mean reward: {meanrew}")
    assert meanrew >= 0.99


def test_masked_count() -> None:
    cfg = TrainConfig(
        total_timesteps=2000,
        cuda=False,
        net=RogueNetConfig(d_model=16, n_layer=1),
        env=EnvConfig(id="Count", kwargs='{"masked_choices": 2}'),
        rollout=RolloutConfig(steps=1, num_envs=16),
        optim=OptimizerConfig(bs=16, lr=0.01),
        ppo=PPOConfig(),
    )
    meanrw = _train(cfg)
    print(f"Final mean reward: {meanrw}")
    assert meanrw >= 0.99


def test_pick_matching_balls() -> None:
    cfg = TrainConfig(
        total_timesteps=256,
        cuda=False,
        net=RogueNetConfig(d_model=16),
        env=EnvConfig(id="PickMatchingBalls"),
        rollout=RolloutConfig(steps=16, num_envs=4),
        optim=OptimizerConfig(bs=64),
        ppo=PPOConfig(),
    )
    meanrew = _train(cfg)
    print(f"Final mean reward: {meanrew}")
    assert meanrew >= 0.0


def test_cherry_pick() -> None:
    cfg = TrainConfig(
        total_timesteps=256,
        cuda=False,
        net=RogueNetConfig(d_model=16),
        env=EnvConfig(id="CherryPick"),
        rollout=RolloutConfig(steps=16),
        optim=OptimizerConfig(bs=64),
        ppo=PPOConfig(),
    )
    meanrew = _train(cfg)
    print(f"Final mean reward: {meanrew}")


def test_relpos_encoding() -> None:
    cfg = TrainConfig(
        total_timesteps=10000,
        cuda=False,
        net=RogueNetConfig(
            d_model=16,
            n_layer=2,
            relpos_encoding=RelposEncodingConfig(
                extent=[1, 1],
                position_features=["x", "y"],
                per_entity_values=True,
            ),
        ),
        env=EnvConfig(id="FloorIsLava"),
        rollout=RolloutConfig(steps=2, num_envs=64),
        optim=OptimizerConfig(bs=32, lr=0.03),
        ppo=PPOConfig(
            ent_coef=1.0,
            anneal_entropy=True,
        ),
    )
    meanrew = _train(cfg)
    print(f"Final mean reward: {meanrew}")
    assert meanrew >= 0.95

    cfg.net.relpos_encoding = None
    meanrew = _train(cfg)
    print(f"Final mean reward: {meanrew}")
    assert meanrew < 0.2


def test_asymmetric_relpos_encoding() -> None:
    cfg = TrainConfig(
        total_timesteps=3000,
        cuda=False,
        net=RogueNetConfig(
            d_model=16,
            n_layer=2,
            relpos_encoding=RelposEncodingConfig(
                extent=[5, 1],
                position_features=["x", "y"],
                per_entity_values=True,
            ),
        ),
        env=EnvConfig(id="FloorIsLava"),
        rollout=RolloutConfig(steps=1, num_envs=64),
        optim=OptimizerConfig(bs=16, lr=0.02),
        ppo=PPOConfig(
            ent_coef=0.1,
            anneal_entropy=True,
        ),
    )

    meanrew = _train(cfg)
    print(f"Final mean reward: {meanrew}")
    assert meanrew >= 0.15


def test_rock_paper_scissors() -> None:
    cfg = TrainConfig(
        total_timesteps=4000,
        cuda=False,
        net=RogueNetConfig(d_model=16, n_layer=2),
        env=EnvConfig(id="RockPaperScissors"),
        rollout=RolloutConfig(steps=1, num_envs=256),
        optim=OptimizerConfig(bs=64, lr=0.01),
        ppo=PPOConfig(ent_coef=0.1),
    )
    meanrew = _train(cfg)
    print(f"Final mean reward: {meanrew}")
    assert 0.8 <= meanrew <= 1.1

    cfg.env.kwargs = '{"cheat": true}'
    meanrew = _train(cfg)
    print(f"Final mean reward: {meanrew}")
    assert meanrew > 1.9
