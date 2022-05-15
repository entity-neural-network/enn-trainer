import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import hyperstate
from hyperstate.schema.rewrite_rule import ChangeDefault, RewriteRule
from rogue_net.rogue_net import RogueNetConfig


@dataclass
class EnvConfig:
    """Environment settings.

    :param kwargs: JSON dictionary with keyword arguments for the environment.
    :param id: The id of the environment.
    :param validate: Perform runtime checks to ensure that the environment correctly implements the interface.
    """

    kwargs: str = "{}"
    id: str = "MoveToOrigin"
    validate: bool = True


@dataclass
class RolloutConfig:
    """Settings for rollout phase of PPO.

    :param steps: The number of steps to run in each environment per policy rollout.
    :param num_envs: The number of parallel game environments.
    :param processes: The number of processes to use to collect env data. The envs are split as equally as possible across the processes.
    """

    steps: int = 16
    num_envs: int = 128
    processes: int = 4


@dataclass
class EvalConfig:
    """Evaluation settings.

    :param interval: Number of environment steps between evaluations.
    :param capture_videos: Render videos of the environments during evaluation.
    :param capture_samples: Write samples from evals to this file.
    :param capture_logits: Record full logits of the agent during evaluation (requires ``capture_samples``).
    :param capture_samples_subsample: Only persist every nth sample, chosen randomly (requires ``capture_samples``).
    :param run_on_first_step: Whether to run an eval on step 0.
    :param env: Settings for the eval environment. If not set, use same settings as rollouts.
    :param num_envs: The number of parallel game environments to use for evaluation. If not set, use same settings as rollouts.
    :param processes: The number of processes used to run the environment. If not set, use same settings as rollouts.
    :param opponent: Path to opponent policy to evaluate against.
    :param opponent_only: Don't evaluate the policy, but instead run the opponent against itself.
    """

    steps: int
    interval: int

    num_envs: Optional[int] = None
    processes: Optional[int] = None
    env: Optional[EnvConfig] = None
    capture_videos: bool = False
    capture_samples: str = ""
    capture_logits: bool = True
    capture_samples_subsample: int = 1
    run_on_first_step: bool = True
    opponent: Optional[str] = None
    opponent_only: bool = False


@dataclass
class PPOConfig:
    """Proximal Policy Optimization settings.

    :param gae: Whether to use generalized advantage estimation for advantage computation.
    :param gamma: Temporal discount factor gamma.
    :param gae_lambda: The lambda for the generalized advantage estimation.
    :param norm_adv: Normalize advantages to 0 mean and 1 std.
    :param clip_coef: The PPO surrogate clipping coefficient.
    :param clip_vloss: Whether to use a clipped loss for the value function.
    :param ent_coef: Coefficient for entropy loss term.
    :param vf_coef: Coefficient for value function loss term.
    :param target_kl: Stop optimization if the KL divergence between the old and new policy exceeds this threshold.
    :param anneal_entropy: Linearly anneal the entropy coefficient from its initial value to 0.
    """

    gae: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.1
    vf_coef: float = 0.5
    target_kl: Optional[float] = None
    anneal_entropy: bool = True


@dataclass
class OptimizerConfig:
    """Optimizer settings.

    :param lr: Adam learning rate.
    :param bs: Batch size.
    :param micro_bs: Micro batch size size used for gradient accumulation. Using a lower micro batch
        size reduces memory usage and performance without affecting training dyanmics.
    :param weight_decay: Adam weight decay.
    :param anneal_lr: Linearly anneal learning rate from initial learning rate to 0.
    :param update_epochs: Number of optimizer passes over each batch of rollout samples.
    :param max_grad_norm: Gradient norm clipping.
    """

    lr: float = 0.001
    bs: int = 1024
    weight_decay: float = 0.0
    micro_bs: Optional[int] = None
    anneal_lr: bool = True
    update_epochs: int = 3
    max_grad_norm: float = 2.0


@dataclass
class TrainConfig(hyperstate.Versioned):
    """Training settings.

    :param env: Settings for the environment.
    :param net: Hyperparameters for policy network.
    :param optim: Hyperparameters for optimizer.
    :param ppo: Hyperparameters for PPO.
    :param rollout: Hyperparameters for rollout phase.
    :param eval: Optional evaluation settings.
    :param vf_net: Hyperparameters for value function network (if not set, policy and value function share the same network).
    :param name: The name of the experiment.
    :param seed: Seed of the experiment.
    :param total_timesteps: Total number of timesteps to run for.
    :param max_train_time: Train for at most this many seconds.
    :param torch_deterministic: Sets the value of ``torch.backends.cudnn.deterministic``.
    :param cuda: If ``True``, cuda will be enabled by default.
    :param track: Track experiment metrics with Weights and Biases.
    :param wandb_project_name: Name of the W&B project to log metrics to.
    :param wandb_entity: The entity (team) of the W&B project to log metrics to.
    :param capture_samples: Write all samples collected from environments during training to this file.
    :param capture_logits: Record full logits of the agent (requires ``capture_samples``).
    :param capture_samples_subsample: Only persist every nth sample, chosen randomly (requires ``capture_samples``).
    :param data_dir: Directory to save output from training and logging.
    :param cuda_empty_cache: Empty the torch cuda cache after each optimizer step.
    """

    env: EnvConfig
    net: RogueNetConfig
    optim: OptimizerConfig
    ppo: PPOConfig
    rollout: RolloutConfig
    eval: Optional[EvalConfig] = None
    vf_net: Optional[RogueNetConfig] = None

    name: str = field(default_factory=lambda: os.path.basename(__file__).rstrip(".py"))
    seed: int = 1
    total_timesteps: int = 1000000
    max_train_time: Optional[int] = None
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "enn-ppo"
    wandb_entity: str = "entity-neural-network"
    capture_samples: Optional[str] = None
    capture_logits: bool = False
    capture_samples_subsample: int = 1
    trial: Optional[int] = None
    data_dir: str = "."
    cuda_empty_cache: bool = False

    @classmethod
    def version(clz) -> int:
        return 3

    @classmethod
    def upgrade_rules(clz) -> Dict[int, List[RewriteRule]]:
        return {
            0: [
                ChangeDefault(
                    field=("net", "relpos_encoding", "per_entity_values"),
                    old_default=True,
                    new_default=False,
                ),
                ChangeDefault(
                    field=("vf_net", "relpos_encoding", "per_entity_values"),
                    old_default=True,
                    new_default=False,
                ),
            ],
            1: [
                ChangeDefault(field=("net", "n_layer"), old_default=1, new_default=2),
                ChangeDefault(field=("net", "d_model"), old_default=64, new_default=32),
                ChangeDefault(
                    field=("optim", "lr"), old_default=0.00025, new_default=0.001
                ),
                ChangeDefault(field=("optim", "bs"), old_default=128, new_default=1024),
                ChangeDefault(
                    field=("ppo", "ent_coef"), old_default=0.01, new_default=0.1
                ),
                ChangeDefault(
                    field=("ppo", "anneal_entropy"), old_default=False, new_default=True
                ),
                ChangeDefault(
                    field=("rollout", "steps"), old_default=128, new_default=16
                ),
                ChangeDefault(
                    field=("rollout", "num_envs"), old_default=4, new_default=128
                ),
                ChangeDefault(
                    field=("rollout", "processes"), old_default=1, new_default=4
                ),
                ChangeDefault(
                    field=("vf_net", "n_layer"), old_default=1, new_default=2
                ),
                ChangeDefault(
                    field=("vf_net", "d_model"), old_default=64, new_default=32
                ),
            ],
            2: [
                ChangeDefault(
                    field=("total_timesteps",), old_default=25000, new_default=1000000
                ),
            ],
        }


if __name__ == "__main__":
    hyperstate.schema_evolution_cli(TrainConfig)
