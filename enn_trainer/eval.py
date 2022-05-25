import json
from typing import Callable, List, Mapping, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
import torch.distributed as dist
from entity_gym.env import *
from entity_gym.env.add_metrics_wrapper import AddMetricsWrapper
from entity_gym.env.vec_env import Metric
from entity_gym.serialization import SampleRecordingVecEnv
from entity_gym.simple_trace import Tracer
from torch.utils.tensorboard import SummaryWriter

from enn_trainer.agent import PPOAgent
from enn_trainer.config import EnvConfig, EvalConfig, RolloutConfig
from enn_trainer.rollout import Rollout


def run_eval(
    cfg: EvalConfig,
    env_cfg: EnvConfig,
    rollout: RolloutConfig,
    create_env: Callable[[EnvConfig, int, int, int], VecEnv],
    create_opponent: Callable[
        [str, ObsSpace, Mapping[str, ActionSpace], torch.device], PPOAgent
    ],
    agent: PPOAgent,
    device: torch.device,
    tracer: Tracer,
    writer: Optional[SummaryWriter],
    global_step: int,
    rank: int,
    parallelism: int,
) -> None:
    # TODO: metrics are biased towards short episodes
    processes = cfg.processes or rollout.processes
    num_envs = cfg.num_envs or rollout.num_envs

    envs: VecEnv = create_env(
        cfg.env or env_cfg,
        num_envs // parallelism,
        processes,
        rank * num_envs // parallelism,
    )
    obs_space = envs.obs_space()
    action_space = envs.action_space()

    metric_filter: Optional[npt.NDArray[np.bool8]] = None
    if cfg.opponent is not None:
        opponent = create_opponent(cfg.opponent, obs_space, action_space, device)
        if cfg.opponent_only:
            agents: Union[
                PPOAgent, List[Tuple[npt.NDArray[np.int64], PPOAgent]]
            ] = opponent
        else:
            agents = [
                (np.array([2 * i for i in range(num_envs // parallelism // 2)]), agent),
                (
                    np.array([2 * i + 1 for i in range(num_envs // parallelism // 2)]),
                    opponent,
                ),
            ]
            metric_filter = np.arange(num_envs // parallelism) % 2 == 0
    else:
        agents = agent

    envs = AddMetricsWrapper(envs, metric_filter)

    assert num_envs % parallelism == 0, (
        "Number of eval environments must be divisible by parallelism: "
        f"{num_envs} % {parallelism} = {num_envs % parallelism}"
    )

    if cfg.capture_samples:
        envs = SampleRecordingVecEnv(
            envs, cfg.capture_samples, cfg.capture_samples_subsample
        )
    eval_rollout = Rollout(
        envs,
        obs_space=obs_space,
        action_space=action_space,
        agent=agents,
        device=device,
        tracer=tracer,
    )
    _, _, metrics = eval_rollout.run(
        cfg.steps,
        record_samples=False,
        capture_videos=cfg.capture_videos,
        capture_logits=cfg.capture_logits,
    )
    for i in range(rank + 1):
        metrics[f"r{rank}m{i}"] = Metric(
            count=2, sum=10 * (rank + 1), min=0, max=10 * (rank + 1)
        )

    if parallelism > 1:
        serialized_metrics = json.dumps(
            {
                k: {
                    "count": int(v.count),
                    "sum": float(v.sum),
                    "min": float(v.min),
                    "max": float(v.max),
                }
                for k, v in metrics.items()
            }
        )
        metrics_tensor = torch.tensor(
            bytearray(serialized_metrics.encode("utf-8")), dtype=torch.uint8
        )
        metrics = {}
        for metrics_tensor in allgather(metrics_tensor, rank, parallelism):
            for k, v in json.loads(
                metrics_tensor.numpy().tobytes().decode("utf-8")
            ).items():
                m = Metric(**v)
                if k in metrics:
                    metrics[k] += m
                else:
                    metrics[k] = m
    if writer is not None:
        if cfg.capture_videos:
            # save the videos
            writer.add_video(
                f"eval/video",
                torch.tensor(eval_rollout.rendered).permute(1, 0, 4, 2, 3),
                global_step,
                fps=30,
            )

        for name, value in metrics.items():
            writer.add_scalar(f"eval/{name}.mean", value.mean, global_step)
            writer.add_scalar(f"eval/{name}.min", value.min, global_step)
            writer.add_scalar(f"eval/{name}.max", value.max, global_step)
            writer.add_scalar(f"eval/{name}.count", value.count, global_step)
    print(
        f"[eval] global_step={global_step} {'  '.join(f'{name}={value.mean}' for name, value in metrics.items())}"
    )
    envs.close()


def allgather(tensor: torch.Tensor, rank: int, parallelism: int) -> List[torch.Tensor]:
    sizes = [torch.zeros(1, dtype=torch.long) for _ in range(parallelism)]
    dist.all_gather(sizes, torch.tensor([tensor.nelement()], dtype=torch.long))
    alltensors = []
    for src, size in enumerate(sizes):
        if src == rank:
            dist.broadcast(tensor, src)
            alltensors.append(tensor)
        else:
            result = torch.zeros(int(size.item()), dtype=tensor.dtype)
            dist.broadcast(result, src)
            alltensors.append(result)
    return alltensors
