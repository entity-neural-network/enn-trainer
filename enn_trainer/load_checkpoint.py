from hyperstate import StateManager

from .agent import RogueNetAgent
from .config import TrainConfig
from .train import State, init_train_state


def load_checkpoint(path: str) -> StateManager[TrainConfig, State]:
    """
    Loads a training checkpoint from a given path.

    The returned StateManager has a ``config`` attribute that contains the
    configuration of the training run and a ``state`` attribute that contains
    the state of the training run, including the agent.
    """
    return StateManager(TrainConfig, State, init_train_state, init_path=path)


def load_agent(path: str) -> RogueNetAgent:
    """
    Loads a training checkpoint from a given path and returns the agent.
    """
    return RogueNetAgent(
        StateManager(TrainConfig, State, init_train_state, init_path=path).state.agent
    )
