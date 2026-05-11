from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Type

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


@dataclass
class ExperimentConfig:
    env_id: str = "CartPole-v1"
    algorithm: str = "DQN"  # Options: DQN, PPO, A2C
    total_timesteps: int = 30_000
    learning_rate: float = 1e-3
    seed: int = 42
    log_dir: Path = field(default_factory=lambda: Path("runs/cartpole_demo"))
    model_dir: Path = field(default_factory=lambda: Path("models"))
    video_dir: Path = field(default_factory=lambda: Path("videos"))
    model_name: str = "cartpole_model"
    eval_episodes: int = 5
    video_episode_trigger: int = 0

    def ensure_directories(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model_path(self) -> Path:
        algo_name = self.algorithm.lower()
        return self.model_dir / f"{algo_name}_{self.model_name}.zip"


class EnvironmentFactory:
    """Owns Gymnasium environment creation and wrapper composition."""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def make_training_env(self) -> gym.Env:
        env = gym.make(self.config.env_id)
        env = RecordEpisodeStatistics(env)
        env = Monitor(env)
        env.reset(seed=self.config.seed)
        return env

    def make_evaluation_env(self, record_video: bool = False) -> gym.Env:
        render_mode = "rgb_array" if record_video else None
        env = gym.make(self.config.env_id, render_mode=render_mode)
        env = RecordEpisodeStatistics(env)
        env = Monitor(env)

        if record_video:
            env = RecordVideo(
                env,
                video_folder=str(self.config.video_dir),
                episode_trigger=lambda episode_id: episode_id >= self.config.video_episode_trigger,
                name_prefix=f"{self.config.algorithm.lower()}-{self.config.env_id}",
            )

        env.reset(seed=self.config.seed)
        return env


class MetricsCallback(BaseCallback):
    """Shows how Stable-Baselines3 callbacks can inject custom logging."""

    def __init__(self, log_every_steps: int = 1_000, verbose: int = 0):
        super().__init__(verbose)
        self.log_every_steps = log_every_steps

    def _on_step(self) -> bool:
        if self.n_calls % self.log_every_steps == 0:
            self.logger.record("custom/steps_seen", self.num_timesteps)
        return True


class ModelFactory:
    """Maps a friendly algorithm name to an SB3 model class."""

    ALGORITHMS: dict[str, Type[BaseAlgorithm]] = {
        "DQN": DQN,
        "PPO": PPO,
        "A2C": A2C,
    }

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def create(self, env: gym.Env) -> BaseAlgorithm:
        algorithm_cls = self.ALGORITHMS.get(self.config.algorithm.upper())
        if algorithm_cls is None:
            available = ", ".join(self.ALGORITHMS.keys())
            raise ValueError(f"Unknown algorithm '{self.config.algorithm}'. Choose one of: {available}")

        common_kwargs = dict(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            learning_rate=self.config.learning_rate,
            tensorboard_log=str(self.config.log_dir),
            seed=self.config.seed,
        )

        if algorithm_cls is DQN:
            return algorithm_cls(
                **common_kwargs,
                buffer_size=50_000,
                learning_starts=1_000,
                batch_size=64,
                train_freq=4,
                target_update_interval=250,
            )

        return algorithm_cls(**common_kwargs)

    def load(self, env: gym.Env) -> BaseAlgorithm:
        algorithm_cls = self.ALGORITHMS.get(self.config.algorithm.upper())
        if algorithm_cls is None:
            available = ", ".join(self.ALGORITHMS.keys())
            raise ValueError(f"Unknown algorithm '{self.config.algorithm}'. Choose one of: {available}")
        return algorithm_cls.load(str(self.config.model_path), env=env)


class Trainer:
    """Owns the training lifecycle for an SB3 model."""

    def __init__(self, config: ExperimentConfig, env_factory: EnvironmentFactory, model_factory: ModelFactory):
        self.config = config
        self.env_factory = env_factory
        self.model_factory = model_factory

    def train(self) -> BaseAlgorithm:
        train_env = self.env_factory.make_training_env()
        model = self.model_factory.create(train_env)
        callbacks = CallbackList([MetricsCallback()])

        model.learn(
            total_timesteps=self.config.total_timesteps,
            callback=callbacks,
            tb_log_name=self.config.algorithm.lower(),
            progress_bar=True,
        )

        model.save(str(self.config.model_path))
        train_env.close()
        return model


class Evaluator:
    """Runs deterministic evaluation and optional video capture."""

    def __init__(self, config: ExperimentConfig, env_factory: EnvironmentFactory, model_factory: ModelFactory):
        self.config = config
        self.env_factory = env_factory
        self.model_factory = model_factory

    def evaluate(self, model: Optional[BaseAlgorithm] = None, record_video: bool = True) -> Tuple[float, float]:
        eval_env = self.env_factory.make_evaluation_env(record_video=record_video)

        if model is None:
            model = self.model_factory.load(eval_env)

        mean_reward, std_reward = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=self.config.eval_episodes,
            deterministic=True,
        )

        print(f"Evaluation over {self.config.eval_episodes} episodes -> mean_reward={mean_reward:.2f}, std_reward={std_reward:.2f}")
        eval_env.close()
        return mean_reward, std_reward


class CartPolePipeline:
    """High-level orchestration layer connecting config, envs, model, training, and evaluation."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.config.ensure_directories()
        self.env_factory = EnvironmentFactory(config)
        self.model_factory = ModelFactory(config)
        self.trainer = Trainer(config, self.env_factory, self.model_factory)
        self.evaluator = Evaluator(config, self.env_factory, self.model_factory)

    def run(self) -> None:
        model = self.trainer.train()
        self.evaluator.evaluate(model=model, record_video=True)
        self.print_next_steps()

    def print_next_steps(self) -> None:
        print("\nNext steps:")
        print(f"- Launch TensorBoard: tensorboard --logdir {self.config.log_dir}")
        print(f"- Saved model: {self.config.model_path}")
        print(f"- Saved videos: {self.config.video_dir.resolve()}")
        print("- Try changing algorithm to PPO or A2C to compare behavior.")


if __name__ == "__main__":
    config = ExperimentConfig(
        env_id="CartPole-v1",
        algorithm="DQN",
        total_timesteps=25_000,
        learning_rate=1e-3,
        eval_episodes=3,
    )

    pipeline = CartPolePipeline(config)
    pipeline.run()
