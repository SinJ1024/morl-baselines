import heapq
import os
from dataclasses import dataclass
from typing import List, Optional, Type, Union, Dict, Callable

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import wandb

from morl_baselines.common.evaluation import log_all_multi_policy_metrics, _normalized_hypervolume
from morl_baselines.multi_policy.gcn.hyperparam_scheduler import HyperparamScheduler
from morl_baselines.multi_policy.gcn.gcn_model_classes import BaseGCNModel, DefaultGCNModel
from morl_baselines.multi_policy.gcn.helpers import crowding_distance
from morl_baselines.common.morl_algorithm import MOAgent, MOPolicy
from morl_baselines.common.performance_indicators import hypervolume

@dataclass
class Transition:
    """Transition dataclass."""

    observation: np.ndarray
    action: Union[float, int]
    action_mask: np.ndarray
    reward: np.ndarray
    next_observation: np.ndarray
    terminal: bool
    cell_index: int = -1

class GCN(MOAgent, MOPolicy):

    def __init__(
        self,
        env: Optional[gym.Env],
        scaling_factor: np.ndarray,
        learning_rate: float = 1e-3,
        gamma: float = 1.0,
        batch_size: int = 32,
        nr_layers: int = 1,
        hidden_dim: int = 64,
        noise: float = 0.1,
        project_name: str = "MORL-Baselines",
        experiment_name: str = "GCN",
        wandb_entity: Optional[str] = None,
        log: bool = True,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        model_class: Optional[Type[BaseGCNModel]] = None,
        dominance_func: Optional[Callable] = None,
        l2_func: Optional[Callable] = None,
        l2_params: Optional[Dict] = None,
        hyperparam_scheduler: HyperparamScheduler = None,
        cd_threshold: float = 0.2
    ) -> None:
        """Initialize GCN agent.

        Args:
            env (Optional[gym.Env]): Gym environment.
            scaling_factor (np.ndarray): Scaling factor for the desired return and horizon used in the model.
            learning_rate (float, optional): Learning rate. Defaults to 1e-2.
            gamma (float, optional): Discount factor. Defaults to 1.0.
            batch_size (int, optional): Batch size. Defaults to 32.
            nr_layers (int, optional): Number of NN Linear layers. Defaults to 1.
            hidden_dim (int, optional): Hidden dimension. Defaults to 64.
            noise (float, optional): Standard deviation of the noise to add to the action in the continuous action case. Defaults to 0.1.
            project_name (str, optional): Name of the project for wandb. Defaults to "MORL-Baselines".
            experiment_name (str, optional): Name of the experiment for wandb. Defaults to "GCN".
            wandb_entity (Optional[str], optional): Entity for wandb. Defaults to None.
            log (bool, optional): Whether to log to wandb. Defaults to True.
            seed (Optional[int], optional): Seed for reproducibility. Defaults to None.
            device (Union[th.device, str], optional): Device to use. Defaults to "auto".
            model_class (Optional[Type[BaseGCNModel]], optional): Model class to use. Defaults to None.
            dominance_func (Callable, optional): Function that provides a fairness ranking between strategies.
            l2_func (Callable, optional): Function that provides the distances for crowding calculation
            l2_params (Dict, optional): Paramaters for l2_func
            hyperparam_scheduler (HyperparamScheduler): A scheduler that may edit the model's hyperparameters during training. Currently only used for LCN's lambda
        """
        MOAgent.__init__(self, env, device=device, seed=seed)
        MOPolicy.__init__(self, device)

        self.experience_replay = []  # List of (distance, time_step, transition)
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.nr_layers = nr_layers
        self.hidden_dim = hidden_dim
        self.scaling_factor = scaling_factor
        self.desired_return = None
        self.desired_horizon = None
        self.noise = noise

        assert dominance_func is not None, "No fairness function defined for GCN cannot initialize model"
        self.dominance_func = dominance_func
        self.l2_func = l2_func
        self.l2_params = l2_params if l2_params is not None else {}
        self.hyperparam_scheduler = hyperparam_scheduler
        self.cd_threshold = cd_threshold
 
        if model_class and not issubclass(model_class, BaseGCNModel):
            raise ValueError("model_class must be a subclass of BaseGCNModel")

        if model_class is None:
            model_class = DefaultGCNModel

        self.model = model_class(
            self.observation_dim, self.action_dim, self.reward_dim, self.scaling_factor, hidden_dim=self.hidden_dim, nr_layers=self.nr_layers
        ).to(self.device)
        self.opt = th.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.log = log
        if log:
            self.setup_wandb(project_name, experiment_name, wandb_entity)

    def get_config(self) -> dict:
        """Get configuration of GCN model."""
        return {
            "env_id": self.env.unwrapped.spec.id,
            "reward_dim": self.reward_dim,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
            "hidden_dim": self.hidden_dim,
            "nr_layers": self.nr_layers,
            "scaling_factor": self.scaling_factor,
            "noise": self.noise,
            "seed": self.seed,
            "cd_threshold": self.cd_threshold,
    }

    def update(self):
        """Update GCN model."""
        batch = []

        # randomly choose episodes from experience buffer
        s_i = self.np_random.choice(np.arange(len(self.experience_replay)), size=self.batch_size, replace=True)

        for i in s_i:
            # episode is tuple (return, transitions)
            ep = self.experience_replay[i][2]
            # choose random timestep from episode,
            # use it's return and leftover timesteps as desired return and horizon
            t = self.np_random.integers(0, len(ep))
            # reward contains return until end of episode
            s_t, a_t, r_t, h_t, am_t = ep[t].observation, ep[t].action, np.float32(ep[t].reward), np.float32(len(ep) - t), ep[t].action_mask
            batch.append((s_t, a_t, r_t, h_t, am_t))

        obs, actions, desired_return, desired_horizon, _ = zip(*batch)
        probs = self.model(
            th.tensor(obs).to(self.device),
            th.tensor(desired_return).to(self.device),
            th.tensor(desired_horizon).unsqueeze(1).to(self.device),
        )
        prediction = nn.functional.log_softmax(probs, dim=-1)

        self.opt.zero_grad()
        # one-hot of action for CE loss
        actions = F.one_hot(th.tensor(actions).long().to(self.device), len(prediction[0]))
        # cross-entropy loss
        l = th.sum(-actions * prediction, -1)
        l = l.mean()
        l.backward()
        self.opt.step()

        return l, prediction

    def _add_episode(self, transitions: List[Transition], max_size: int, step: int) -> None:
        # compute return
        for i in reversed(range(len(transitions) - 1)):
            transitions[i].reward += self.gamma * transitions[i + 1].reward
        # pop smallest episode of heap if full, add new episode
        # heap is sorted by negative distance, (updated in nlargest)
        # put positive number to ensure that new item stays in the heap
        if len(self.experience_replay) == max_size:
            heapq.heappushpop(self.experience_replay, (1, step, transitions))
        else:
            heapq.heappush(self.experience_replay, (1, step, transitions))

    def _compute_route_contexts(self):
        """Compute normalized demand context for each episode in the ER buffer.
 
        Returns: array of shape (len(experience_replay),) with values in [0, 1].
        """
        if self.l2_params.get('demand_context') is None:
            return None

        contexts = []
        for ep in self.experience_replay:
            transitions = ep[2]
            cell_indices = [t.cell_index for t in transitions if t.cell_index >= 0]
            if cell_indices:
                contexts.append(np.mean(self.l2_params['demand_context'][cell_indices]))
            else:
                contexts.append(0.0)
        return np.array(contexts)


    def _nlargest(self, n):
        returns = np.array([e[2][0].reward for e in self.experience_replay])
        # crowding distance of each point, check ones that are too close together
        distances = crowding_distance(returns)
        sma = np.argwhere(distances <= self.cd_threshold).flatten()

        self.l2_params['route_contexts'] = self._compute_route_contexts()
        l2 = self.l2_func( returns, sma, self.l2_params )

        sorted_i = np.argsort(l2)
        largest = [self.experience_replay[i] for i in sorted_i[-n:]]
        # before returning largest elements, update all distances in heap
        for i in range(len(l2)):
            self.experience_replay[i] = (l2[i], self.experience_replay[i][1], self.experience_replay[i][2])
        heapq.heapify(self.experience_replay)
        return largest

    def _choose_commands(self, num_episodes: int):
        # get best episodes, according to their crowding distance
        episodes = self._nlargest(num_episodes)
        returns, horizons = list(zip(*[(e[2][0].reward, len(e[2])) for e in episodes]))
        # keep only non-dominated returns

        #### New
        nd_i, returns = self.dominance_func(np.array(returns), self.l2_params)
        ####

        horizons = np.array(horizons)[nd_i]
        # pick random return from random best episode
        r_i = self.np_random.integers(0, len(returns))
        desired_horizon = np.float32(horizons[r_i] - 2)
        # mean and std per objective
        _, s = np.mean(returns, axis=0), np.std(returns, axis=0)
        # desired return is sampled from [M, M+S], to try to do better than mean return
        desired_return = returns[r_i].copy()
        # random objective
        r_i = self.np_random.integers(0, len(desired_return))
        desired_return[r_i] += self.np_random.uniform(high=s[r_i])
        desired_return = np.float32(desired_return)
        return desired_return, desired_horizon

    def _act(self, obs: np.ndarray, desired_return, desired_horizon, action_mask, eval_mode=False) -> int:
        probs = self.model(
            th.tensor([obs]).float().to(self.device),
            th.tensor([desired_return]).float().to(self.device),
            th.tensor([desired_horizon]).unsqueeze(1).float().to(self.device),
        )
        # probs = probs.detach().cpu().numpy()[0]
        probs = probs.detach()

        # Apply the mask before log_softmax -- we add a large large number to the unmasked actions (Linear can return negative values)
        prediction = nn.functional.log_softmax(probs.cpu() + action_mask * 10000, dim=-1)
 
        log_probs = prediction.detach().cpu().numpy()[0]

        if eval_mode:
            action = np.argmax(log_probs)
        else:
            action = self.np_random.choice(np.arange(len(log_probs)), p=np.exp(log_probs))
        return action

    def _run_episode(self, env, desired_return, desired_horizon, max_return, starting_loc=None, eval_mode=False):
        transitions = []
        state, info = env.reset(options={'loc':starting_loc})
        states = [info['location_grid_coordinates']]
        obs = state
        done = False
        while not done:
            action = self._act(obs, desired_return, desired_horizon, info['action_mask'], eval_mode=eval_mode)
            n_state, reward, terminated, truncated, info = env.step(action)
            states.append(info['location_grid_coordinates'])
            n_obs = n_state
            done = terminated or truncated

            cell_idx = info.get('location_grid_index', -1)
            transitions.append(
                Transition(
                    observation=obs,
                    action=action,
                    action_mask=info['action_mask'],
                    reward=np.float32(reward).copy(),
                    next_observation=n_obs,
                    terminal=terminated,
                    cell_index=cell_idx,
                )
            )

            obs = n_obs
            # clip desired return, to return-upper-bound,
            # to avoid negative returns giving impossible desired returns
            desired_return = np.clip(desired_return - reward, None, max_return, dtype=np.float32)
            # clip desired horizon to avoid negative horizons
            desired_horizon = np.float32(max(desired_horizon - 1, 1.0))
        return transitions, states

    def set_desired_return_and_horizon(self, desired_return: np.ndarray, desired_horizon: int):
        """Set desired return and horizon for evaluation."""
        self.desired_return = desired_return
        self.desired_horizon = desired_horizon

    def eval(self, obs, w=None):
        """Evaluate policy action for a given observation."""
        return self._act(obs, self.desired_return, self.desired_horizon, eval_mode=True)

    def evaluate(self, env, max_return, n=10, starting_loc=None):
        """Evaluate policy in the given environment."""
        n = min(n, len(self.experience_replay))
        episodes = self._nlargest(n)
        returns, horizons = list(zip(*[(e[2][0].reward, len(e[2])) for e in episodes]))
        returns = np.float32(returns)
        horizons = np.float32(horizons)
        e_returns = []
        e_states = []
        e_cell_satisfaction = []
        city = env.unwrapped.city if hasattr(env.unwrapped, 'city') else None
        for i in range(n):
            transitions, states = self._run_episode(env, returns[i], np.float32(horizons[i]), max_return, starting_loc=starting_loc, eval_mode=True)
            # compute return
            for j in reversed(range(len(transitions) - 1)):
                transitions[j].reward += self.gamma * transitions[j + 1].reward
            e_returns.append(transitions[0].reward)
            e_states.append(states)

            if city is not None:
                line_indices = city.grid_to_index(np.array(states))
                sat_rates, _ = city.compute_cell_satisfaction(line_indices)
                e_cell_satisfaction.append(sat_rates)

        distances = np.linalg.norm(np.array(returns) - np.array(e_returns), axis=-1)
        return np.array(e_returns), np.array(returns), distances, e_states, e_cell_satisfaction

    def save(self, filename: str = "GCN_model", savedir: str = "weights"):
        """Save GCN."""
        if not os.path.isdir(savedir):
            os.makedirs(savedir)
        th.save(self.model, f"{savedir}/{filename}.pt")

    def train(
        self,
        total_timesteps: int,
        eval_env: gym.Env,
        ref_point: np.ndarray,
        known_pareto_front: Optional[List[np.ndarray]] = None,
        num_eval_weights_for_eval: int = 50,
        num_er_episodes: int = 500,
        num_step_episodes: int = 10,
        num_model_updates: int = 100,
        max_return: np.ndarray = 250.0,
        max_buffer_size: int = 500,
        num_points_pf: int = 100,
        starting_loc: Optional[np.ndarray] = None,
        nr_stations: int = 9,
        save_dir: str = "weights",
        pf_plot_limits: Optional[List[int]] = [0, 0.5],
        n_policies: int = 10
    ):
        """Train GCN.

        Args:
            total_timesteps: total number of time steps to train for
            eval_env: environment for evaluation
            ref_point: reference point for hypervolume calculation
            known_pareto_front: Optimal pareto front for metrics calculation, if known.
            num_eval_weights_for_eval (int): Number of weights use when evaluating the Pareto front, e.g., for computing expected utility.
            num_er_episodes: number of episodes to fill experience replay buffer.
            num_step_episodes: number of steps per episode
            num_model_updates: number of model updates per episode
            max_return: maximum return for clipping desired return
            max_buffer_size: maximum buffer size
            num_points_pf: int = 100,
            starting_loc: starting location for episodes, if None, random location is used
            save_dir: directory to save model weights
            pf_plot_limits: limits for the pareto front plot (only for 2 objectives)
            n_policies: number of policies to evaluate at each checkpoint
            cd_threshold: threshold for crowding distance
        """
        max_return = max_return if max_return is not None else np.full(self.reward_dim, 100.0, dtype=np.float32)
        if self.log:
            self.register_additional_config(
                {
                    "total_timesteps": total_timesteps,
                    "ref_point": ref_point.tolist(),
                    "known_front": known_pareto_front,
                    "num_eval_weights_for_eval": num_eval_weights_for_eval,
                    "num_er_episodes": num_er_episodes,
                    "num_step_episodes": num_step_episodes,
                    "num_model_updates": num_model_updates,
                    "starting_loc": starting_loc,
                    "max_buffer_size": max_buffer_size,
                    "num_policies": n_policies,
                    "save_dir": save_dir,
                    "nr_stations": nr_stations,
                }
            )
            if self.hyperparam_scheduler is not None:
                self.register_additional_config(
                    self.hyperparam_scheduler.get_config()
                    #"spatial_alpha": self.l2_params['spatial_alpha']
                )
        self.global_step = 0

        if self.hyperparam_scheduler is not None:
            self.hyperparam_scheduler.step(0, self.l2_params)

        if hasattr(self.env.unwrapped, 'city'):
            agg_od = self.env.unwrapped.city.agg_od_mx().flatten()
            max_od = agg_od.max()
            self.l2_params['demand_context'] = agg_od / max_od if max_od > 0 else agg_od
        total_episodes = num_er_episodes
        n_checkpoints = 0

        # fill buffer with random episodes
        self.experience_replay = []

        for _ in range(num_er_episodes):
            transitions = []
            obs, info = self.env.reset(options={'loc':starting_loc})
            done = False
            while not done:
                action = self.env.action_space.sample(mask=info['action_mask'])
                n_obs, reward, terminated, truncated, info = self.env.step(action)
                cell_idx = info.get('location_grid_index', -1)
                transitions.append(Transition(obs, action, info['action_mask'], np.float32(reward).copy(), n_obs, terminated, cell_index=cell_idx))
                done = terminated or truncated
                obs = n_obs
                self.global_step += 1
            # add episode in-place
            self._add_episode(transitions, max_size=max_buffer_size, step=self.global_step)

        returns = None
        while self.global_step < total_timesteps:
            if self.hyperparam_scheduler is not None:
                self.hyperparam_scheduler.step(self.global_step, self.l2_params)
                if self.log:
                    key = self.hyperparam_scheduler.target_key
                    wandb.log({f"train/{key}": self.l2_params[key], "global_step": self.global_step}, commit=False)

            loss = []
            entropy = []
            for _ in range(num_model_updates):
                l, lp = self.update()
                loss.append(l.detach().cpu().numpy())
                lp = lp.detach().cpu().numpy()
                ent = np.sum(-np.exp(lp) * lp)
                entropy.append(ent)

            desired_return, desired_horizon = self._choose_commands(num_er_episodes)

            # get all leaves, contain biggest elements, experience_replay got heapified in choose_commands
            leaves_r = np.array([e[2][0].reward for e in self.experience_replay[len(self.experience_replay) // 2 :]])
            # leaves_h = np.array([len(e[2]) for e in self.experience_replay[len(self.experience_replay) // 2 :]])

            if self.log:
                hv, _ = _normalized_hypervolume(list(leaves_r), self.reward_dim)
                hv_est = hv
                wandb.log(
                    {
                        "train/hypervolume": hv_est,
                        "train/loss": np.mean(loss),
                        "global_step": self.global_step,
                        "train/entropy": np.mean(entropy),
                    },
                )

            returns = []
            horizons = []
            for _ in range(num_step_episodes):
                transitions, _ = self._run_episode(self.env, desired_return, desired_horizon, max_return, starting_loc=starting_loc)
                self.global_step += len(transitions)
                self._add_episode(transitions, max_size=max_buffer_size, step=self.global_step)
                returns.append(transitions[0].reward)
                horizons.append(len(transitions))

            total_episodes += num_step_episodes
            if self.log:
                wandb.log(
                    {
                        "train/episode": total_episodes,
                        "train/horizon_desired": desired_horizon,
                        "train/mean_horizon_distance": np.linalg.norm(np.mean(horizons) - desired_horizon),
                        "global_step": self.global_step,
                    },
                )

                for i in range(self.reward_dim):
                    wandb.log(
                        {
                            f"train/desired_return_{i}": desired_return[i],
                            f"train/mean_return_{i}": np.mean(np.array(returns)[:, i]),
                            f"train/mean_return_distance_{i}": np.linalg.norm(
                                np.mean(np.array(returns)[:, i]) - desired_return[i]
                            ),
                            "global_step": self.global_step,
                        },
                    )
            print(
                f"step {self.global_step} \t return {np.mean(returns, axis=0)}, ({np.std(returns, axis=0)}) \t loss {np.mean(loss):.3E} \t horizons {np.mean(horizons)}"
            )

            if self.global_step >= (n_checkpoints + 1) * total_timesteps / int(os.environ.get("GCN_N_EVALS", "30")):
                self.save(savedir=save_dir, filename=f"GCN_model_{n_checkpoints}")
                n_checkpoints += 1
                e_returns, returns, _, e_states, e_cell_satisfaction = self.evaluate(eval_env, max_return, n=int(os.environ.get("GCN_N_POINTS_PF", "20")), starting_loc=starting_loc)

                if self.log:
                    city = eval_env.unwrapped.city if hasattr(eval_env.unwrapped, 'city') else None
                    cell_sat = np.array(e_cell_satisfaction) if e_cell_satisfaction else None
                    cell_dem = None
                    agg_od = None
                    house_prices = None
                    if city is not None and cell_sat is not None and len(cell_sat) > 0:
                        cell_dem = np.sum(city.od_mx, axis=1) + np.sum(city.od_mx, axis=0)
                        agg_od = city.agg_od_mx().flatten()
                        house_prices = getattr(city, 'house_prices', None)

                    log_all_multi_policy_metrics(
                        current_front=e_returns,
                        hv_ref_point=ref_point,
                        reward_dim=self.reward_dim,
                        global_step=self.global_step,
                        n_sample_weights=num_eval_weights_for_eval,
                        ref_front=known_pareto_front,
                        cell_satisfaction_rates=cell_sat,
                        cell_demands=cell_dem,
                        agg_od_by_cell=agg_od,
                        house_prices=house_prices,
                    )
        self.env.close()
