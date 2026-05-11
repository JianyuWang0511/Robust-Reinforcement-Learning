import numpy as np
from typing import List


class K_Bandit_Env:
    def __init__(self, k: int):
        self.k = k
        self.variance_ls = np.ones(k)
        self.generate_new_means()

    def generate_new_means(self):
        self.mean_ls = np.random.normal(0, 1, self.k)
        best_choice = np.argmax(self.mean_ls)
        return best_choice

    def interact(self, action: int):
        if not (0 <= action and action < self.k):
            print(f"invalid action, action must be in [0, k], with k = {k} in the system")
            exit(0)
        else:
            reward = np.random.normal( self.mean_ls[action], self.variance_ls[action])
            return reward


class EpsilonPolicy:
    def __init__(self, k: int, epsilon: float):
        self.k = k
        self.epsilon = epsilon
        self.reset()

    def reset(self):
        self.estimated_value_ls = np.full(self.k, fill_value=5.0)
        self.num_trials_ls = np.zeros(self.k)

    def is_nongreedy_trial(self):
        return np.random.uniform(0, 1) <= self.epsilon

    def update_estimated_value(self, k: int, reward: float):
        self.estimated_value_ls[k] = (self.estimated_value_ls[k] * self.num_trials_ls[k] + reward) / (self.num_trials_ls[k] + 1)
        self.num_trials_ls[k] += 1

    def select_action(self):
        greedy_action = np.random.choice(np.flatnonzero(self.estimated_value_ls == self.estimated_value_ls.max()))
        action = greedy_action
        if self.is_nongreedy_trial():
            while action == greedy_action:
                action = np.random.randint(0, self.k, dtype=int)
        return action


class K_Bandit_Simulation:
    def __init__(self, policy_ls: List[EpsilonPolicy], env: K_Bandit_Env):
        self.policy_ls = policy_ls
        self.env = env

    def reset(self):
        best_choice = self.env.generate_new_means()
        for p in self.policy_ls:
            p.reset()
        return best_choice

    def one_simulation_run(self, T: int):
        best_choice = self.reset()
        action_history = [ [] for _ in range(len(self.policy_ls)) ]
        reward_history = [ [] for _ in range(len(self.policy_ls)) ]
        for t in range(T):
            for idx, policy in enumerate(self.policy_ls):
                action = policy.select_action()
                reward = self.env.interact(action)
                policy.update_estimated_value(action, reward)
                action_history[idx].append(action)
                reward_history[idx].append(reward)
        
        return best_choice, action_history, reward_history

        



    