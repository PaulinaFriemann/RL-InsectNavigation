"""
Implementation of a gridworld MDP. This code is copied and adapted from
https://github.com/NrLabFreiburg/inverse-q-learning

which is itself copied and adapted from

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""
import functools
import numpy as np
import numpy.random as rn
import pkgutil
import sys
#sys.path.insert(0, "c:/users/paulina/documents/github/inverse-q-learning/")
#import iql
from insect_rl.mdp.utils import grid_math as gm
import iql.mdp.value_iteration as value_iteration
import insect_rl.mdp.mrl as mrl

#np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)


class Gridworld:
    """
    Gridworld MDP.
    """

    def __init__(self, width, height, reward, wind, discount, actions, transition_probability=None, **kwargs):
        """
        grid_size: Grid size. int.
        wind: Chance of moving randomly. float.
        discount: MDP discount. float.
        -> Gridworld
        """
        self.actions = actions
        self.n_actions = len(self.actions)
        self.width = width
        self.height = height
        self.reward_ = reward
        self.grid_size = width
        self.n_states = width * height
        self.wind = wind
        self.discount = discount

        if transition_probability is None:
            # Preconstruct the transition probability array.
            self.transition_probability = np.array(
                [[[gm.transition_probability(i, j, k, self.width, self.height, self.actions, wind=self.wind)
                for k in range(self.n_states)]
                for j in range(self.n_actions)]
                for i in range(self.n_states)])
        else:
            self.transition_probability = transition_probability

    def reward(self, state_int):
        """
        Reward for being in state state_int.

        state_int: State integer. int.
        -> Reward.
        """

        if state_int == self.n_states - 1:
            return self.reward_
        return 0

    def average_reward(self, n_trajectories, trajectory_length, policy):
        """
        Calculate the average total reward obtained by following a given policy
        over n_paths paths.

        policy: Map from state integers to action integers.
        n_trajectories: Number of trajectories. int.
        trajectory_length: Length of an episode. int.
        -> Average reward, standard deviation.
        """

        trajectories = self.generate_trajectories(n_trajectories,
                                             trajectory_length, policy)
        rewards = [[r for _, _, r in trajectory] for trajectory in trajectories]
        rewards = np.array(rewards)

        # Add up all the rewards to find the total reward.
        total_reward = rewards.sum(axis=1)

        # Return the average reward and standard deviation.
        return total_reward.mean(), total_reward.std()

    def optimal_policy(self, state_int):
        """
        The optimal policy for this gridworld.

        state_int: What state we are in. int.
        -> Action int.
        """

        sx, sy = self.int_to_point(state_int)

        if sx < self.grid_size and sy < self.grid_size:
            return rn.randint(0, 2)
        if sx < self.grid_size-1:
            return 0
        if sy < self.grid_size-1:
            return 1
        raise ValueError("Unexpected state.")

    def optimal_policy_deterministic(self, state_int):
        """
        Deterministic version of the optimal policy for this gridworld.

        state_int: What state we are in. int.
        -> Action int.
        """

        sx, sy = self.int_to_point(state_int)
        if sx < sy:
            return 0
        return 1

    def generate_trajectories(self, n_trajectories, trajectory_length, policy,
                                    random_start=False):
        """
        Generate n_trajectories trajectories with length trajectory_length,
        following the given policy.

        n_trajectories: Number of trajectories. int.
        trajectory_length: Length of an episode. int.
        policy: Map from state integers to action integers.
        random_start: Whether to start randomly (default False). bool.
        -> [[(state int, action int, reward float)]]
        """

        trajectories = []
        for _ in range(n_trajectories):
            if random_start:
                sx, sy = rn.randint(self.grid_size), rn.randint(self.grid_size)
            else:
                sx, sy = 0, 0

            trajectory = []
            for _ in range(trajectory_length):
                if rn.random() < self.wind:
                    action = self.actions[rn.randint(0, 4)]
                else:
                    # Follow the given policy.
                    action = self.actions[policy(self.point_to_int((sx, sy)))]

                if (0 <= sx + action[0] < self.grid_size and
                        0 <= sy + action[1] < self.grid_size):
                    next_sx = sx + action[0]
                    next_sy = sy + action[1]
                else:
                    next_sx = sx
                    next_sy = sy

                state_int = self.point_to_int((sx, sy))
                action_int = self.actions.index(action)
                next_state_int = self.point_to_int((next_sx, next_sy))
                reward = self.reward(next_state_int)
                trajectory.append((state_int, action_int, reward,next_state_int))

                sx = next_sx
                sy = next_sy

            trajectories.append(trajectory)

        return np.array(trajectories)
    
    def find_policy(self, ground_r):
        return value_iteration.find_policy(self.n_states, self.n_actions, self.transition_probability,
                           ground_r, self.discount, stochastic=False)

    def collect_demonstrations(self, n_trajectories, trajectory_length, random_start=False):
      # 1 for goal (last) state, 0 else 
        ground_r = np.array([self.reward(s) for s in range(self.n_states)])

        policy = self.find_policy(ground_r)

        trajectories = self.generate_trajectories(n_trajectories,
                                            trajectory_length,
                                            lambda s: policy[s],
                                            random_start=random_start)
        action_probabilities = gm.get_action_probabilities(trajectories, self.n_states, self.n_actions)

        return trajectories, action_probabilities, self.transition_probability, ground_r

    def __str__(self):
        return f"Gridworld({self.grid_size}, {self.wind}, {self.discount})"


class CustomGridworld(Gridworld):
    def __init__(self, width, height, goals, reward, start, traps, trapcost, wind, discount, actions, transition_probability=None, **kwargs) -> None:
        self.goals = goals
        self.start = start
        self.traps = traps
        self.trapcost = trapcost

        super().__init__(width, height, reward, wind, discount, list(actions), transition_probability=transition_probability)

    @functools.cached_property
    def reward_matrix(self):
        mat = np.zeros((self.width, self.height))
        # x ->, y down
        for goal in self.goals:
            mat[goal] = self.reward_
        for t in self.traps:
            mat[t] = self.trapcost
        return mat

    def reward(self, state_int):
        """
        Reward for being in state state_int.

        state_int: State integer. int.
        -> Reward.
        """
        state = gm.int_to_point(state_int, self.width)
        return int(self.reward_matrix[state])
    
    @property
    def ground_r(self):
        return np.array([self.reward(s) for s in range(self.n_states)])
    
    def optimal_policy(self, state_int):
        raise NotImplementedError
    
    def optimal_policy_deterministic(self, state_int):
        raise NotImplementedError
    
    def convert_trajectories(self, trajectories):
        # has to be [[(state int, action int, reward float, next state int)]]
        print("convert trajectories")
        converted = []
        for trajectory in trajectories:
            converted.insert(-1, [])
            for s1, s2 in zip(trajectory[:-1], trajectory[1:]):
                converted[-1].append([gm.point_to_int(s1, self.width),
                                      self.actions.index(tuple(np.array(s2) - np.array(s1))),
                                      0,#int(self.reward_matrix[s2]),
                                      gm.point_to_int(s2, self.width)])
            converted[-1][-1][2] = self.reward_
        return converted

    def generate_trajectories(self, n_trajectories, policy,
                                    random_start=False):
        """
        Generate n_trajectories trajectories, each until the goal is reached,
        following the given policy.

        n_trajectories: Number of trajectories. int.
        policy: Map from state integers to action integers.
        random_start: Whether to start randomly (default False). bool.
        -> [[(state int, action int, reward float, next state int)]]
        """

        trajectories = []
        for _ in range(n_trajectories):
            if random_start:
                sx, sy = rn.randint(self.width), rn.randint(self.height)
            else:
                sx, sy = self.start

            trajectory = []
            while (sx, sy) not in self.goals:
                if rn.random() < self.wind:
                    action = self.actions[rn.randint(0, len(self.actions))]
                else:
                    # Follow the given policy.
                    action = self.actions[policy(gm.point_to_int((sx, sy), self.width))]

                if (0 <= sx + action[0] < self.width and
                        0 <= sy + action[1] < self.height):
                    next_sx = sx + action[0]
                    next_sy = sy + action[1]
                else:
                    next_sx = sx
                    next_sy = sy

                state_int = gm.point_to_int((sx, sy), self.width)
                action_int = self.actions.index(action)
                next_state_int = gm.point_to_int((next_sx, next_sy), self.width)
                reward = self.reward(next_state_int)
                trajectory.append((state_int, action_int, reward,next_state_int))

                sx = next_sx
                sy = next_sy

            trajectories.append(trajectory)
        #try:
        return np.array(trajectories)
        #except np.VisibleDeprecationWarning:
        #    print(trajectories)
        #    raise
    
    def get_action_probabilities(self, trajectories):
        print("calculate action probabilities")
        return gm.get_action_probabilities(trajectories, self.n_states, self.n_actions)

    def collect_demonstrations(self, n_trajectories, random_start=False):
      # 1 for goal (last) state, 0 else 
        print("Calculating policy")
        policy = self.find_policy(self.ground_r)
        print("Generating trajectories")
        # TODO trajectories might pose a problem with nparray if of differing lens
        trajectories = self.generate_trajectories(n_trajectories,
                                            lambda s: policy[s],
                                            random_start=random_start)
        print("Calculating action probabilities")
        action_probabilities = gm.get_action_probabilities(trajectories)

        return trajectories, action_probabilities, self.transition_probability, self.ground_r


class LearningGridworld(CustomGridworld):
    def __init__(self, width, height, goal, reward, start, traps, trapcost, wind, discount, actions, transition_probability=None, **kwargs):
        super().__init__(width, height, goal, reward, start, traps, trapcost, wind, discount, actions, **kwargs)
        self.mrl_gw = mrl.MyGridWorld(self.width, self.height, self.goal, self.reward_, self.start, self.actions, self.traps, self.trapcost, transition_probability=transition_probability)

    def collect_demonstrations(self, agent, n_trajectories, random_start=False):
        return self.mrl_gw.collect_demonstrations(agent, n_trajectories, wind=self.wind)
