"""
Uses https://github.com/PaulinaFriemann/inverse-q-learning, adapted from https://github.com/NrLabFreiburg/inverse-q-learning

install as iql
"""

from insect_rl.mdp.utils import grid_math
import matplotlib.pyplot as plt

from iql.algorithms.iavi import inverse_action_value_iteration
import iql.plot_experiments as plt_exp
from insect_rl.plot import plot_grids

from icecream import ic

class IAVIEvaluation:
    def __init__(self, width: int, height: int, start: tuple[int, int]) -> None:
        self.width = width
        self.height = height
        self.start = start

    def __call__(self, action_probabilities, transition_probabilities, gamma=0.99, feature_map="ident"):
        n_actions = action_probabilities.shape[1]
        gamma=0.99

        # q: value function, r: reward function, boltz: boltzmann distribution
        # q: (n_S, n_A) r: (n_S, n_A), boltz: (n_S, n_A)
        feature_matrix = grid_math.feature_matrix(self.width, self.height, feature_map=feature_map)
        q, r, boltz = inverse_action_value_iteration(feature_matrix, n_actions, gamma, transition_probabilities, action_probabilities, theta=0.01)
        return {"q": q,
                "r": r,
                "boltz": boltz}

        #p_start_state = (np.bincount(trajectories[:, 0, 0], minlength=env.n_states)/trajectories.shape[0])
        nS = self.width * self.height

        # V(s) <- sum[Au: T(s,a,u) * p(s,a) * (r(s) + discount * V(u)) ]

        # results of boltzman policy inferred via iavi and the ground truth reward
        v_iavi = plt_exp.policy_eval(boltz, ground_r, transition_probabilities, nS, n_actions, discount_factor=0.9, theta=0.001)

        # results of the action probabilities taken from the trajectories
        v_true = plt_exp.policy_eval(action_probabilities, ground_r, transition_probabilities, nS, n_actions, discount_factor=0.9, theta=0.001) 

        # found via value iteration
        b = plt_exp.find_policy(nS, n_actions, transition_probabilities, ground_r, discount=0.9, threshold=1e-2)
        v_valueit = plt_exp.policy_eval(b, ground_r, transition_probabilities, nS, n_actions, discount_factor=0.9, theta=0.001)
        
        return {"q": q,
                "r": r,
                "boltz": boltz,
                "v_iavi":v_iavi,
                "v_true":v_true,
                "v_valueit":v_valueit}
