import copy
from mushroom_rl.core.environment import Environment

import numpy as np

from mushroom_rl.environments.grid_world import AbstractGridWorld
from mushroom_rl.policy.td_policy import Boltzmann
from mushroom_rl.utils import spaces
from mushroom_rl.core import MDPInfo
from mushroom_rl.utils.viewer import Viewer
from mushroom_rl.core import Core
from mushroom_rl.utils.callbacks.callback import Callback
from mushroom_rl.utils.table import Table

import insect_rl.mdp.utils.grid_math as gm
from insect_rl.mdp.utils.grid_math import Action, INTERCARDINALS
from icecream import ic

class ActionSpace(spaces.Discrete):
    def __init__(self, actions):
        self.actions = actions
        super().__init__(len(actions))

    def action_step(self, action: int) -> np.array:
        return np.array(self.actions[action])

    def inverse(self, action: int) -> int:
        inv = tuple((-1) * dim for dim in self.action_step(action))
        return self.actions.index(inv)



class MyGridWorld(AbstractGridWorld):

    def __init__(self, width: int, height : int, goal : tuple, reward: float, start : tuple, actions: tuple[Action]=INTERCARDINALS, traps=None, trap_exits=None, trap_cost=-10, gamma=.9, r=None, **kwargs):
        # MDP properties
        observation_space = spaces.Discrete(height * width)
        self.action_space = ActionSpace(actions)
        self.n_states = observation_space.n
        self.n_actions = self.action_space.n
        horizon = np.inf
        mdp_info = MDPInfo(observation_space, self.action_space, gamma, horizon)

        if traps is None:
            traps = []
        if trap_exits is None:
            trap_exits = []
        self.hidden_traps = list(map(np.array, traps))
        self._traps = []
        self._trap_exits = [tuple(map(np.array, t_exit)) for t_exit in trap_exits]

        self.trap_cost = trap_cost
        self.reward_ = reward

        self.last_state = None

        super().__init__(mdp_info, height=height, width=width, goal=goal, start=start)

        #self._viewer._background = (250, 250, 250)
        #self._viewer.screen
        #self._viewer.display(.1)
        self.r = r

    
    def open_trap(self):
        self._traps = self.hidden_traps
    
    def close_trap(self):
        self._traps = []

    def render(self):
        def center(state):
            return [self._height - (.5 + state[1]), .5 + state[0]]

        for row in range(1, self._height):
            for col in range(1, self._width):
                self._viewer.line(np.array([col, 0]),
                                  np.array([col, self._height]),
                                  color=(175, 175, 175))
                self._viewer.line(np.array([0, row]),
                                  np.array([self._width, row]),
                                  color=(175, 175, 175))
        for trap_state in self._traps:
            trap_center = np.array(center(trap_state))
            self._viewer.square(trap_center, 0, 1, (50, 50, 50))

        goal_center = np.array(center(self._goal))
        self._viewer.square(goal_center, 0, 1, (0, 255, 0))

        start_center = np.array(center(self._start))
        self._viewer.square(start_center, 0, 1, (255, 100, 0))     
        
        state_grid = self.convert_to_grid(self._state, self._width)
        state_center = np.array(center(state_grid))
        
        self._viewer.circle(state_center, .4, (0, 0, 255))
        
        self._viewer.display(.1)
    
    def collect_demonstrations(self, agent, n_trajectories: int, wind: float=0.0):
        """
        Method to match the Inverse Q-Learning GridWorld method.

        """
        #* IAVI converter is just a callback function to make the data acquired by MushroomRL usable to IAVI
        iavi_dataconverter = IAVICallback(self)
        #* The core learning
        core = Core(agent, self, callbacks_fit=[], callback_step=iavi_dataconverter, preprocessors=[])
        core.learn(n_episodes=100, n_steps_per_fit=1, render=False)

        #* emptying the storage, because we want to analyze the data after learning only
        iavi_dataconverter.clean()
        #previous state, the action sampled, reward, reached state, absorbing flag, last step flag.
        dataset = core.evaluate(n_episodes=n_trajectories, render=False)

        trajectories, action_probabilities, transition_probabilities, ground_r = iavi_dataconverter.get(wind)
        return trajectories, action_probabilities, transition_probabilities, ground_r

    def _clip_width(self, x : int) -> int:
        return max(0, min(x, self._width - 1))
    
    def _clip_height(self, y : int) -> int:
        return max(0, min(y, self._height - 1))

    def _in_trap(self, state : np.ndarray) -> bool:
        return any(np.array_equal(trap, state) for trap in self._traps)
    
    def _get_reward(self, state : np.ndarray, action: np.ndarray=None) -> float:
        step = self.action_space.action_step(action[0])
        if self.r is not None:
            return self.r[state[1]][state[0]]#[action[0]]
        movecost = 0.0#np.linalg.norm(step)
        if np.array_equal(state, self._goal):
            return self.reward_ - movecost
        elif self._in_trap(state):
            return self.trap_cost - movecost
        #if action is None:
        #    action = self.convert_to_grid(state, self._width) - self.convert_to_grid(self.last_state, self._width)
        return -movecost#-1.0#0.0

    # ((1, 0), (0, 1), (-1, 0), (0, -1), (-1,-1), (-1,1), (1,-1), (1,1))
    def _step(self, state : np.ndarray, action : np.ndarray):
        #print("calling _step:", state, action, self.action_space.action_step(action[0]))
        #self.last_state = copy.deepcopy(state)
        self._grid_step(state, action)
        #print("after grid step", state)
        absorbing = np.array_equal(state, self._goal)
        reward = self._get_reward(state, action)
        #print("returning", state)
        #print(state, reward)
        return state, reward, absorbing, {}
    
    def _grid_step(self, state : np.ndarray, action : np.ndarray):
        """
        state: width, height
        """
        #print(state)
        #print(old_state)
        temp_state = state + self.action_space.action_step(action[0])
        # print("t",temp_state)
        # print("s", state)
        # print("o", old_state)
        # print(self._in_trap(old_state))
        if len(self._traps) == 0:
            state[0] = self._clip_height(temp_state[0])
            state[1] = self._clip_width(temp_state[1])
        else:

            if self._in_trap(state): # coming from a trap

                if self._in_trap(temp_state): # next step is still in trap
                    #print("temp in trap")
                    state[0] = temp_state[0]
                    state[1] = temp_state[1]
                # next step is not in trap. only do it if it is an exit
                elif not any([np.array_equal(state,t_exit[0]) and np.array_equal(temp_state,t_exit[1]) for t_exit in self._trap_exits]):
                    pass
                else:
                    state[0] = temp_state[0]
                    state[1] = temp_state[1]
            else:
                state[0] = self._clip_height(temp_state[0])
                state[1] = self._clip_width(temp_state[1])
            #print(state)


class TDCallback(Callback):

    def __init__(self, env, agent):
        self._agent = agent
        self.gamma = env._mdp_info.gamma
        self._nS = env._mdp_info.observation_space.n
        self.V_ns = Table(self._nS)
        self.td_errors = {s:0 for s in range(self._nS)}
        self.states = []

        super().__init__()

    def __call__(self, dataset):
        """
        Add samples to the samples list.

        Args:
            dataset (list): the samples to collect.

        """
        dataset=dataset[0]
        state, _, reward, next_state, _, _ = dataset

        action_probs = self._agent.policy(next_state)
        vs_ns = [self._agent.Q[next_state, action] for action in range(self._agent.Q.n_actions)]
        old_v = self.V_ns[next_state]
        self.V_ns[state] = sum(action_probs * vs_ns)
        td = abs(reward + self.gamma * self.V_ns[state] - old_v)
        #for k in self.td_errors.keys():
        #    self.td_errors[k].append(0)
        self.states.append(state[0])
        self.td_errors[state[0]] += td

    def get(self):
        """
        Returns:
             The current collected data as a list.

        """
        return self.td_errors, self.states

    def clean(self):
        """
        Delete the current stored data list

        """
        self._data_list = list()#
        self.states = []
        self.td_errors = {s:0 for s in range(self._nS)}



class IAVICallback(Callback):
    """
    This callback can be used to collect the action values in all states at the
    current time step.

    """
    def __init__(self, env, agent, logger=None, statement=""):
        """
        Constructor.

        Args:
            env (Environment): the mushroom environment.

        """
        self.gamma = env._mdp_info.gamma
        self._width = env._width
        self._height = env._height
        self._nS = env._mdp_info.observation_space.n
        self._nA = env._mdp_info.action_space.n
        self._actions = env._mdp_info.action_space.actions
        self._agent = agent
        self._logger = logger
        self.statement = statement

        super().__init__()
        self._data_list = [[]]
        self.V = Table(self._nS)
        self.V_ns = Table(self._nS)
        self.cum_td = Table(self._nS)
        self.cum_td_ns = Table(self._nS)
        self.cum_td_q = Table(self._agent.Q.shape)

        
        #self.td_errors_ns = []
        #self.td_errors_action_value = []

    def __call__(self, dataset):
        # state, action, reward, next state, _, _
        dataset=dataset[0]
        state, action, reward, next_state, _, _ = dataset
        #self._data_list[-1].append([state, action, reward, next_state])
        #if dataset[-2]: # absorbing
        #    self._data_list.append([])
        
        self.cum_td_q[state, action] += self._agent.action_value_td_error
        #ic(self._agent.policy(next_state))
        action_probs = self._agent.policy(state)
        vs = [self._agent.Q[state, action] for action in range(self._agent.Q.n_actions)]
        old_v = self.V[state]
        self.V[state] = sum(action_probs * vs)
        self.cum_td[state] = abs(reward + self.gamma * self.V[state] - old_v)

        action_probs = self._agent.policy(next_state)
        vs_ns = [self._agent.Q[next_state, action] for action in range(self._agent.Q.n_actions)]
        old_v = self.V_ns[next_state]
        self.V_ns[state] = sum(action_probs * vs_ns)
        self.cum_td_ns[state] = abs(reward + self.gamma * self.V_ns[state] - old_v)
#        self.td_errors[state].append(abs(reward + self.gamma * self.V_ns[state] - old_v))


    # def __call__(self, dataset):
    #     # state, action, reward, next state, _, _
    #     dataset=dataset[0]
    #     state, action, reward, next_state, _, _ = dataset
    #     #self._data_list[-1].append([state, action, reward, next_state])
    #     #if dataset[-2]: # absorbing
    #     #    self._data_list.append([])
        
    #     self.td_errors_action_value.append(self._agent.action_value_td_error)
    #     #ic(self._agent.policy(next_state))
    #     action_probs = self._agent.policy(state)
    #     vs = [self._agent.Q[state, action] for action in range(self._agent.Q.n_actions)]
    #     old_v = self.V[state]
    #     self.V[state] = sum(action_probs * vs)
    #     td = reward + self._env._mdp_info.gamma * self.V[state] - old_v
    #     self.td_errors.append(td)

    #     action_probs = self._agent.policy(next_state)
    #     vs_ns = [self._agent.Q[next_state, action] for action in range(self._agent.Q.n_actions)]
    #     old_v = self.V_ns[next_state]
    #     self.V_ns[state] = sum(action_probs * vs_ns)
    #     td = reward + self._env._mdp_info.gamma * self.V_ns[state] - old_v

    #     self.td_errors_ns.append(td)
    
    #def get_trajectories(self):
    #    return self._data_list[:-1]

    # TODO I guess we could sum up here??
    def get_td_errors_action_value(self):
        #return list(self.td_errors_action_value)
        return self.cum_td_q
    
    def get_td_errors(self):
        #return copy.deepcopy(self.td_errors),\
        #       copy.deepcopy(self.td_errors_ns)
        return self.cum_td, self.cum_td_ns
               

    def get(self, wind):
        """
        Args:
            wind: float
                The probability of wind being applied.
        Returns:
            trajectories, action_probabilities, transition_probabilities, ground_r
            usable for the algorithms in https://github.com/NrLabFreiburg/inverse-q-learning

        """
        # filter out the empty trajectory at the end
        trajectories = self._data_list[:-1]
        #trajectories=np.array(self._data_list, dtype=int)

        #trajectories = np.array(tuple(filter(lambda x: bool(x), self._data_list)))
        # add the goal as the last state in all trajectories
        # TODO this is a hack, and I don't like it.
        # for trajectory in trajectories:
        #     last_state = trajectory[-1]
        #     trajectory.append([last_state[3], 4, last_state[2], last_state[3]])

        action_probabilities = gm.get_action_probabilities(trajectories, self._nS, self._nA)

        transition_probabilities = np.array(
                [[[gm.transition_probability(i, j, k, self._width, self._height, self._actions, wind=wind)
                for k in range(self._nS)]
                for j in range(self._nA)]
                for i in range(self._nS)])
        
        ground_r = np.array([self._env._get_reward(self._env.convert_to_grid([s], self._env._width)) for s in range(self._nS)])
        return trajectories, action_probabilities, transition_probabilities, ground_r

    def clean(self):
        """
        Delete the current stored data list

        """
        self._data_list = [[]]
    
    def format(self):
        raise NotImplementedError
