from inspect import signature
import itertools
from dataclasses import dataclass, asdict, field
from functools import partial
from typing_extensions import Literal
import os
import numpy as np
import json
from tqdm import tqdm

from datetime import datetime


from tqdm import tqdm

from icecream import install
install()
import insect_rl.mdp.utils.grid_math as gm
from insect_rl.mdp.utils.grid_math import INTERCARDINALS, CARDINALS, Action, transition_probability
from insect_rl.mdp.gridworld import Gridworld, CustomGridworld, LearningGridworld
from insect_rl.algorithms.evaluation import IAVIEvaluation



Point = tuple[float, float]


@dataclass
class EnvironmentConfig:
    width: int
    height: int
    goals: list[Point]
    start: Point
    reward: float = 1
    traps: list[Point] = None
    trapcost: float = -10.0
    trap_exits: list[tuple[Point, Point]] = None
    wind: float = 0.0

    def __post_init__(self):
        if self.traps is None:
            self.traps = []
        if self.trap_exits is None:
            self.trap_exits = []
    
    def update(self, data: dict):
        for k, v in data.items():
            setattr(self, k, v)

    @classmethod
    def from_kwargs(cls, **kwargs):
        # fetch the constructor's signature
        cls_fields = set(signature(cls).parameters)

        args = {name: value for name, value in kwargs.items() if name in cls_fields}
        return cls(**args)
    
    def __str__(self):
        d = asdict(self)
        d.pop('traps')
        d.pop('trap_exits')
        #d.pop('start')
        d.pop('goals')
        #d.pop('width')
        #d.pop('height')
        s = "Env Config: " + str(d)
        return s


@dataclass
class AgentConfig:
    actions: tuple[Action] = INTERCARDINALS
    discount: float = .99
    #algorithm: mrl.core.Agent = mrl.algorithms.value.SARSA
    #learning_rate: mrl.utils.parameters.Parameter = mrl.utils.parameters.Parameter(value=0.6)
    #policy: mrl.policy.Policy = mrl.policy.Boltzmann(beta=1.0)
    #param: mrl.utils.parameters.Parameter = mrl.utils.parameters.Parameter(value=0.1)

    def __str__(self):
        return str(asdict(self)).replace('{', '').replace('}', '').replace('\'', '').replace(':','')


def _iql_simulation(agent_config, env_config, transition_probability, iterations):

    env = CustomGridworld(
        **asdict(env_config),
        **asdict(agent_config),
        transition_probability=transition_probability
    )
    return env.collect_demonstrations(iterations)


def _mrl_simulation(agent_config, env_config, iterations):

    #* construct environment
    env = LearningGridworld(
        **asdict(env_config),
        **asdict(agent_config)
    )

    #* RL algorithm
    ant = agent_config.algorithm(
        env.mrl_gw.info, agent_config.policy, agent_config.learning_rate
    )
    return env.collect_demonstrations(ant, iterations)


class Simulation:
    def __init__(self, actions, env_config, backend: Literal["iql", "mrl"] = "iql") -> None:
        
        _agentc = AgentConfig(actions=actions)

        self._results = None
        self._eval = IAVIEvaluation(env_config.width, env_config.height, env_config.start)
        _probs = Simulation.calc_transition_probabilities(envc, actions)

        if backend=="iql":
            self._sim = partial(_iql_simulation, _agentc, env_config, _probs)
        elif backend=="mrl":
            self._sim = partial(_mrl_simulation, _agentc, env_config, _probs)
        else:
            raise NotImplementedError("backend not implemented")
    
    @staticmethod
    def calc_transition_probabilities(width, height, actions, wind, traps, trap_exits, **kwargs):
        goal_id = kwargs["goals"][0]
        goal_id = gm.point_to_int(goal_id, width)

        n_states = width * height
        n_actions = len(actions)
        # 20cm wide exit board
        trap_exit_ids = [(gm.point_to_int(xy1, width), gm.point_to_int(xy2, width)) for (xy1, xy2) in trap_exits]

        traps_ids = [gm.point_to_int(trap, width) for trap in traps]
        print("get trans prons")
        # TODO calculate ones and then apply to all
        transition_probability = np.array(
            [[[gm.transition_probability(i, j, k, width, height, actions, wind=wind)
            for k in range(n_states)]
            for j in range(n_actions)]
            for i in range(n_states)])
        print("tp:", transition_probability)
        for t in traps_ids:
            for i in range(n_states):
                if i not in traps_ids and (t,i) not in trap_exit_ids:
                    transition_probability[t,:,i] = np.array([0]*n_actions)
        
        for i in range(n_states):
            transition_probability[goal_id,:,i] = np.array([0]*n_actions)
        print( np.max(transition_probability[goal_id]))

        return transition_probability

    def run(self, iterations: int=100):
        """
        returns trajectories, action_probabilities, transition_probabilities, ground_r
        """
        self._results = self._sim(iterations)
        ic(self._results)
        exit()
        return self._results
    
    @staticmethod
    def run_with_data(trajectories, envc, actions):

        _agentc = AgentConfig(actions=actions)
        print("calculate transition probabilities")
        transition_probabilities = Simulation.calc_transition_probabilities(envc, actions)
        print("construct Gridworld")
        env = CustomGridworld(
            **asdict(envc),
            **asdict(_agentc),
            transition_probability= transition_probabilities
        )
        trajectories = env.convert_trajectories(trajectories)
        return trajectories, env.get_action_probabilities(trajectories), \
            transition_probabilities, env.ground_r
    
    def evaluate(self, **kwargs) -> dict:
        return self._eval(*self._results[1:], **kwargs)

    @staticmethod
    def evaluate_with_data(envc, action_probabilities, transition_probabilities, ground_r, **kwargs):
        _eval = IAVIEvaluation(envc.width, envc.height, envc.start)
        return _eval(action_probabilities, transition_probabilities, ground_r, **kwargs)



# TODO I would really like a configurable Boltzmann distribution
# TODO discount propagation
# TODO visualize trajectories


def reachable(s1, s2, prob):
    return any(prob[s1, :, s2])


def run_with_data(trajectories, envc, actions, reward, wind, trapcost):
    envc.reward = reward
    envc.wind = wind
    envc.trap_cost = trapcost

    trajectories, action_probabilities, transition_probs, ground_rb = Simulation.run_with_data(trajectories, envc, actions)
    print("evaluate")

    #np.savetxt(res_dir + '/ground_r.out', ground_rb, delimiter=",")

    label = f"Reward: {reward}, Trap Cost: {trapcost}, Wind: {wind}"

    fig_name = "gridworld_imgs/" + str(envc).replace(" ", "").replace("(", "").replace(")", "").replace(",", "_")
    return Simulation.evaluate_with_data(envc, action_probabilities, transition_probs, ground_rb)


def run(envc, actions, iterations, reward, wind, trap_cost, res_dir="./", plot=False, save_plots=False):
    #envc = EnvironmentConfig(width=15,height=25,goal=(14,20),start=(10,0), reward=20, traps=[(0,10), (1,10),(2,10),(3,10),(4,10),(5,10)], trap_cost=-50)
    envc.reward = reward
    envc.wind = wind
    envc.trap_cost = trap_cost

    sim = Simulation(actions, envc, backend="iql")
    trajectoriesb, action_probabilitiesb, transition_probabilitiesb, ground_rb = sim.run(iterations)

    label = f"Reward: {reward}, Trap Cost: {trap_cost}, Wind: {wind}"

    fig_name = "gridworld_imgs/" + str(envc).replace(" ", "").replace("(", "").replace(")", "").replace(",", "_")
    return sim.evaluate(res_dir=res_dir, plot=plot, save_plots=save_plots, fig_name=fig_name, label=label)


WYSTRACH2020_ENVIRONMENT = {
    'width': 23, # Melaphorus bagoti: minX -.565 maxX 1.62
    'height': 60, # M. bagoti: minY -.179 maxY 5.768
    'start': (11,1), # start at (1.1,-.1) 
    'goals': [(10,50)], # (.4, 5.0)
    'traps': [(x,30) for x in range(1,22)], # M.bagoti 10 cm wide, 2m long, 10cm deep with 20cm wide exit board
    'trap_exits': [((17,30), (17,31)), ((18,30), (18,31))]
}

TEST_ENVIRONMENT = {
    'width': 3,
    'height': 3,
    'start': (0,0),
    'goals': [(2,2)],
    'traps': [(1,1), (2,1)],
    'trap_exits': [((1,1),(1,2))]
}


if __name__=="__main__":
    rewards = [1, 10, 50, 100, 500]
    winds = [0.0, .2, .5, .8, 1.0]
    trap_costs = [-1, -10, -100]

    actions = CARDINALS
    iterations = 10000

    env = TEST_ENVIRONMENT
    envc = EnvironmentConfig.from_kwargs(**env)

    dir_name = f"./results/{envc.width}x{envc.height}/"
    # dd/mm/YY H:M:S
    dir_name += datetime.now().strftime("%d_%m_%Y_%H_%M") + "/"
    os.makedirs(dir_name + "gridworld_imgs/")

    with open(dir_name + 'readme.txt', 'w') as f:
        f.write(f'{envc.start=}\n{envc.goals=}\n{envc.traps=}\n{envc.trap_exits=}\n{actions=}\n{iterations=}')

    for reward, wind, trap_cost in tqdm(itertools.product(rewards, winds, trap_costs)):
        res = run(envc, actions, iterations, reward, wind, trap_cost, res_dir=dir_name, save_plots=True)
        for (name, arr) in res.items():
            np.savetxt(f"{dir_name}{name}.csv", arr, delimiter=",")
