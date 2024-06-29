from dvrp.dvrp import DVRP

from agent.reinforce import REINFORCE
from meta_heuristics.alns_agent import ALNS_Solver
from meta_heuristics.ga import GA
from meta_heuristics.bso_aco import BSO_ACO

env = DVRP()
agent = REINFORCE()

# reinforcement learning
print("RL")
obs = env.reset()
while True:
    action = agent.get_action(obs, False)
    obs, reward, is_done = env.step(action)

    if is_done:
        break

# meta heuristics
print("Meta-Heuristics")
obs = env.reset()
while True:
    action = GA(obs).run()
    obs, reward, is_done = env.step(action)

    if is_done:
        break

