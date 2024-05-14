from dvrp.dvrp import DVRP
from dvrp.route_manager import RouteManager
from agent.aco import ACO

env = DVRP()
mgr = RouteManager(env)

obs = env.reset()
while True:
    time = env.get_current_time()
    sol = ACO(obs, time).run()
    print(sol)
    route = mgr.action_to_route(sol)
    obs, reward, done = env.step(route)

    if done:
        print(reward)
        break
