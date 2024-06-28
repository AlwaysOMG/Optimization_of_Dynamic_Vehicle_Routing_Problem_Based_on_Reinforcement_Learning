from dvrp.dvrp import DVRP

def generate_route_list(obs):
    vehicle_obs = obs[0]
    node_obs = obs[1]

    route_list = []
    skip_list = [v_obs[0] for v_obs in vehicle_obs]
    for v_obs in vehicle_obs:
        current_capacity = v_obs[1]
        route = []
        for n_id, n_obs in enumerate(node_obs):
            if n_id == 0 or n_id in skip_list:
                continue
            
            if n_obs[-1] == True:
                continue

            if n_obs[2] > current_capacity:
                continue

            route.append(n_id)
            current_capacity -= n_obs[2]
            skip_list.append(n_id)
        
        route.append(0)
        route_list.append(route)
    
    return route_list

if __name__ == '__main__':
    env = DVRP()
    obs = env.reset()

    while not env.check_done():
        action = generate_route_list(obs)
        obs, reward, done = env.step(action)

    print(f"total cost: {-reward}")
    print([node.check_served() if node != env.get_depot() else None 
           for node in env.node_list])