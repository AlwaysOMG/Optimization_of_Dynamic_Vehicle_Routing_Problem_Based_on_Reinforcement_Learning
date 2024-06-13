import torch

from dvrp.object.route import Route

class RouteManager:
    route = None

    def __init__(self, dvrp):
        self.dvrp = dvrp
        self.static_feature_dim = 7
        self.dynamic_feature_dim = dvrp.get_num_node()

    def obs_to_tensor(self, obs):
        vehicle_obs = obs[0]
        node_obs = obs[1]
        road_obs = obs[2]
        current_time = obs[3]
        
        static_tensor = torch.tensor([node[:-1] for node in node_obs])
        dynamic_tensor = torch.tensor(road_obs, dtype=torch.float32)

        vehicle_info = vehicle_obs                          # loc_node_id, capacity
        node_info = [[row[2], row[-1]] for row in node_obs] # demand, is_served

        return [static_tensor, dynamic_tensor], [vehicle_info, node_info, current_time]
    
    def action_to_route(self, list):
        route = []
        for row in list:
            r = Route()
            for id in row:
                node = self.dvrp.get_node(id)
                r.add_node(node)
            route.append(r)
        return route
    
    def get_feature_dim(self):
        return [self.static_feature_dim, self.dynamic_feature_dim]
    