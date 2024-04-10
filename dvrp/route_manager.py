import torch

from dvrp.object.route import Route

class RouteManager:
    vehicle_obs = None
    node_obs = None
    road_obs = None
    route = None

    def __init__(self, dvrp):
        self.dvrp = dvrp

    def set_obs(self, obs):
        self.vehicle_obs = obs[0]
        self.node_obs = obs[1]
        self.road_obs = obs[2]

    def set_route(self, route):
        self.route = route

    def get_obs_tensor(self):
        static_tensor = torch.tensor([node[:-1] for node in self.node_obs])
        dynamic_tensor = torch.tensor(self.road_obs)

        return static_tensor, dynamic_tensor

    def get_node_info(self):
        # demand, is_served
        return [[row[2], row[-1]] for row in self.node_obs]
    
    def get_vehicle_info(self):
        # loc_node_id, capacity
        return self.vehicle_obs
    
    def get_route(self):
        route_list = []
        for row in self.route:
            r = Route()
            for id in row:
                node = self.dvrp.get_node(id)
                r.add_node(node)
            route_list.append(r)
        return route_list
    