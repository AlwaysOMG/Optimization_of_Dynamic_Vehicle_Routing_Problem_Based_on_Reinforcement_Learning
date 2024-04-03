import configparser

from dvrp.object.vehicle import Vehicle
from dvrp.object.node import Depot, Customer
from dvrp.object.network import Network

config = configparser.ConfigParser()
config.read("config.cfg")
instance_config = config['instance']

class DVRP:
    current_time = 0

    travel_time_cv = float(instance_config["travel_time_cv"])
    update_interval = int(instance_config["travel_time_update_interval"])

    def __init__(self, param_list):
        self.node_list = []
        for i, p in enumerate(param_list[1]):
            if i == 0:
                depot = Depot(self, p)
                self.depot = depot
                self.node_list.append(depot)
            else:
                c = Customer(self, p)
                self.node_list.append(c)
        
        self.vehicle_list = []
        for p in param_list[0]:
            v = Vehicle(self, p)
            self.vehicle_list.append(v)
        
        self.network = Network(self.node_list, self.travel_time_cv)
    
    def get_depot(self):
        return self.depot
    
    def get_road(self, node_1, node_2):
        return self.network.get_road(node_1, node_2)
    
    def update_travel_time(self):
        self.network.update_travel_time()
    
    def step(self, route_list):
        self.move_vehicle(route_list)
        obs = self.get_info()
        is_finished = self.check_done()
        reward = self.get_reward()
        return obs, reward, is_finished
    
    def move_vehicle(self, route_list):
        # set route
        for vehicle, route in zip(self.vehicle_list, route_list):
            vehicle.set_route(route)
        # drive
        for vehicle in self.vehicle_list:
            vehicle.drive(self.current_time, self.update_interval)
        self.current_time += self.update_interval
        
    def get_info(self):
        pass

    def check_done(self):
        for node in self.node_list:
            if node == self.depot:
                continue
            if node.check_served() == False:
                return False
        return True
    
    def get_reward(self, is_finished):
        if is_finished == True:
            cost = 0
            for vehicle in self.vehicle_list:
                cost += vehicle.get_cost()
                return - cost
        else:
            return 0