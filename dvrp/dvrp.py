import configparser

from dvrp.param.param_generator import ParamGenerator
from dvrp.object.vehicle import Vehicle
from dvrp.object.node import Depot, Customer
from dvrp.object.network import Network

config = configparser.ConfigParser()
config.read("./config.cfg")
instance_config = config['instance']

class DVRP:
    node_num = int(instance_config["customer_num"])+1
    travel_time_cv = float(instance_config["travel_time_cv"])
    update_interval = int(instance_config["travel_time_update_interval"])

    def __init__(self):
        self.param_generator = ParamGenerator()
    
    def reset(self):
        self.current_time = 0

        self.node_param, self.vehicle_param = self.param_generator.generate_parameter()      
        self.node_list = [Depot(self, self.node_param[0])] + \
            [Customer(self, p) for p in self.node_param[1:]]
        self.depot = self.node_list[0]
        self.vehicle_list = [Vehicle(self, p) for p in self.vehicle_param]
        self.network = Network(self.node_list, self.travel_time_cv)

        return self.get_observation()

    def step(self, route_list):
        self.move_vehicle(route_list)
        self.update_travel_time()

        obs = self.get_observation()
        is_done = self.check_done()
        reward = self.get_reward(is_done)
        
        return obs, reward, is_done

    def move_vehicle(self, route_list):
        latest_finish_time = 0
        for vehicle, route in zip(self.vehicle_list, route_list):
            vehicle.set_route(route)
            finish_time = vehicle.drive(self.current_time, self.update_interval)
            if finish_time > latest_finish_time:
                latest_finish_time = finish_time
        self.current_time += latest_finish_time

    def update_travel_time(self):
        self.network.update_travel_time()

    def get_observation(self):
        vehicle_obs = [vehicle.get_obs() for vehicle in self.vehicle_list]
        node_obs = [node.get_obs() for node in self.node_list]
        road_obs = self.network.get_obs()

        return [vehicle_obs, node_obs, road_obs, self.get_current_time()]
    
    def check_done(self):
        # check vehicle back to the depot
        for vehicle in self.vehicle_list:
            if vehicle.get_location() != self.depot:
                return False

        # check customer been served
        for node in self.node_list:
            if node == self.depot:
                continue
            if node.check_served() == False:
                return False
                
        return True
    
    def get_reward(self, is_done):
        if is_done == True:
            total_cost = sum(vehicle.get_total_cost() for vehicle in self.vehicle_list)
            return -total_cost
        else:
            return 0
        
    def get_travel_cost(self):
        return sum(vehicle.get_travel_cost() for vehicle in self.vehicle_list)
    
    def get_penalty_cost(self):
        return sum(vehicle.get_penalty_cost() for vehicle in self.vehicle_list)
    
    def get_depot(self):
        return self.depot
    
    def get_node(self, id):
        return self.node_list[id]

    def get_road(self, node_1, node_2):
        return self.network.get_road(node_1, node_2)

    def get_num_node(self):
        return self.node_num
    
    def get_current_time(self):
        return self.current_time
