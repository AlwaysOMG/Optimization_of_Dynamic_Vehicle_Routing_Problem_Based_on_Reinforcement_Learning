import configparser

from dvrp.param.param_generator import ParamGenerator
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

    def __init__(self):
        self.param_generator = ParamGenerator()
    
    def reset(self):
        node_param, vehicle_param = self.param_generator.generate_parameter()

        self.node_list = []
        for i, p in enumerate(node_param):
            if i == 0:
                depot = Depot(self, p)
                self.depot = depot
                self.node_list.append(depot)
            else:
                c = Customer(self, p)
                self.node_list.append(c)
        
        self.vehicle_list = []
        for p in vehicle_param:
            v = Vehicle(self, p)
            self.vehicle_list.append(v)
        
        self.network = Network(self.node_list, self.travel_time_cv)

    def step(self, route_list):
        self.move_vehicle(route_list)
        self.update_travel_time()

        obs = self.get_observation()
        is_finished = self.check_done()
        reward = self.get_reward(is_finished)
        
        return obs, reward, is_finished

    def move_vehicle(self, route_list):
        for vehicle, route in zip(self.vehicle_list, route_list):
            vehicle.set_route(route)
            vehicle.drive(self.current_time, self.update_interval)
        self.current_time += self.update_interval

    def update_travel_time(self):
        self.network.update_travel_time()

    def check_done(self):
        for node in self.node_list:
            if node == self.depot:
                continue
            if node.check_served() == False:
                return False
        
        for vehicle in self.vehicle_list:
            loc = vehicle.get_location()
            if loc != self.depot:
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
    
    def get_depot(self):
        return self.depot
    
    def get_node(self, id):
        return self.node_list[id]

    def get_road(self, node_1, node_2):
        return self.network.get_road(node_1, node_2)
        
    def get_observation(self):
        pass
    
    def display(self):
        print("Node")
        for node in self.node_list:
            print(f"{node.id}: ({node.x_loc}, {node.y_loc})")
            if node != self.depot:
                print(f"demand: {node.demand}, time window: [{node.earliest_service_time}, {node.latest_service_time}]")
                print(f"penalty: ({node.early_penalty}, {node.late_penalty})")
        
        print("Vehicle")
        for vehicle in self.vehicle_list:
            print(f"{vehicle.id}: {vehicle.capacity}")
        
        print("Network")
        for row in self.network.road_matrix:
            for r in row:
                if r != None:
                    print(f"dist: {r.get_dist()}, travel time: {r.get_travel_time()}")