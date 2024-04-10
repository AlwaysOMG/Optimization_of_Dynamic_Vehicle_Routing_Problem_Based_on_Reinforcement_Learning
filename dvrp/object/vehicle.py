class Vehicle:
    total_travel_time = 0
    penalty = 0
    route = None
    road = None
    road_left_distance = 0
    finish_service = False

    def __init__(self, dvrp, param):
        self.dvrp = dvrp
        self.id = param.id
        self.capacity = param.capacity
        self.current_node = dvrp.get_depot()
        self.target_node = None

    def drive(self, start_time, time_interval):
        if not self.finish_service:
            current_time = start_time
            left_time = time_interval
            
            while left_time > 0:
                if self.road == None:
                    # set the destination and road
                    self.target_node = self.route.get_next_customer()
                    self.road = self.dvrp.get_road(self.current_node, self.target_node)
                    self.road_left_distance = self.road.get_dist()
                    travel_time = self.road.get_travel_time()
                else:
                    travel_time = self.road_left_distance / self.road.get_speed()
            
                if travel_time <= left_time:
                    # time advance
                    current_time += travel_time
                    self.total_travel_time += travel_time
                    # move vehicle to destination
                    self.current_node = self.target_node
                    self.target_node = None
                    self.road = None
                    self.road_left_distance = 0
                    
                    # service
                    if self.current_node == self.dvrp.get_depot():
                        self.finish_service = True
                        print(f"{self.id} back to depot")
                        print(f"cost {self.total_travel_time + self.penalty}")
                        break
                    else:
                        self.provide_service(self.current_node, current_time)
                        left_time -= travel_time
                else:
                    # time advance
                    current_time += left_time
                    self.total_travel_time += left_time
                    # vehicle is still on the road
                    self.road_left_distance -= self.road.get_speed() * left_time
                    left_time = 0
                    print(f"{self.id} on the way to {self.target_node.get_id()} at {current_time}")
                    
    
    def provide_service(self, node, current_time):
        self.capacity -= node.get_demand()
        self.penalty += node.receive_service(current_time)
        print(f"{self.id} finish {node.get_id()} at {current_time}")

    def set_route(self, route):
        self.route = route
    
    def get_id(self):
        return self.id

    def get_location(self):
        return self.current_node

    def get_cost(self):
        return self.total_travel_time + self.penalty
    
    def get_obs(self):
        return [self.target_node.get_id() if self.target_node else self.current_node.get_id(), 
                self.capacity]