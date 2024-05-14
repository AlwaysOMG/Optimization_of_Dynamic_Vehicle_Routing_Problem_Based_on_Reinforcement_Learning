class Vehicle:
    total_travel_time = 0
    penalty = 0
    route = None
    road = None
    road_left_distance = 0

    def __init__(self, dvrp, param):
        self.dvrp = dvrp
        self.id = param.id
        self.max_capacity = param.capacity
        self.capacity = param.capacity
        self.current_node = dvrp.get_depot()
        self.target_node = None

    def drive(self, start_time, time_interval):
        current_time = start_time
        left_time = time_interval
            
        while left_time > 0:
            if self.road == None:
                # set the destination and road
                self.target_node = self.route.get_next_customer()
                if self.target_node == None or self.current_node == self.target_node:
                    break
                if self.target_node.get_id() != 0 and self.target_node.check_served():
                    break
                #print(f"vehicle {self.id}: set to drive to {self.target_node.get_id()}")

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
                    self.capacity = self.max_capacity
                    #print(f"vehicle {self.id}: bact to depot at time {current_time}")
                else:
                    self.provide_service(self.current_node, current_time)
                    #print(f"vehicle {self.id}: finish {self.current_node.get_id()} at time {current_time}")
                left_time -= travel_time
            else:
                # time advance
                self.total_travel_time += left_time
                # vehicle is still on the road
                self.road_left_distance -= self.road.get_speed() * left_time
                left_time = 0
    
    def provide_service(self, node, current_time):
        self.capacity -= node.get_demand()
        if self.capacity < 0:
            raise ValueError("capacity cant be smaller than 0")
        self.penalty += node.receive_service(current_time)

    def set_route(self, route):
        self.route = route
    
    def get_id(self):
        return self.id

    def get_location(self):
        return self.current_node

    def get_travel_cost(self):
        return self.total_travel_time
    
    def get_penalty_cost(self):
        return self.penalty
    
    def get_total_cost(self):
        return self.total_travel_time + self.penalty
    
    def get_obs(self):
        loc = self.target_node.get_id() if self.target_node else self.current_node.get_id()
        capacity = self.capacity - self.target_node.get_demand() \
            if self.target_node != None and self.target_node != self.dvrp.get_depot() \
                else self.capacity
        
        return [loc, capacity]
