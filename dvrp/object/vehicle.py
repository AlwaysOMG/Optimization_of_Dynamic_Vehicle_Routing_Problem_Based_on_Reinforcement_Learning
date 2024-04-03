class Vehicle:
    total_travel_time = 0
    penalty = 0
    route = None
    road = None
    left_distance = 0

    def __init__(self, dvrp, param):
        self.dvrp = dvrp
        self.id = param.id
        self.capacity = param.capacity
        self.current_node = dvrp.get_depot()

    def drive(self, start_time, time_interval):
        current_time = start_time
        left_time = time_interval
        while left_time > 0:
            if self.road == None:
                next_customer = self.route.get_next_customer()
                self.road = self.dvrp.get_road(self.current_node, next_customer)
                travel_time = self.road.get_travel_time()
                self.current_node = next_customer
            else:
                travel_time = self.left_distance / self.road.get_speed()
            
            if travel_time <= left_time:
                current_time += travel_time
                self.provide_service(next_customer, current_time)
                self.total_travel_time += travel_time
                left_time -= travel_time
                self.road = None
                self.left_distance = 0
            else:
                self.left_distance = self.road.get_dist() - self.road.get_speed() * left_time
                self.total_travel_time += left_time
                left_time = 0
    
    def provide_service(self, customer, current_time):
        self.capacity -= customer.demand
        self.penalty += customer.receive_service(current_time)

    def set_route(self, route):
        self.route = route
    
    def get_current_node(self):
        return self.current_node
    
    def get_cost(self):
        return self.total_travel_time + self.penalty
