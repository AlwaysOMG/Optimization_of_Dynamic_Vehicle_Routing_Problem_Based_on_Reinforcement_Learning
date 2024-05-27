import numpy as np

class ACO:
    best_solution = None
    best_cost = float('inf')
    
    def __init__(self, obs, current_time):
        self.vehicle_obs = obs[0]
        self.node_obs = obs[1]
        self.road_obs = obs[2]
        self.current_time = current_time
        
        self.node_num = len(self.node_obs)
        self.vehicle_num = len(self.vehicle_obs)

        self.num_ants = 10
        self.num_iterations = 100
        self.alpha = 1
        self.beta = 5
        self.evaporation_rate = 0.1
        self.pheromone_deposit = 10
        self.pheromone_matrix = np.ones((self.node_num, self.node_num))

    def run(self):
        for _ in range(self.num_iterations):
            solutions = [self.construct_solution() for _ in range(self.num_ants)]
            self.update_pheromones(solutions)
            self.update_best_solution(solutions)
        
        routes = []
        for route in self.best_solution:
            route.pop(0)
            routes.append(route)
        return routes

    def construct_solution(self):
        solution = []
        unvisited = set(range(1, self.node_num))
        for i in range(self.node_num):
            if self.node_obs[i][7] == True:
                unvisited.remove(i)
        for i in range(self.vehicle_num):
            if self.vehicle_obs[i][0] in unvisited:
                unvisited.remove(self.vehicle_obs[i][0])
        
        for i in range(self.vehicle_num):
            route = [self.vehicle_obs[i][0]]
            capacity = self.vehicle_obs[i][1]
            while unvisited:
                feasible_customers = [c for c in unvisited if self.is_feasible(c, capacity)]
                if not feasible_customers:
                    break
                
                current_customer = route[-1]
                next_customer = self.select_next_customer(current_customer, feasible_customers)
                route.append(next_customer)
                capacity -= self.node_obs[next_customer][2]
                unvisited.remove(next_customer)
            route.append(0)
            solution.append(route)
        return solution

    def is_feasible(self, customer, capacity):
        return self.node_obs[customer][2] <= capacity

    def select_next_customer(self, current_customer, feasible_customers):
        pheromones = np.array([self.pheromone_matrix[current_customer][c] for c in feasible_customers])
        travel_time = np.array([self.road_obs[current_customer][c] for c in feasible_customers])
        penalty = np.array([self.calculate_penalty(current_customer, c) for c in feasible_customers])
        probabilities = (pheromones ** self.alpha) * ((1 / travel_time) ** self.beta) * ((1 / 1 + penalty) ** self.beta)
        probabilities /= probabilities.sum()
        return np.random.choice(feasible_customers, p=probabilities)

    def calculate_penalty(self, current_customer, next_customer):
        arrival_time = self.current_time + self.road_obs[current_customer][next_customer]
        if arrival_time < self.node_obs[next_customer][3]:
            return self.node_obs[next_customer][5] * (self.node_obs[next_customer][3] - arrival_time)
        elif arrival_time > self.node_obs[next_customer][4]:
            return self.node_obs[next_customer][6] * (arrival_time - self.node_obs[next_customer][4])
        else:
            return 0

    def update_pheromones(self, solutions):
        self.pheromone_matrix *= (1 - self.evaporation_rate)
        for solution in solutions:
            for route in solution:
                for i in range(len(route) - 1):
                    cost = self.calculate_route_cost(route)
                    if cost != 0:
                        self.pheromone_matrix[route[i]][route[i + 1]] += self.pheromone_deposit / self.calculate_route_cost(route)

    def update_best_solution(self, solutions):
        for solution in solutions:
            cost = self.calculate_solution_cost(solution)
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_solution = solution

    def calculate_solution_cost(self, solution):
        return sum(self.calculate_route_cost(route) for route in solution)

    def calculate_route_cost(self, route):
        cost = 0
        time = self.current_time
        for i in range(len(route)-1):
            time += self.road_obs[route[i]][route[i+1]]
            cost += self.calculate_penalty(route[i], route[i+1])
        cost += time
        return cost