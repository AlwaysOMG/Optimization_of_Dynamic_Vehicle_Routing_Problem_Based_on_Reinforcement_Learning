import copy
import random
import numpy.random as rnd

from alns import ALNS
from alns.accept import SimulatedAnnealing
from alns.select import RouletteWheel
from alns.stop import MaxIterations

class State:
    def __init__(self, routes, obj):
        self.routes = routes
        self.removed_list = []
        self.obj = obj

    def copy(self):
        return State(copy.deepcopy(self.routes))

    def objective(self):
        return self.obj

    def get_routes(self):
        return self.routes
    
    def get_removed_list(self):
        return self.removed_list

    def set_routes(self, routes):
        self.routes = routes

    def set_removed_list(self, removed_list):
        if self.removed_list is []:
            self.removed_list = removed_list
        else:
            for customer in removed_list:
                if customer not in self.removed_list:
                    self.removed_list.append(customer)

    def set_obj(self, obj):
        self.obj = obj

class ALNS_Solver:
    def __init__(self, obs):
        self.vehicle_obs = obs[0]
        self.node_obs = obs[1]
        self.road_obs = obs[2]
        self.current_time = obs[3]

        self.node_num = len(self.node_obs)
        self.vehicle_num = len(self.vehicle_obs)

        self.degree_of_destruction = 0.15
        self.travel_time_weight = 0.4
        self.time_window_weight = 0.8
        self.demand_weight = 0.3
        self.initial_temperature = 400
        self.cooling_rate = 0.99985
        self.end_temperature = 1e-3
        self.itr_num = 30000

    def run(self):
        SEED = 0
        alns = ALNS(rnd.RandomState(SEED))

        alns.add_destroy_operator(self.random_removal)
        alns.add_destroy_operator(self.random_route_removal)
        alns.add_destroy_operator(self.worst_removal)
        alns.add_destroy_operator(self.shaw_removal)
        alns.add_destroy_operator(self.cost_reducing_removal)
        alns.add_destroy_operator(self.exchange_reducing_removal)
        alns.add_repair_operator(self.greedy_insert)
        alns.add_repair_operator(self.regret_insert)

        init_sol = self.generate_initial_solution()
        vprtw_state = State(init_sol, 
                            sum([self.calculate_route_cost(r) for r in init_sol]))

        select = RouletteWheel([25, 5, 1, 0], 0.8, 6, 2)
        accept = SimulatedAnnealing(self.initial_temperature, 
                                    self.cooling_rate,
                                    self.end_temperature)
        stop = MaxIterations(self.itr_num)

        result = alns.iterate(vprtw_state, select, accept, stop)
        routes = [r[1:] for r in result.best_state.get_routes()]
        return routes

    def generate_initial_solution(self):
        solution = []
        unvisited = self.get_unvisited_customer()

        for i in range(self.vehicle_num):
            route = [self.vehicle_obs[i][0]]
            capacity = self.vehicle_obs[i][1]
            while unvisited:
                current_load = sum([self.node_obs[customer][2] for customer in route])
                feasible_customers = [c for c in unvisited if self.is_feasible(c, capacity, current_load)]
                if not feasible_customers:
                    break
                
                best_increase = float('inf')
                best_customer = None
                best_position = None
                for customer in feasible_customers:
                    old_cost = self.calculate_route_cost(route)
                    for pos in range(1, len(route)+1):
                        new_route = copy.deepcopy(route)
                        new_route.insert(pos, customer)
                        new_cost = self.calculate_route_cost(new_route)
                        increase = new_cost - old_cost
                        if increase < best_increase:
                            best_increase = increase
                            best_customer = customer
                            best_position = pos
                
                if best_customer is None:
                    break

                route.insert(best_position, best_customer)
                unvisited.remove(best_customer)
            
            route.append(0)
            solution.append(route)
        
        return solution

    def random_removal(self, vrp_state, rnd_state):
        solution = vrp_state.get_routes()
        customer_list = [customer for route in solution for customer in route[1:-1]]
        num_to_remove = int(len(customer_list) * self.degree_of_destruction)

        removed_customers = random.sample(customer_list, num_to_remove)
        vrp_state.set_removed_list(removed_customers)
        new_solution = [[customer for customer in route if customer not in removed_customers] for route in solution]
        vrp_state.set_routes(new_solution)
        
        return vrp_state

    def random_route_removal(self, vrp_state, rnd_state):
        solution = vrp_state.get_routes()

        route_to_remove = random.sample(range(0, self.vehicle_num), 1)[0]
        removed_customers = solution[route_to_remove][1:-1]
        vrp_state.set_removed_list(removed_customers)
        new_solution = [[customer for customer in route if customer not in removed_customers] for route in solution]
        vrp_state.set_routes(new_solution)
        
        return vrp_state

    def worst_removal(self, vrp_state, rnd_state):
        solution = vrp_state.get_routes()
        customer_list = [customer for route in solution for customer in route[1:-1]]
        num_to_remove = int(len(customer_list) * self.degree_of_destruction)

        customer_save = {}
        for route in solution:
            old_cost = self.calculate_route_cost(route)
            for i in range(1, len(route)-1):
                new_route = copy.deepcopy(route)
                customer = route[i]
                new_route.remove(customer)
                new_cost = self.calculate_route_cost(new_route)
                save = old_cost - new_cost
                customer_save[customer] = save
        
        worst_customers = sorted(customer_save, key=customer_save.get, reverse=True)[:num_to_remove]
        vrp_state.set_removed_list(worst_customers)
        new_solution = [[customer for customer in route if customer not in worst_customers] for route in solution]
        vrp_state.set_routes(new_solution)
        
        return vrp_state
    
    def shaw_removal(self, vrp_state, rnd_state):
        solution = vrp_state.get_routes()
        customer_list = [customer for route in solution for customer in route[1:-1]]
        num_to_remove = int(len(customer_list) * self.degree_of_destruction)

        removed_customers = []
        while len(removed_customers) < num_to_remove:
            seed_customer = random.sample(customer_list, 1)[0]
            similarities = [(self.calculate_similarties(seed_customer, i), i) 
                            for i in customer_list if i not in removed_customers and i != seed_customer]
            _, customer = min(similarities)
            removed_customers.append(customer)

        vrp_state.set_removed_list(removed_customers)
        new_solution = [[customer for customer in route if customer not in removed_customers] for route in solution]
        vrp_state.set_routes(new_solution)

        return vrp_state

    def cost_reducing_removal(self, vrp_state, rnd_state):
        solution = vrp_state.get_routes()
        customer_list = [customer for route in solution for customer in route[1:-1]]
        num_to_remove = int(len(customer_list) * self.degree_of_destruction)

        removed_customers = []
        iter_count = 0
        is_removed = False
        while iter_count < num_to_remove:
            iter_count += 1
            
            random_route_idx = random.randint(0, len(solution)-1)
            random_route = solution[random_route_idx]
            for v in random_route[1:-1]:
                if v in removed_customers:
                    continue

                i_1 = random_route[random_route.index(v) - 1]
                j_1 = random_route[random_route.index(v) + 1]
                
                for r in solution:
                    for k in range(1, len(r)-1):
                        i_2 = r[k - 1]
                        j_2 = r[k]
                        if i_1 != i_2 and j_1 != j_2:
                            if (self.road_obs[i_1][v] + self.road_obs[v][j_1] + self.road_obs[i_2][j_2] > 
                                self.road_obs[i_1][j_1] + self.road_obs[i_2][v] + self.road_obs[v][j_2]):
                                
                                removed_customers.append(v)
                                is_removed = True
                                break
                    if is_removed:
                        break
                if is_removed:
                    is_removed = False
                    break
        
        vrp_state.set_removed_list(removed_customers)
        new_solution = [[customer for customer in route if customer not in removed_customers] for route in solution]
        vrp_state.set_routes(new_solution)
                                            
        return vrp_state

    def exchange_reducing_removal(self, vrp_state, rnd_state):
        solution = vrp_state.get_routes()
        customer_list = [customer for route in solution for customer in route[1:-1]]
        num_to_remove = int(len(customer_list) * self.degree_of_destruction)

        removed_customers = []
        iter_count = 0
        is_removed = False
        while iter_count < num_to_remove:
            iter_count += 1
            
            random_route_idx = random.randint(0, len(solution)-1)
            random_route = solution[random_route_idx]
            for v_1 in random_route[1:-1]:
                if v_1 in removed_customers:
                    continue

                i_1 = random_route[random_route.index(v_1) - 1]
                j_1 = random_route[random_route.index(v_1) + 1]
                
                for r in solution:
                    for k in range(1, len(r)-2):
                        i_2 = r[k - 1]
                        v_2 = r[k]
                        j_2 = r[k + 1]

                        if v_2 in removed_customers:
                            continue

                        if i_1 != i_2 and j_1 != j_2:
                            if (self.road_obs[i_1][v_1] + self.road_obs[v_1][j_1] + 
                                self.road_obs[i_2][v_2] + self.road_obs[v_2][j_2] > 
                                self.road_obs[i_1][v_2] + self.road_obs[v_2][j_1] + 
                                self.road_obs[i_2][v_1] + self.road_obs[v_1][j_2]):
                                
                                removed_customers.append(v_1)
                                removed_customers.append(v_2)
                                is_removed = True
                                break
                    if is_removed:
                        break
                if is_removed:
                    is_removed = False
                    break
        
        vrp_state.set_removed_list(removed_customers)
        new_solution = [[customer for customer in route if customer not in removed_customers] for route in solution]
        vrp_state.set_routes(new_solution)
                                            
        return vrp_state

    def greedy_insert(self, vrp_state, rnd_state):
        solution = vrp_state.get_routes()
        removed_list = vrp_state.get_removed_list()
        capacity_list = [info[1] for info in self.vehicle_obs]

        while removed_list:
            best_cost = float('inf')
            best_vehicle = None
            best_position = None
            best_customer = None

            for i, route in enumerate(solution):
                current_load = sum([self.node_obs[customer][2] for customer in route])
                feasible_customers = [c for c in removed_list if self.is_feasible(c, capacity_list[i], current_load)]
                if not feasible_customers:
                    continue

                old_cost = self.calculate_route_cost(route)
                for customer in feasible_customers:
                    for pos in range(1, len(route)):
                        new_route = copy.deepcopy(route)
                        new_route.insert(pos, customer)
                        new_cost = self.calculate_route_cost(new_route)
                        increase = new_cost - old_cost
                        if increase < best_cost:
                            best_cost = increase
                            best_vehicle = i
                            best_position = pos
                            best_customer = customer

            if best_customer is not None:
                solution[best_vehicle].insert(best_position, best_customer)
                removed_list.remove(best_customer)
            else:
                break

        vrp_state.set_routes(solution)
        vrp_state.set_obj(sum([self.calculate_route_cost(r) for r in solution]))
        vrp_state.set_removed_list(removed_list)
        
        return vrp_state
    
    def regret_insert(self, vrp_state, rnd_state):
        solution = vrp_state.get_routes()
        removed_list = vrp_state.get_removed_list()
        capacity_list = [info[1] for info in self.vehicle_obs]

        while removed_list:
            best_customer = None
            best_vehicle = None
            best_pos = None
            best_regret = float('-inf')

            for customer in removed_list:
                best_cost = float('inf')
                second_best_cost = float('inf')
                tmp_vehicle = None
                tmp_pos = None
                for i, route in enumerate(solution):
                    current_load = sum([self.node_obs[customer][2] for customer in route])
                    if not self.is_feasible(customer, capacity_list[i], current_load):
                        continue

                    for pos in range(1, len(route)):
                        new_route = copy.deepcopy(route)
                        new_route.insert(pos, customer)
                        new_cost = self.calculate_route_cost(new_route)

                        if new_cost < best_cost:
                            second_best_cost = best_cost if best_cost != float('inf') else new_cost
                            best_cost = new_cost
                            tmp_vehicle = i
                            tmp_pos = pos
                        elif new_cost < second_best_cost:
                            second_best_cost = new_cost
                
                regret = second_best_cost - best_cost
                if regret > best_regret:
                    best_customer = customer
                    best_vehicle = tmp_vehicle
                    best_pos = tmp_pos
                    best_regret = regret
    
            if best_customer is not None:
                solution[best_vehicle].insert(best_pos, best_customer)
                removed_list.remove(best_customer)
            else:
                break
        
        vrp_state.set_routes(solution)
        vrp_state.set_obj(sum([self.calculate_route_cost(r) for r in solution]))
        vrp_state.set_removed_list(removed_list)
        
        return vrp_state

    def get_unvisited_customer(self):
        unvisited = set(range(1, self.node_num))
        for i in range(self.node_num):
            if self.node_obs[i][7] == True:
                unvisited.remove(i)
        for i in range(self.vehicle_num):
            if self.vehicle_obs[i][0] in unvisited:
                unvisited.remove(self.vehicle_obs[i][0])
        
        return unvisited

    def is_feasible(self, customer, capacity, current_load):
        return current_load + self.node_obs[customer][2] <= capacity

    def calculate_route_cost(self, route):
        cost = 0
        time = self.current_time
        for i in range(len(route)-1):
            time += self.road_obs[route[i]][route[i+1]]
            cost += self.calculate_penalty(route[i], route[i+1])
        cost += time
        return cost

    def calculate_penalty(self, current_customer, next_customer):
        arrival_time = self.current_time + self.road_obs[current_customer][next_customer]
        if arrival_time < self.node_obs[next_customer][3]:
            return self.node_obs[next_customer][5] * (self.node_obs[next_customer][3] - arrival_time)
        elif arrival_time > self.node_obs[next_customer][4]:
            return self.node_obs[next_customer][6] * (arrival_time - self.node_obs[next_customer][4])
        else:
            return 0

    def calculate_similarties(self, customer_i, customer_j):
        max_travel_time = max(max(sublist) for sublist in self.road_obs)
        travel_time = self.road_obs[customer_i][customer_j]
        max_time_window = max(self.node_obs[:][3])
        time_window_diff = abs(self.node_obs[customer_i][3]-self.node_obs[customer_j][3])
        max_demand = max(self.node_obs[:][2])
        demand_diff = abs(self.node_obs[customer_i][2]-self.node_obs[customer_j][1])

        similarities = self.travel_time_weight * (travel_time / max_travel_time) + \
            self.time_window_weight * (time_window_diff / max_time_window) + \
            self.demand_weight * (demand_diff / max_demand)
        
        return similarities
