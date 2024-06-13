import sys
import time
import copy
import random

class Individual:
    def __init__(self, routes, fitness):
        self.routes = routes
        self.fitness = fitness
        
class GA:
    def __init__(self, obs):
        self.vehicle_obs = obs[0]
        self.node_obs = obs[1]
        self.road_obs = obs[2]
        self.current_time = obs[3]

        self.node_num = len(self.node_obs)
        self.vehicle_num = len(self.vehicle_obs)

        self.population_size = 100
        self.num_generations = 20
        self.mutation_rate = 0.2
        self.corssover_rate = 0.8
    
    def run(self):
        population = self.init_population()
        for _ in range(self.num_generations):
            population = self.evolve_population(population)
        best_individual = max(population, key=lambda x: x.fitness)
        best_routes = [r[1:] for r in best_individual.routes]
        return best_routes

    def init_population(self):
        population = []
        for _ in range(self.population_size):
            routes = self.generate_initial_solution()
            fitness = self.calculate_fitness(routes)
            population.append(Individual(routes, fitness))
        return population

    def evolve_population(self, population):
        new_population = []
        while len(new_population) < self.population_size:
            parent_1, parent_2 = self.selection(population)
            child_1, child_2 = self.crossover(parent_1, parent_2)
            self.mutate(child_1)
            self.mutate(child_2)
            new_population.append(child_1)
            new_population.append(child_2)
        return new_population

    def selection(self, population):
        while True:
            total_fitness = sum(individual.fitness for individual in population)
            [parent_1, parent_2] = random.choices(population, 
                                                  weights=[individual.fitness / total_fitness for individual in population], 
                                                  k=2)
            
            if all(x == y for x, y in zip(parent_1.routes, parent_2.routes)) and parent_1 != parent_2:
                population.remove(parent_2)
            else:
                return parent_1, parent_2

    def crossover(self, ind_1, ind_2):
        # uniform crossover
        parent_1 = self.encode(ind_1.routes)
        parent_2 = self.encode(ind_2.routes)
        child_1 = parent_1[:]
        child_2 = parent_2[:]
        for i in range(min(len(parent_1), len(parent_2))):
            if random.random() < self.corssover_rate:
                child_1[i] = parent_2[i]
                child_2[i] = parent_1[i]

        # decode
        routes_1 = self.decode(child_1, copy.deepcopy(ind_1.routes))
        routes_2 = self.decode(child_2, copy.deepcopy(ind_2.routes))

        # fix
        routes_1 = self.capacity_fix(self.duplicate_fix(routes_1))
        routes_2 = self.capacity_fix(self.duplicate_fix(routes_2))
        self.check(routes_1, "1")
        self.check(routes_2, "2")

        fitness_1 = self.calculate_fitness(routes_1)
        fitness_2 = self.calculate_fitness(routes_2)
        return Individual(routes_1, fitness_1), Individual(routes_2, fitness_2)

    def encode(self, routes):
        l = [sublist[1:-1] for sublist in routes]
        gene = [item for sublist in l for item in sublist]
        return gene

    def decode(self, gene, routes):
        # check
        if len(gene) != sum([len(r[1:-1]) for r in routes]):
            print("wrong decode")
            sys.exit()
        
        counter = 0
        for r_idx in range(len(routes)):
            for c_idx in range(1, len(routes[r_idx])-1):
                routes[r_idx][c_idx] = gene[counter]
                counter += 1
        
        return routes

    def duplicate_fix(self, routes):
        duplicate_routes = copy.deepcopy(routes)
        flatten_routes = [c for r in duplicate_routes for c in r]
        duplicates = {c for c in flatten_routes if flatten_routes.count(c) > 1}
        clean_routes = []
        for r in duplicate_routes:
            clean_route = [r[0]]
            for c in r[1:-1]:
                if c not in duplicates:
                    clean_route.append(c)
            clean_route.append(r[-1])
            clean_routes.append(clean_route)
        unvisited = self.get_unvisited_customer() - set([c for r in clean_routes for c in r])

        solution = []
        for i in range(self.vehicle_num):
            route = copy.deepcopy(clean_routes[i])
            capacity = self.vehicle_obs[i][1]
            while unvisited:
                current_load = sum([self.node_obs[c][2] for c in route[1:]])
                feasible_customers = [c for c in unvisited if self.is_feasible(c, capacity, current_load)]
                if not feasible_customers:
                    break         
                
                best_increase = float('inf')
                best_customer = None
                best_position = None
                for customer in feasible_customers:
                    old_cost = self.calculate_route_cost(route)
                    for pos in range(1, len(route)):
                        new_route = copy.deepcopy(route)
                        new_route.insert(pos, customer)
                        new_cost = self.calculate_route_cost(new_route)
                        increase = new_cost - old_cost
                        if increase < best_increase:
                            best_increase = increase
                            best_customer = customer
                            best_position = pos

                route.insert(best_position, best_customer)
                unvisited.remove(best_customer)

            solution.append(route)

        return solution

    def capacity_fix(self, routes):
        solution = []
        for i in range(self.vehicle_num):
            current_load_1 = sum([self.node_obs[c][2] for c in routes[i][1:]])
            capacity_1 = self.vehicle_obs[i][1]
            if current_load_1 <= capacity_1:
                solution.append(routes[i])
                continue

            overweight_route = routes[i]
            while current_load_1 > capacity_1:
                best_cost = float('inf')
                best_customer = None
                best_vehicle = None
                best_position = None
                for candidate in overweight_route[1:-1]:
                    for another_vehicle, another_route in enumerate(routes):
                        if i == another_vehicle:
                            continue
                        
                        current_load_2 = sum([self.node_obs[c][2] for c in another_route])
                        capacity_2 = self.vehicle_obs[another_vehicle][1]
                        if not self.is_feasible(candidate, capacity_2, current_load_2):
                            continue

                        for pos in range(1, len(another_route)):
                            tmp_route_1 = copy.deepcopy(overweight_route)
                            tmp_route_2 = copy.deepcopy(another_route)
                            tmp_route_1.remove(candidate)
                            tmp_route_2.insert(pos, candidate)
                            new_cost = self.calculate_route_cost(tmp_route_1) + self.calculate_route_cost(tmp_route_2)
                            if new_cost < best_cost:
                                best_customer = candidate
                                best_vehicle = another_vehicle
                                best_position = pos
                
                if best_customer is not None:
                    routes[i].remove(best_customer)
                    routes[best_vehicle].insert(best_position, best_customer)
                else:
                    max_demand = 0
                    max_customer = None
                    for customer in routes[i][1:-1]:
                        demand = self.node_obs[customer][2]
                        if demand > max_demand:
                            max_demand = demand
                            max_customer = customer
                    routes[i].remove(max_customer)
                current_load_1 = sum([self.node_obs[c][2] for c in routes[i][1:]])
            solution.append(routes[i])

        return solution            

    def mutate(self, ind):
        if random.random() < self.mutation_rate:
            routes = ind.routes            
            rand_vehicle = random.randint(0, self.vehicle_num-1)
            rand_route = routes[rand_vehicle]
            if len(rand_route) <= 2:
                return
            rand_customer = random.sample(rand_route[1:-1], 1)[0]
            rand_idx = rand_route.index(rand_customer)
            current_cost_1 = self.calculate_route_cost(rand_route)

            best_vehicle = None
            best_customer = None
            best_idx = None
            best_improv = float('inf')
            for new_v, new_r in enumerate(routes):
                if new_v == rand_vehicle:
                    for i in range(1, len(new_r)-1):
                        if i != rand_idx:
                            tmp_r = copy.deepcopy(new_r)
                            c = tmp_r[i]
                            tmp_r[rand_idx] = c
                            tmp_r[i] = rand_customer
                            improv = current_cost_1 - self.calculate_route_cost(tmp_r)

                            if improv > best_improv:
                                best_vehicle = rand_vehicle
                                best_customer = c
                                best_idx = i
                                best_improv = improv
                else:
                    current_cost_2 = self.calculate_route_cost(new_r)
                    for i in range(1, len(new_r)-1):
                        c = new_r[i]
                        tmp_r_1 = copy.deepcopy(rand_route)
                        tmp_r_2 = copy.deepcopy(new_r)
                        tmp_r_1[rand_idx] = c
                        tmp_r_2[i] = rand_customer

                        current_load_1 = sum([self.node_obs[c][2] for c in tmp_r_1[1:]])
                        current_load_2 = sum([self.node_obs[c][2] for c in tmp_r_2[1:]])
                        if current_load_1 > self.vehicle_obs[rand_vehicle][1] or current_load_2 > self.vehicle_obs[new_v][1]:
                            continue

                        improv = (current_cost_1 + current_cost_2) - \
                            (self.calculate_route_cost(tmp_r_1) + self.calculate_route_cost(tmp_r_2))
    
                        if improv > best_improv:
                            best_vehicle = new_v
                            best_customer = c
                            best_idx = i
                            best_improv = improv
        
            if best_customer != None:
                routes[rand_vehicle][rand_idx] = best_customer
                routes[best_vehicle][best_idx] = rand_customer
                ind.fitness = self.calculate_fitness(routes)
            
            self.check(routes, "mutate")

    def check(self, routes, str):
        for i, r in enumerate(routes):
            if r[0] != self.vehicle_obs[i][0]:
                print(f"loc at {str}")
                sys.exit()
            
            load = 0
            for c in r[1:]:
                load += self.node_obs[c][2]
            if load > self.vehicle_obs[i][1]:
                print(f"cap at {str}: {i} / {load} / {self.vehicle_obs[i][1]}")
                sys.exit()

    def generate_initial_solution(self):
        solution = []
        unvisited = self.get_unvisited_customer()
        for i in range(self.vehicle_num):
            route = [self.vehicle_obs[i][0]]
            capacity = self.vehicle_obs[i][1]
            while unvisited:
                current_load = sum([self.node_obs[customer][2] for customer in route[1:]])
                feasible_customers = [c for c in unvisited if self.is_feasible(c, capacity, current_load)]
                if not feasible_customers:
                    break
                
                if len(route) == 1:
                    random_customer = random.choice(feasible_customers)
                    route.append(random_customer)
                    unvisited.remove(random_customer)
                    continue                
                
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

    def calculate_fitness(self, solution):
        cost = 0
        for route in solution:
            cost += self.calculate_route_cost(route)
        return -cost

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