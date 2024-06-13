import random
import copy

from agent.aco import ACO

class BSO_ACO:
    best_solution = None
    best_cost = float('inf')
    
    def __init__(self, obs):
        self.vehicle_obs = obs[0]
        self.node_obs = obs[1]
        self.road_obs = obs[2]
        self.current_time = obs[3]
        
        self.node_num = len(self.node_obs)
        self.vehicle_num = len(self.vehicle_obs)

        self.aco = ACO(obs)
        self.perc_elitist = 0.1
        self.p_elitists = 0.2
        self.p_one = 0.6
        self.solutions_num = 20
    
    def run(self):
        init_solutions = self.aco.get_solutions()
        elitists, normals = self.clustering(init_solutions)
        solutions = []
        for i in range(self.solutions_num):
            if random.random() < self.perc_elitist:
                if random.random() < self.p_one:
                    si = random.choice(elitists)
                    si_prime = self.neighborhood_search(si)
                else:
                    si = random.choice(elitists)
                    sj = random.choice(elitists)
                    si_prime = self.aco.new_solution(si, sj)
            else:
                if random.random() < self.p_one:
                    si = random.choice(normals)
                    si_prime = self.neighborhood_search(si)
                else:
                    si = random.choice(normals)
                    sj = random.choice(normals)
                    si_prime = self.aco.new_solution(si, sj)
            
            if self.calculate_fitness(si_prime) > self.calculate_fitness(si):
                solutions.append(si_prime)
            else:
                solutions.append(si)
        
        best_solution = self.select_best_solution(solutions)
        return [r[1:] for r in best_solution]

    def clustering(self, solutions):
        solutions.sort(key=lambda s: self.calculate_fitness(s), reverse=True)
        n_elitist = int(self.perc_elitist * len(solutions))
        return solutions[:n_elitist], solutions[n_elitist:]
    
    def neighborhood_search(self, solution):
        s = self.intra_two_opt(solution)
        s = self.inter_two_opt(s)
        s = self.relocate(s)
        s = self.exchange(s)
        return s

    def intra_two_opt(self, solution):
        s = copy.deepcopy(solution)
        for vehicle, route in enumerate(s):
            if len(route) < 3:
                continue
            
            best_cost = self.calculate_route_cost(route)
            best_route = None
            improved = True
            while improved:
                improved = False
                for i in range(1, len(route)-2):
                    for j in range(i+2, len(route)):
                        tmp_route = copy.deepcopy(route)
                        tmp_route = tmp_route[:i] + tmp_route[i:j][::-1] + route[j:]
                        tmp_cost = self.calculate_route_cost(tmp_route)
                        if tmp_cost < best_cost:
                            best_cost = tmp_cost
                            best_route = tmp_route
                            improved = True
            
            if best_route is not None:
                s[vehicle] = best_route
        return s

    def inter_two_opt(self, solution):
        s = copy.deepcopy(solution)
        for vehicle_1 in range(len(s)):
            if len(solution[vehicle_1]) <= 2:
                continue
            for vehicle_2 in range(vehicle_1+1, len(s)):
                if len(solution[vehicle_2]) <= 2:
                    continue
                route_1 = s[vehicle_1]
                route_2 = s[vehicle_2]
                for i in range(1, len(route_1)-1):
                    for j in range(1, len(route_2)-1):
                        tmp_route_1 = route_1[:i] + route_2[j:]
                        tmp_route_2 = route_2[:j] + route_1[i:]
                        if self.is_feasible(vehicle_1, tmp_route_1) and self.is_feasible(vehicle_2, tmp_route_2):
                            current_cost_1 = self.calculate_route_cost(s[vehicle_1])
                            current_cost_2 = self.calculate_route_cost(s[vehicle_2])
                            tmp_cost_1 = self.calculate_route_cost(tmp_route_1)
                            tmp_cost_2 = self.calculate_route_cost(tmp_route_2)
                            if tmp_cost_1 + tmp_cost_2 < current_cost_1 + current_cost_2:
                                s[vehicle_1] = tmp_route_1
                                s[vehicle_2] = tmp_route_2
        return s            

    def relocate(self, solution):
        s = copy.deepcopy(solution)
        for vehicle_1, route_1 in enumerate(solution):
            if len(route_1) <= 2:
                continue
            for start_idx in range(1, len(route_1)-1):
                for end_idx in range(start_idx+1, len(route_1)):
                    for vehicle_2 in range(len(solution)):
                        if vehicle_1 != vehicle_2:
                            for target_idx in range(1, len(solution[vehicle_2])):
                                tmp_route_1 = copy.deepcopy(solution[vehicle_1])
                                tmp_route_2 = copy.deepcopy(solution[vehicle_2])
                                new_route_1 = tmp_route_1[:start_idx] + tmp_route_1[end_idx:]
                                new_route_2 = tmp_route_2[:target_idx] + tmp_route_1[start_idx:end_idx] + tmp_route_2[target_idx:]

                                if self.is_feasible(vehicle_1, new_route_1) and self.is_feasible(vehicle_2, new_route_2):
                                    current_cost_1 = self.calculate_route_cost(s[vehicle_1])
                                    current_cost_2 = self.calculate_route_cost(s[vehicle_2])
                                    new_cost_1 = self.calculate_route_cost(new_route_1)
                                    new_cost_2 = self.calculate_route_cost(new_route_2)
                                    if new_cost_1 + new_cost_2 < current_cost_1 + current_cost_2:
                                        s[vehicle_1] = new_route_1
                                        s[vehicle_2] = new_route_2
        return s

    def exchange(self, solution):
        return solution

    def select_best_solution(self, solutions):
        best_fitness = float('-inf')
        best_solution = None
        for solution in solutions:
            fitness = self.calculate_fitness(solution)
            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = solution
        
        return best_solution

    def calculate_fitness(self, solution):
        return self.aco.calculate_fitness(solution)

    def calculate_route_cost(self, route):
        return self.aco.calculate_route_cost(route)
    
    def is_feasible(self, vehicle, route):
        current_load = 0
        for c in route[1:]:
            current_load += self.node_obs[c][2]
        return current_load <= self.vehicle_obs[vehicle][1]
