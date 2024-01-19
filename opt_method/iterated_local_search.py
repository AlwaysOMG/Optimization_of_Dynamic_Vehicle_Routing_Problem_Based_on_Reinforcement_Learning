import random
import copy

from opt_method.utils import INF
from opt_method.utils import cal_cost
from opt_method.init_sol import insertion
from opt_method.local_search import two_opt_star, exchange, relocate

def local_search_block(routes, c_data, t_data, capacity):
    """    
    opt-sequence: 2-opt*, exchange, relocate.
    
    Whenever a better solution is found, immediately accept it, and resume the
    search from the 2-opt*.
    """

    new_routes = copy.deepcopy(routes)
    total_cost = sum(cal_cost(r, c_data, t_data, capacity) for r in new_routes)
    best_total_cost = INF
    while(total_cost < best_total_cost):
        best_total_cost = total_cost

        two_opt_star(new_routes, c_data, t_data, capacity)
        total_cost = sum(cal_cost(r, c_data, t_data, capacity) for r in new_routes)
        if total_cost < best_total_cost:
            continue

        exchange(new_routes, c_data, t_data, capacity)
        total_cost = sum(cal_cost(r, c_data, t_data, capacity) for r in new_routes)
        if total_cost < best_total_cost:
            continue

        relocate(new_routes, c_data, t_data, capacity)
        total_cost = sum(cal_cost(r, c_data, t_data, capacity) for r in new_routes)

    return new_routes

def perturbation(routes):
    """
    Relocate a customer without any restriction.
    """

    new_routes = copy.deepcopy(routes)
    route_1_idx = random.randint(0, len(new_routes)-1)
    route_2_idx = random.randint(0, len(new_routes)-1)
    customer_idx = random.randint(1, len(new_routes[route_1_idx])-2)
    customer = new_routes[route_1_idx].pop(customer_idx)
    pos = random.randint(1, len(new_routes[route_2_idx])-1)    
    new_routes[route_2_idx].insert(pos, customer)

    return new_routes

def acception_criteria(old_routes, new_routes, c_data, t_data, capacity):
    """
    Choose the better one as the current optimal solution.
    """
    
    cost1 = sum(cal_cost(r, c_data, t_data, capacity) for r in old_routes)
    cost2 = sum(cal_cost(r, c_data, t_data, capacity) for r in new_routes)
    if cost1 <= cost2:
        return old_routes, False
    else:
        return new_routes, True

def iterated_local_search(c_data, t_data, capacity, max_iter_num=10):
    """
    Iterates LS many times to find optimal solution, and perturb it 
    to make it escape from the local optimal.
    """
    
    init_routes = insertion(c_data, t_data, capacity)
    best_routes = local_search_block(init_routes, c_data, t_data, capacity)

    terminate_counter = 0
    while(terminate_counter < max_iter_num):
        new_routes = perturbation(best_routes)
        best_new_routes = local_search_block(new_routes, c_data, t_data, capacity)
        best_routes, is_improved = acception_criteria(best_routes, best_new_routes, c_data, t_data, capacity)
        terminate_counter = 0 if is_improved else terminate_counter+1
    
    return best_routes