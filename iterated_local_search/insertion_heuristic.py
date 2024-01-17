import numpy as np
import copy

INF = 999999999

def init_route(c_data):
    """
    Initialize the route with the unoruted customer which has the earliest deadline
    """

    unrouted_customer = np.where(c_data[:, 8] == 0)[0]
    seed = unrouted_customer[np.argmin(c_data[unrouted_customer, 5])]
    c_data[seed, 8] = 1

    return [0, seed, 0]

def cal_rest_capacity(route, c_data, capacity):
    """
    Calculate the rest capacity of the route
    """

    demand = 0
    for customer in route:
        demand += c_data[customer, 3]

    return capacity - demand

def cal_begin_time(route, c_data, t_data):
    """
    Calculate the begin service time of each customer in the route
    """

    b = [0] * len(route)
    for i in range(len(route)):
        if i == 0:
            b[i] = 0
            continue

        customer = route[i]
        previous_customer = route[i-1]
        arrival_time = b[i-1] + c_data[previous_customer, 6] + t_data[previous_customer, customer]
        if arrival_time < c_data[customer, 4]:
            b[i] = c_data[customer, 4]
        else:
            b[i] = arrival_time
    
    return b

def find_best_place(route, c_data, t_data, capacity, alpha1=0, alpha2=1):
    """
    For each unrouted customer, we first compute its best insertion place
    in the emerging route with smallest c1

    c_1 = alpha1 * c_11 + alpha2 * c_12
    c_11 = d_iu + d_uj - d_ij
    c_12 = b_ju - b_j
    """
    
    insertion_dict = {}
    current_capacity = cal_rest_capacity(route, c_data, capacity)
    current_begin_time = cal_begin_time(route, c_data, t_data)
    unrouted_customer = np.where(c_data[:, 8] == 0)[0]
    for u in unrouted_customer:
        best_place = -1
        best_c1 = INF

        # check capacity feasibility
        if c_data[u, 3] > current_capacity:
            continue

        for place in range(1, len(route)):
            test_route = copy.deepcopy(route)
            test_route.insert(place, u)
            i = test_route[place-1]
            j = test_route[place+1]

            # check time feasibility
            is_time_feasible = True
            new_begin_time = cal_begin_time(test_route, c_data, t_data)
            for i in range(len(test_route)):
                if new_begin_time[i] > c_data[test_route[i], 5]:
                    is_time_feasible = False
                    break
            if not is_time_feasible:
                continue

            c11 = t_data[i, u] + t_data[u, j] - t_data[i, j]
            c12 = new_begin_time[place+1] - current_begin_time[place]
            c1 = alpha1 * c11 + alpha2 * c12
            if c1 < best_c1:
                best_place = place
                best_c1 = c1
                insertion_dict[u] = (best_place, best_c1)
    
    return insertion_dict

def insert_best_customer(route, c_data, t_data, insertion_dict, ld=2):
    """
    The best customer to be inserted in the route is selected as the one which
    has the biggest c_2

    c_2 = lambda * d_0u - c_1
    """

    best_customer = -1
    best_c2 = -INF
    for u, value in insertion_dict.items():
        c2 = ld * t_data[0, u] - value[1]
        if c2 > best_c2:
            best_customer = u
            best_c2 = c2
    
    best_place = insertion_dict[best_customer][0]
    route.insert(best_place, best_customer)
    c_data[best_customer, 8] = 1

def insertion(c_data, t_data, capacity):
    """
    Sequential, route-building heuristic. Use two criteria at every iteration to insert a
    new customer into the partial route.
    """

    route_list = []
    unrouted_num = len(np.where(c_data[:, 8] == 0)[0])
    while(unrouted_num > 0):
        route = init_route(c_data)
        is_insertion_feasible = True
        while(is_insertion_feasible):
            insertion_dict = find_best_place(route, c_data, t_data, capacity)
            if not insertion_dict:
                is_insertion_feasible = False
            else:
                insert_best_customer(route, c_data, t_data, insertion_dict)
        
        route_list.append(route)
        unrouted_num = len(np.where(c_data[:, 8] == 0)[0])
    
    return route_list
