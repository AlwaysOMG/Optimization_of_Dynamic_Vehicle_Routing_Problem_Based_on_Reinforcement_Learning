import time

OVERTIME_PENALTY = 1e3
CAPACITY_PENALTY = 1e10
INF = 1e12

def cal_rest_capacity(route, c_data, capacity):
    """
    Calculate the rest capacity of the vehicle.
    """

    demand = 0
    for customer in route:
        demand += c_data[customer, 3]

    return capacity - demand

def cal_begin_time(route, c_data, t_data):
    """
    Calculate the begin service time of each customer in a route.
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

def cal_cost(route, c_data, t_data, capacity):
    """
    Calculate the cost for a route.
    
    Cost = total dist + overtime penalty
    """
    
    # check capacity
    if cal_rest_capacity(route, c_data, capacity) < 0:
        return CAPACITY_PENALTY

    cost = 0
    dist = 0
    time = 0
    current_node = 0
    for customer in route:
        dist += t_data[current_node, customer]
        time += t_data[current_node, customer]
        if time < c_data[customer, 4]:
            # wait until the earliest service time
            time = c_data[customer, 4]
        elif time > c_data[customer, 5]:
            # overtime
            cost += (time - c_data[customer, 5]) * OVERTIME_PENALTY
        time += c_data[customer, 6]
        current_node = customer

    cost += dist
    return cost

def check_customer(routes, str=None):
    flattened_list = [item for sublist in routes for item in sublist]
    filtered_list = list(filter(lambda x: x != 0, flattened_list))
    sort_list = sorted(filtered_list)
    if sort_list != list(range(1, 101)):
        print(str)
        print(len(sort_list))
        print(sort_list)
        time.sleep(10)