import configparser
import matplotlib
import matplotlib.pyplot as plt

class Plotter:
    config = configparser.ConfigParser()
    config.read("./config.cfg")
    map_size = int(config["instance"]["map_size"])
    vehicle_num = int(config["instance"]["vehicle_num"])
    customer_num = int(config["instance"]["customer_num"])

    def __init__(self):
        plt.rcParams['font.sans-serif'] = ['DFKai-SB']
        self.color_dict = {-1:'orange', 0:'green', 1:'red'}
        self.marker_dict = {False:'o', True:'*'}
        self.vehicle_route_color = ['yellow', 'magenta', 'deepskyblue', 'navy', 'springgreen']

    def plot(self, instance_num, node_data, service_status, vehicle_route):
        plt.figure(figsize=(10, 10))
        axes = plt.gca()
        plt.xlim(0, self.map_size)
        plt.ylim(0, self.map_size)

        # plot node
        node_loc = [(c[0], c[1]) for c in node_data]
        
        for i, loc in enumerate(node_loc):
            if i == 0:
                plt.scatter(loc[0], loc[1], color='black')
                continue

            is_time_window_tight = True if node_data[i][4] - node_data[i][3] <= 10 else False
            plt.scatter(loc[0], loc[1], s=80, color=self.color_dict[service_status[i]], 
                        marker=self.marker_dict[is_time_window_tight])

        # plot route
        for i, route in enumerate(vehicle_route):
            route_node_loc = [node_loc[node_id] for node_id in route]
            vehicle_route_x, vehicle_route_y = zip(*route_node_loc)
            plt.plot(vehicle_route_x, vehicle_route_y, color=self.vehicle_route_color[i], 
                     linewidth=2, label=f"車輛途程 {i+1}")

        plt.legend(loc='upper right', fontsize='16', bbox_to_anchor=(1.3, 1.02))
        plt.grid(True)
        plt.xticks(size=10)
        plt.yticks(size=10)
        plt.savefig(f"./output/route_pic/instance_{instance_num}.jpg", format='jpg', bbox_inches='tight', dpi=500)