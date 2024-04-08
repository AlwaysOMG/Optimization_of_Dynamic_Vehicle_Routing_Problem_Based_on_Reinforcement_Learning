from dvrp.object.route import Route

class RouteManager:
    def __init__(self, dvrp):
        self.dvrp = dvrp

    def trans_route(self, matrix):
        route_list = []
        for row in matrix:
            r = Route()
            for id in row:
                node = self.dvrp.get_node(id)
                r.add_node(node)
            route_list.append(r)
        return route_list

    def get_info(self):
        pass