from dvrp.dvrp import DVRP
from dvrp.route_manager import RouteManager
from model.new_dynamic_attention_model.dynamic_attention_model import DynamicAttentionModel

env = DVRP()
mgr = RouteManager(env)
model = DynamicAttentionModel(mgr.get_feature_dim())

obs = env.reset()
while not env.check_done():
    obs_tensor, obs_info = mgr.obs_to_tensor(obs)
    model.set_info(obs_info)
    output = model(obs_tensor)
    print(output)
    route = mgr.list_to_route(output)
    obs, reward, done = env.step(route)
