from dvrp.dvrp import DVRP
from dvrp.route_manager import RouteManager
from model.new_dynamic_attention_model.dynamic_attention_model import DynamicAttentionModel

env = DVRP()
mgr = RouteManager(env)
model = DynamicAttentionModel(mgr.get_feature_dim())

obs = env.reset()
while True:
    obs_tensor, obs_info = mgr.obs_to_tensor(obs)
    action, prob = model(obs_tensor, obs_info, True)
    route = mgr.action_to_route(action)
    obs, reward, done = env.step(route)

    if done:
        break
