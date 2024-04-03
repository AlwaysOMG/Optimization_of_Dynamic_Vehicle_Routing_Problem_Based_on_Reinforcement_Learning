from dvrp.param.param_generator import ParamGenerator
from dvrp.dvrp import DVRP

a = ParamGenerator()
p = a.generate_parameter()
env = DVRP(p)