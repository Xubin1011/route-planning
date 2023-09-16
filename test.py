import random
import sys

from nearest_location import nearest_location
from consumption_duration import consumption_duration
from consumption_duration import haversine
from way_calculation import way
from environment_n_pois import rp_env

env = rp_env()
x = 49.12345
y = 9.67890
loop = env.check_loop(x,y)
print(loop)