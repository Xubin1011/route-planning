from deployment import visualization
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import sys
import folium
from env_deploy import rp_env
from way_noloops import way
from global_var import initial_data_p, initial_data_ch, data_p, data_ch


env = rp_env()
myway = way()
try_number = 43
cs_path = "cs_combo_bbox.csv"
p_path = "parking_bbox.csv"
route_path = f"G:\Tuning_results\dqn_route_43.csv"
map_name = f"G:\Tuning_results\dqn_route_test.html"
visualization(cs_path, p_path, route_path, myway.x_source, myway.y_source, myway.x_target, myway.y_target, map_name)

