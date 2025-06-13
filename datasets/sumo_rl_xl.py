'''
##############################################################
# Created Date: Monday, June 9th 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''

from __future__ import absolute_import

import argparse
import os
import sys
from datetime import datetime


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy


import sumo_rl
env = sumo_rl.parallel_env(net_file=r'datasets\resco\grid4x4.net.xml',
                           route_file=r'datasets\resco\grid4x4_1.rou.xml',
                           use_gui=False,
                           num_seconds=3600)
observations = env.reset()
while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
