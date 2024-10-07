#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 14:24:08 2024

@author: juan
"""

import gymnasium as gym
from gymnasium.spaces import *
from gymnasium import envs
import sys

def get_games_list():
    env_names = [env[0] for env in envs.registry.items()]
    return env_names

def print_spaces(space):
    print(space)
    if isinstance(space, Box):
        print("\n Cota inferior ", space.low)
        print("\n Cota superior ", space.high)
        
        
if __name__ == "__main__":
    environment = gym.make(sys.argv[1])
    print("Espacio de observaciones:")
    print_spaces(environment.observation_space)
    print("Espacio de acciones: ")
    print_spaces(environment.action_space)
    try:
        print("Descripción de las acciones: ", environment.unwrapped.get_action_meanings())
    except AttributeError:
        pass
    

envs = get_games_list()